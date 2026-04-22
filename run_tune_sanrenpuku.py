"""3連複 上位3頭BOX × v6 の詰め作業

事前仮説のフィルタを順に適用し、ROI > 100% を狙う。
過学習対策: テスト期間を前半/後半に分けて再現性も確認する。
"""

import numpy as np
import pandas as pd
from itertools import combinations

from config.settings import DB_PATH
from src.features.build_features import build_all_features
from src.model.train import load_model, prepare_dataset, get_available_features
from src.evaluation.multi_bet_backtest import _load_combo_payouts, _sorted_combo


def build_race_level_df(model_name="lightgbm_v6", test_frac=0.2):
    """レース単位のメタ情報 + 上位3頭BOX の配当情報を作る"""
    print(f"[1/3] 特徴量構築 & {model_name} 読込...")
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    model = load_model(model_name)

    split_idx = int(len(df) * (1 - test_frac))
    test_df = df.iloc[split_idx:].copy()
    test_df["pred_prob"] = model.predict_proba(test_df[features])[:, 1]

    race_ids = test_df["race_id"].unique().tolist()
    print(f"[2/3] 払戻読込 ({len(race_ids)} races)...")
    payouts_map = _load_combo_payouts(race_ids, "sanrenpuku", DB_PATH)

    print(f"[3/3] レース単位集計...")
    rows = []
    for race_id, g in test_df.groupby("race_id"):
        if len(g) < 3:
            continue
        top3 = g.nlargest(3, "pred_prob")
        probs = top3["pred_prob"].values
        horses = top3["post_number"].astype(int).tolist()
        combo_key = _sorted_combo(horses)
        pay = payouts_map.get(race_id, {}).get(combo_key, 0) / 100.0  # 倍率

        # 1番人気(最小odds)の馬番
        valid_odds = g[g["odds"].notna() & (g["odds"] > 0)]
        fav_post = None
        if len(valid_odds) > 0:
            fav_post = int(valid_odds.loc[valid_odds["odds"].idxmin(), "post_number"])
        fav_in_top3 = fav_post in horses if fav_post is not None else False

        # 上位3頭のオッズ合計（荒れ度）
        top3_odds = top3["odds"].fillna(999).values
        top3_odds_sum = float(np.sum(top3_odds))

        rows.append({
            "race_id": race_id,
            "date": g["date"].iloc[0],
            "venue": g["venue"].iloc[0],
            "head_count": len(g),
            "top1_prob": probs[0],
            "top2_prob": probs[1],
            "top3_prob": probs[2],
            "sum_top3_prob": float(np.sum(probs)),
            "prob_gap_1_3": float(probs[0] - probs[2]),
            "fav_in_top3": fav_in_top3,
            "top3_odds_sum": top3_odds_sum,
            "payout": pay,   # 外れなら 0
            "is_hit": pay > 0,
        })
    return pd.DataFrame(rows)


def _split_half(race_df):
    """テスト期間を前半/後半に分割"""
    race_df = race_df.sort_values("date").reset_index(drop=True)
    mid = len(race_df) // 2
    return race_df.iloc[:mid], race_df.iloc[mid:]


def _evaluate(name, filtered, full_n):
    if len(filtered) == 0:
        return {"name": name, "bets": 0, "hit_rate": 0.0, "roi": 0.0, "coverage": 0.0}
    hit = filtered["is_hit"].sum()
    roi = filtered["payout"].sum() / len(filtered) * 100
    return {
        "name": name,
        "bets": len(filtered),
        "hits": int(hit),
        "hit_rate": hit / len(filtered) * 100,
        "roi": roi,
        "coverage": len(filtered) / full_n * 100,
    }


def _print_result(base, first, second, full_n, n1, n2):
    print(f"\n  {base['name']}")
    print(f"    全期間 : {base['bets']:>5}/{full_n} ({base['coverage']:5.1f}%) "
          f"hit={base['hit_rate']:5.1f}%  ROI={base['roi']:6.1f}%")
    print(f"    前半   : {first['bets']:>5}/{n1} ({first['coverage']:5.1f}%) "
          f"hit={first['hit_rate']:5.1f}%  ROI={first['roi']:6.1f}%")
    print(f"    後半   : {second['bets']:>5}/{n2} ({second['coverage']:5.1f}%) "
          f"hit={second['hit_rate']:5.1f}%  ROI={second['roi']:6.1f}%")


def main():
    race_df = build_race_level_df()
    first_half, second_half = _split_half(race_df)
    N, N1, N2 = len(race_df), len(first_half), len(second_half)

    print(f"\n対象: {N} レース ({race_df['date'].min()} 〜 {race_df['date'].max()})")
    print(f"前半: {N1} / 後半: {N2}")

    # ベースライン
    print("\n" + "=" * 70)
    print("【ベースライン】フィルタなし")
    print("=" * 70)
    base = _evaluate("base", race_df, N)
    first = _evaluate("base", first_half, N1)
    second = _evaluate("base", second_half, N2)
    _print_result(base, first, second, N, N1, N2)

    # 仮説1: pred_prob 絶対値フィルタ
    print("\n" + "=" * 70)
    print("【仮説1】3位馬の pred_prob が閾値以上（自信のあるレース限定）")
    print("=" * 70)
    for th in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
        sub = race_df[race_df["top3_prob"] >= th]
        sub1 = first_half[first_half["top3_prob"] >= th]
        sub2 = second_half[second_half["top3_prob"] >= th]
        _print_result(
            _evaluate(f"top3_prob >= {th}", sub, N),
            _evaluate(f"top3_prob >= {th}", sub1, N1),
            _evaluate(f"top3_prob >= {th}", sub2, N2), N, N1, N2)

    # 仮説2: 上位3頭の確率合計
    print("\n" + "=" * 70)
    print("【仮説2】上位3頭の pred_prob 合計（堅そうなレース）")
    print("=" * 70)
    for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = race_df[race_df["sum_top3_prob"] >= th]
        sub1 = first_half[first_half["sum_top3_prob"] >= th]
        sub2 = second_half[second_half["sum_top3_prob"] >= th]
        _print_result(
            _evaluate(f"sum_top3 >= {th}", sub, N),
            _evaluate(f"sum_top3 >= {th}", sub1, N1),
            _evaluate(f"sum_top3 >= {th}", sub2, N2), N, N1, N2)

    # 仮説3: 1番人気が AI Top3 に含まれる / 含まれない
    print("\n" + "=" * 70)
    print("【仮説3】1番人気と AI Top3 の整合性")
    print("=" * 70)
    for include in [True, False]:
        label = "1番人気 in Top3" if include else "1番人気 not in Top3"
        sub = race_df[race_df["fav_in_top3"] == include]
        sub1 = first_half[first_half["fav_in_top3"] == include]
        sub2 = second_half[second_half["fav_in_top3"] == include]
        _print_result(
            _evaluate(label, sub, N),
            _evaluate(label, sub1, N1),
            _evaluate(label, sub2, N2), N, N1, N2)

    # 仮説4: 頭数フィルタ
    print("\n" + "=" * 70)
    print("【仮説4】頭数フィルタ")
    print("=" * 70)
    for (lo, hi) in [(8, 12), (10, 14), (13, 18), (14, 18), (16, 18)]:
        sub = race_df[(race_df["head_count"] >= lo) & (race_df["head_count"] <= hi)]
        sub1 = first_half[(first_half["head_count"] >= lo) & (first_half["head_count"] <= hi)]
        sub2 = second_half[(second_half["head_count"] >= lo) & (second_half["head_count"] <= hi)]
        _print_result(
            _evaluate(f"head_count {lo}-{hi}", sub, N),
            _evaluate(f"head_count {lo}-{hi}", sub1, N1),
            _evaluate(f"head_count {lo}-{hi}", sub2, N2), N, N1, N2)

    # 仮説5: 組合せ（自信ありレース × 1番人気Top3入り）
    print("\n" + "=" * 70)
    print("【仮説5】複合: top3_prob >= 0.20 AND 1番人気∈Top3")
    print("=" * 70)
    mask = (race_df["top3_prob"] >= 0.20) & (race_df["fav_in_top3"])
    mask1 = (first_half["top3_prob"] >= 0.20) & (first_half["fav_in_top3"])
    mask2 = (second_half["top3_prob"] >= 0.20) & (second_half["fav_in_top3"])
    _print_result(
        _evaluate("複合A", race_df[mask], N),
        _evaluate("複合A", first_half[mask1], N1),
        _evaluate("複合A", second_half[mask2], N2), N, N1, N2)

    print("\n【仮説5b】複合: sum_top3 >= 0.65 AND 1番人気∈Top3")
    mask = (race_df["sum_top3_prob"] >= 0.65) & (race_df["fav_in_top3"])
    mask1 = (first_half["sum_top3_prob"] >= 0.65) & (first_half["fav_in_top3"])
    mask2 = (second_half["sum_top3_prob"] >= 0.65) & (second_half["fav_in_top3"])
    _print_result(
        _evaluate("複合B", race_df[mask], N),
        _evaluate("複合B", first_half[mask1], N1),
        _evaluate("複合B", second_half[mask2], N2), N, N1, N2)

    # 仮説6: 本命フィルタ - head_count >= 13 AND 1番人気 in Top3
    print("\n" + "=" * 70)
    print("【仮説6】本命候補: head_count >= 13 AND 1番人気 in Top3")
    print("=" * 70)
    mask = (race_df["head_count"] >= 13) & (race_df["fav_in_top3"])
    mask1 = (first_half["head_count"] >= 13) & (first_half["fav_in_top3"])
    mask2 = (second_half["head_count"] >= 13) & (second_half["fav_in_top3"])
    _print_result(
        _evaluate("多頭数×市場整合", race_df[mask], N),
        _evaluate("多頭数×市場整合", first_half[mask1], N1),
        _evaluate("多頭数×市場整合", second_half[mask2], N2), N, N1, N2)

    # 仮説6b: さらに頭数帯別に細分化
    print("\n【仮説6b】頭数帯×1番人気in Top3 の細分")
    for (lo, hi) in [(13, 15), (14, 16), (15, 18), (13, 18), (16, 18)]:
        mask = (race_df["head_count"].between(lo, hi)) & (race_df["fav_in_top3"])
        mask1 = (first_half["head_count"].between(lo, hi)) & (first_half["fav_in_top3"])
        mask2 = (second_half["head_count"].between(lo, hi)) & (second_half["fav_in_top3"])
        _print_result(
            _evaluate(f"head {lo}-{hi} & fav in Top3", race_df[mask], N),
            _evaluate(f"head {lo}-{hi} & fav in Top3", first_half[mask1], N1),
            _evaluate(f"head {lo}-{hi} & fav in Top3", second_half[mask2], N2), N, N1, N2)

    # 仮説7: AI自信度 + 多頭数
    print("\n" + "=" * 70)
    print("【仮説7】AI自信度 × 多頭数")
    print("=" * 70)
    for tp, hc in [(0.15, 13), (0.18, 13), (0.20, 13), (0.15, 14), (0.18, 14)]:
        mask = (race_df["top3_prob"] >= tp) & (race_df["head_count"] >= hc) & (race_df["fav_in_top3"])
        mask1 = (first_half["top3_prob"] >= tp) & (first_half["head_count"] >= hc) & (first_half["fav_in_top3"])
        mask2 = (second_half["top3_prob"] >= tp) & (second_half["head_count"] >= hc) & (second_half["fav_in_top3"])
        _print_result(
            _evaluate(f"top3>={tp} & head>={hc} & fav in Top3", race_df[mask], N),
            _evaluate(f"top3>={tp} & head>={hc} & fav in Top3", first_half[mask1], N1),
            _evaluate(f"top3>={tp} & head>={hc} & fav in Top3", second_half[mask2], N2), N, N1, N2)

    # 月別ROI (最終候補の安定性チェック)
    print("\n" + "=" * 70)
    print("【月別ROI】 head_count >= 13 AND 1番人気 in Top3")
    print("=" * 70)
    final = race_df[(race_df["head_count"] >= 13) & (race_df["fav_in_top3"])].copy()
    final["month"] = pd.to_datetime(final["date"]).dt.to_period("M").astype(str)
    monthly = final.groupby("month").agg(
        bets=("is_hit", "size"),
        hits=("is_hit", "sum"),
        roi=("payout", lambda s: s.sum() / len(s) * 100),
    )
    monthly["hit_rate"] = monthly["hits"] / monthly["bets"] * 100
    print(monthly.to_string(float_format=lambda x: f"{x:6.1f}"))
    # プラス月の比率
    positive_months = (monthly["roi"] >= 100).sum()
    print(f"\n→ 100%超えの月: {positive_months}/{len(monthly)}  "
          f"月別ROI平均={monthly['roi'].mean():.1f}%  "
          f"中央値={monthly['roi'].median():.1f}%")

    # 保存
    race_df.to_csv("sanrenpuku_race_level.csv", index=False, encoding="utf-8-sig")
    print("\n→ レース単位の詳細を sanrenpuku_race_level.csv に保存")


if __name__ == "__main__":
    main()
