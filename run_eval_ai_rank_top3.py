"""AI予測の順位 (1位〜) ごとの実 top3率・複勝ROI を比較する

目的: ユーザの体感「2番手推奨の方が3着以内率が高い気がする」を実データで確認。
モデル: v8 (最新) と v6 (比較)
"""

import pandas as pd

from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


def evaluate_ranks(model_name: str, top_n: int = 5) -> pd.DataFrame:
    """各レースで AI予測順位 1〜top_n の馬の実 top3率・単勝率・ROI を集計"""
    print(f"[Eval {model_name}] 特徴量構築...")
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    model = load_model(model_name)

    # テスト期間（最新20%）
    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()
    test = test[test["odds"].notna() & (test["odds"] > 0)].copy()
    test["pred_prob"] = model.predict_proba(test[features])[:, 1]

    # 実複勝オッズ (実データ優先 / 概算フォールバック)
    if "fukusho_odds_actual" in test.columns and test["fukusho_odds_actual"].notna().any():
        test["fukusho_odds"] = test["fukusho_odds_actual"]
        fuku_label = "実複勝"
    else:
        test["fukusho_odds"] = (test["odds"] * 0.3).clip(lower=1.1)
        fuku_label = "概算 (単勝×0.3)"

    test["is_top3"] = (test["finish_position"] <= 3).astype(int)
    test["is_win"] = (test["finish_position"] == 1).astype(int)

    # レース内 AI 順位 (pred_prob 降順)
    test["ai_rank"] = test.groupby("race_id")["pred_prob"].rank(method="first", ascending=False).astype(int)

    rows = []
    for rank in range(1, top_n + 1):
        sub = test[test["ai_rank"] == rank]
        if len(sub) == 0:
            continue
        n = len(sub)
        n_top3 = sub["is_top3"].sum()
        n_win = sub["is_win"].sum()
        fuku_payout = (sub["is_top3"] * sub["fukusho_odds"]).sum()
        win_payout = (sub["is_win"] * sub["odds"]).sum()
        rows.append({
            "AI順位": f"{rank}位",
            "サンプル数": n,
            "複勝率": n_top3 / n * 100,
            "勝率": n_win / n * 100,
            "複勝ROI": fuku_payout / n * 100,
            "単勝ROI": win_payout / n * 100,
            "平均AI確率": sub["pred_prob"].mean(),
            "平均オッズ": sub["odds"].mean(),
            "平均人気": sub["odds"].rank(method="dense").mean(),  # オッズ順位の平均
        })
    return pd.DataFrame(rows), fuku_label, len(test["race_id"].unique())


def print_table(df: pd.DataFrame, title: str):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(f"  {'AI順位':<6} {'n':>6} {'複勝率':>7} {'勝率':>6} {'複勝ROI':>9} {'単勝ROI':>9} "
          f"{'平均AI':>8} {'平均odds':>9}")
    print(f"  {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 6} {'-' * 9} {'-' * 9} "
          f"{'-' * 8} {'-' * 9}")
    for _, r in df.iterrows():
        # 1位より高い指標があるかをマークするためのリファレンス
        print(f"  {r['AI順位']:<6} {int(r['サンプル数']):>6} "
              f"{r['複勝率']:>6.1f}% {r['勝率']:>5.1f}% "
              f"{r['複勝ROI']:>7.1f}%  {r['単勝ROI']:>7.1f}%  "
              f"{r['平均AI確率']*100:>6.1f}% {r['平均オッズ']:>8.1f}")


def cross_with_odds_band(model_name: str = "lightgbm_v8") -> pd.DataFrame:
    """AI順位 × オッズ帯 のクロス集計"""
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    model = load_model(model_name)

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()
    test = test[test["odds"].notna() & (test["odds"] > 0)].copy()
    test["pred_prob"] = model.predict_proba(test[features])[:, 1]

    if "fukusho_odds_actual" in test.columns and test["fukusho_odds_actual"].notna().any():
        test["fukusho_odds"] = test["fukusho_odds_actual"]
    else:
        test["fukusho_odds"] = (test["odds"] * 0.3).clip(lower=1.1)
    test["is_top3"] = (test["finish_position"] <= 3).astype(int)
    test["ai_rank"] = test.groupby("race_id")["pred_prob"].rank(method="first", ascending=False).astype(int)

    bands = [(1, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 999)]

    print("\n" + "=" * 90)
    print(f"  【{model_name}】AI順位 × オッズ帯 クロス集計 (複勝率 / 複勝ROI%)")
    print("=" * 90)

    header = f"  {'AI順位':<6} "
    for lo, hi in bands:
        label = f"{lo:.0f}-{hi:.0f}" if hi < 999 else f"{lo:.0f}+"
        header += f"{label:>14} "
    print(header)
    print(f"  {'-' * 6} " + " ".join(["-" * 14] * len(bands)))

    for rank in range(1, 6):
        line = f"  {rank}位:    "
        for lo, hi in bands:
            sub = test[(test["ai_rank"] == rank) & (test["odds"] >= lo) & (test["odds"] < hi)]
            if len(sub) < 30:
                line += f"{'(n<30)':>14} "
                continue
            top3 = sub["is_top3"].mean() * 100
            roi = (sub["is_top3"] * sub["fukusho_odds"]).sum() / len(sub) * 100
            line += f"{top3:>5.1f}%/{roi:>5.0f}% ".rjust(14) + " "
        print(line)


def main():
    # ── v8 でランク別評価 ──
    df_v8, fuku_label, n_races = evaluate_ranks("lightgbm_v8", top_n=8)
    print(f"\n対象: 約 {n_races} レース ({fuku_label})")
    print_table(df_v8, "v8: AI順位別 集計（テスト期間 = 最新20%）")

    # ── v6 でランク別評価（比較用） ──
    try:
        df_v6, _, _ = evaluate_ranks("lightgbm_v6", top_n=8)
        print_table(df_v6, "v6: AI順位別 集計（比較用）")
    except Exception as e:
        print(f"\n⚠️ v6 比較スキップ: {e}")

    # ── 1位 vs 2位 の差分サマリー ──
    print("\n" + "=" * 90)
    print("  📊 1位 vs 2位 比較サマリー（v8）")
    print("=" * 90)
    rank1 = df_v8.iloc[0]
    rank2 = df_v8.iloc[1]
    print(f"  {'指標':<14} {'1位':>10} {'2位':>10} {'差 (2位 - 1位)':>18}")
    print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 18}")
    for col in ["複勝率", "勝率", "複勝ROI", "単勝ROI", "平均AI確率", "平均オッズ"]:
        v1, v2 = rank1[col], rank2[col]
        diff = v2 - v1
        sign = "+" if diff >= 0 else ""
        marker = "🟢 2位優位" if diff > 0 else ("🔴 1位優位" if diff < 0 else "⚪ 同じ")
        if "確率" in col:
            print(f"  {col:<14} {v1*100:>9.2f}% {v2*100:>9.2f}% {sign}{diff*100:>+9.2f}pt  {marker}")
        elif "ROI" in col or "率" in col:
            print(f"  {col:<14} {v1:>9.2f}% {v2:>9.2f}% {sign}{diff:>+9.2f}pt  {marker}")
        else:
            print(f"  {col:<14} {v1:>9.2f}  {v2:>9.2f}  {sign}{diff:>+9.2f}    {marker}")

    # ── オッズ帯別クロス ──
    cross_with_odds_band("lightgbm_v8")

    # CSV
    df_v8.to_csv("ai_rank_eval_v8.csv", index=False, encoding="utf-8-sig")
    print("\n→ ai_rank_eval_v8.csv に保存")


if __name__ == "__main__":
    main()
