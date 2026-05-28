"""3連複 上位N頭BOX の比較評価

top3 BOX (1点) vs top4 BOX (4点) vs top5 BOX (10点) を同条件で比較する。
- 既存フィルタ: head_count >= 13 AND 1番人気 in AI Top3 (※ top5 では Top5 内)
- フィルタなしのベースラインも併記
- 前半/後半に分けて再現性も確認
"""

from itertools import combinations

import pandas as pd

from config.settings import DB_PATH
from src.evaluation.multi_bet_backtest import _load_combo_payouts, _sorted_combo
from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


def build_race_level_df(model_name: str = "lightgbm_v6", test_frac: float = 0.2,
                         top_n_max: int = 5) -> pd.DataFrame:
    """レース単位で top_n_max 頭までの BOX 配当を集計する"""
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

    print(f"[3/3] レース単位集計 (top_n=3..{top_n_max})...")
    rows = []
    for race_id, g in test_df.groupby("race_id"):
        if len(g) < 3:
            continue
        race_payouts = payouts_map.get(race_id, {})
        # 上位 top_n_max 頭まで
        top_n_df = g.nlargest(top_n_max, "pred_prob")
        horses_n = top_n_df["post_number"].astype(int).tolist()

        # 1番人気
        valid_odds = g[g["odds"].notna() & (g["odds"] > 0)]
        fav_post = (
            int(valid_odds.loc[valid_odds["odds"].idxmin(), "post_number"])
            if len(valid_odds) > 0 else None
        )

        row = {
            "race_id": race_id,
            "date": g["date"].iloc[0],
            "head_count": len(g),
            "fav_post": fav_post,
        }
        # 各 N で BOX 払戻と的中フラグを記録
        for n in range(3, top_n_max + 1):
            if len(top_n_df) < n:
                row[f"top{n}_payout"] = 0.0
                row[f"top{n}_is_hit"] = False
                row[f"top{n}_n_combos"] = 0
                row[f"top{n}_fav_in"] = False
                continue
            horses = horses_n[:n]
            combos = list(combinations(horses, 3))  # C(n,3) 通り
            payout_sum = 0.0
            n_hit = 0
            for c in combos:
                p = race_payouts.get(_sorted_combo(list(c)), 0) / 100.0
                if p > 0:
                    payout_sum += p
                    n_hit += 1
            row[f"top{n}_payout"] = payout_sum  # 倍率の合計（当たれば 1〜複数のcomboがヒット）
            row[f"top{n}_is_hit"] = n_hit > 0
            row[f"top{n}_n_combos"] = len(combos)
            row[f"top{n}_fav_in"] = (fav_post is not None) and (fav_post in horses)

        rows.append(row)

    return pd.DataFrame(rows)


def _evaluate(label: str, sub: pd.DataFrame, full_n: int, n: int) -> dict:
    """N頭BOX の評価"""
    pay_col = f"top{n}_payout"
    hit_col = f"top{n}_is_hit"
    combo_col = f"top{n}_n_combos"

    if len(sub) == 0:
        return {"label": label, "bets": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0, "coverage": 0.0}

    # 賭金: 1レースあたり combo数 * 1単位（100円）
    cost_units = sub[combo_col].sum()  # 総賭金（単位）
    payout_units = sub[pay_col].sum()  # 総払戻（単位）
    hits = int(sub[hit_col].sum())
    bets = len(sub)

    return {
        "label": label,
        "bets": bets,
        "hits": hits,
        "hit_rate": hits / bets * 100 if bets > 0 else 0.0,
        "roi": payout_units / cost_units * 100 if cost_units > 0 else 0.0,
        "coverage": bets / full_n * 100 if full_n > 0 else 0.0,
    }


def _split_half(race_df: pd.DataFrame):
    race_df = race_df.sort_values("date").reset_index(drop=True)
    mid = len(race_df) // 2
    return race_df.iloc[:mid], race_df.iloc[mid:]


def _print_block(title: str, rows: list[dict]):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'戦略':<28} {'賭数':>6} {'的中':>5} {'的中率':>7} {'回収率':>7}  {'対象率':>6}")
    print(f"  {'-' * 28} {'-' * 6} {'-' * 5} {'-' * 7} {'-' * 7}  {'-' * 6}")
    for r in rows:
        print(f"  {r['label']:<28} {r['bets']:>6} {r['hits']:>5} "
              f"{r['hit_rate']:>6.1f}% {r['roi']:>6.1f}%  {r['coverage']:>5.1f}%")


def main():
    race_df = build_race_level_df(top_n_max=5)
    first, second = _split_half(race_df)
    N, N1, N2 = len(race_df), len(first), len(second)

    print(f"\n対象: {N} レース ({race_df['date'].min()} 〜 {race_df['date'].max()})")
    print(f"前半: {N1} / 後半: {N2}")

    # ─────────────────────────────────────────
    # ベースライン (フィルタなし)
    # ─────────────────────────────────────────
    rows_all = []
    rows_first = []
    rows_second = []
    for n in [3, 4, 5]:
        rows_all.append(_evaluate(f"top{n} BOX ({len(list(combinations(range(n), 3)))}点)", race_df, N, n))
        rows_first.append(_evaluate(f"top{n} BOX ({len(list(combinations(range(n), 3)))}点)", first, N1, n))
        rows_second.append(_evaluate(f"top{n} BOX ({len(list(combinations(range(n), 3)))}点)", second, N2, n))
    _print_block("【ベースライン】フィルタなし — 全期間", rows_all)
    _print_block("【ベースライン】前半", rows_first)
    _print_block("【ベースライン】後半", rows_second)

    # ─────────────────────────────────────────
    # 既存の3連複BOX採用フィルタ: head_count >= 13 AND 1番人気 in Top_N
    # ─────────────────────────────────────────
    rows_all = []
    rows_first = []
    rows_second = []
    for n in [3, 4, 5]:
        fav_col = f"top{n}_fav_in"
        mask = (race_df["head_count"] >= 13) & (race_df[fav_col])
        m1 = (first["head_count"] >= 13) & (first[fav_col])
        m2 = (second["head_count"] >= 13) & (second[fav_col])
        n_combos = len(list(combinations(range(n), 3)))
        rows_all.append(_evaluate(f"top{n} BOX ({n_combos}点)", race_df[mask], N, n))
        rows_first.append(_evaluate(f"top{n} BOX ({n_combos}点)", first[m1], N1, n))
        rows_second.append(_evaluate(f"top{n} BOX ({n_combos}点)", second[m2], N2, n))
    _print_block("【既存採用フィルタ】head_count>=13 & 1番人気 in Top_N — 全期間", rows_all)
    _print_block("【既存採用フィルタ】前半", rows_first)
    _print_block("【既存採用フィルタ】後半", rows_second)

    # ─────────────────────────────────────────
    # フィルタなし (ベースライン) のみで top5 の頭数別感度
    # ─────────────────────────────────────────
    rows = []
    for (lo, hi) in [(8, 12), (13, 15), (16, 18), (13, 18), (8, 18)]:
        mask = race_df["head_count"].between(lo, hi)
        rows.append(_evaluate(f"head {lo}-{hi}", race_df[mask], N, 5))
    _print_block("【top5 BOX (10点)】頭数帯別 (フィルタなし)", rows)

    # 1番人気 in Top5 の有無で比較
    rows = []
    rows.append(_evaluate("fav in Top5", race_df[race_df["top5_fav_in"]], N, 5))
    rows.append(_evaluate("fav NOT in Top5", race_df[~race_df["top5_fav_in"]], N, 5))
    _print_block("【top5 BOX (10点)】1番人気の Top5 内外", rows)

    # 月別ROI（top5 既存フィルタ）
    print("\n" + "=" * 78)
    print("【月別ROI】 top5 BOX (10点) × head>=13 × 1番人気 in Top5")
    print("=" * 78)
    final = race_df[(race_df["head_count"] >= 13) & (race_df["top5_fav_in"])].copy()
    if len(final) > 0:
        final["month"] = pd.to_datetime(final["date"]).dt.to_period("M").astype(str)
        agg = final.groupby("month").apply(
            lambda g: pd.Series({
                "bets": len(g),
                "hits": int(g["top5_is_hit"].sum()),
                "hit_rate": g["top5_is_hit"].mean() * 100,
                "roi": g["top5_payout"].sum() / (len(g) * 10) * 100,
            })
        )
        print(agg.to_string(float_format=lambda x: f"{x:6.1f}"))
        positive = (agg["roi"] >= 100).sum()
        print(f"\n→ 100%超えの月: {positive}/{len(agg)}  "
              f"月別ROI平均={agg['roi'].mean():.1f}%  "
              f"中央値={agg['roi'].median():.1f}%")

    race_df.to_csv("sanrenpuku_top5_race_level.csv", index=False, encoding="utf-8-sig")
    print("\n→ sanrenpuku_top5_race_level.csv に保存")


if __name__ == "__main__":
    main()
