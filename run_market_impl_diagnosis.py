"""市場暗黙複勝確率の試算

現行式 (3.0/odds) と Plackett-Luce ベースの厳密計算を比較し、
どの程度のバイアスが発生しているか可視化する。
"""

import numpy as np
import pandas as pd

from src.features.build_features import build_all_features

# JRA 単勝の控除率
TAKEOUT = 0.20


def market_implied_top3(race_df: pd.DataFrame) -> dict:
    """単勝オッズから Plackett-Luce で各馬の市場 top3 確率を計算する

    手順:
      1. 単勝オッズ → 市場勝率 (1/odds)
      2. 控除率を考慮して合計1.0に正規化
      3. 各馬を Plackett-Luce の strength として扱い wide_prob で top3 確率算出
         (wide_prob(A,B) の合計は P(A in top3) に一致する論理ではなく、
          直接 trifecta_prob の総和で計算する方が正確)
    """
    valid = race_df[race_df["odds"].notna() & (race_df["odds"] > 0)]
    if len(valid) < 3:
        return {}
    # 市場勝率（1/odds 合計は 1/(1-takeout) ≒ 1.25 になるので正規化）
    raw = 1.0 / valid["odds"].values
    norm = raw / raw.sum()  # 合計1に正規化
    horses = valid["post_number"].astype(int).tolist()
    strengths = {h: float(s) for h, s in zip(horses, norm)}

    # 各馬の top3 確率 = Σ_{B,C != A} trifecta(A,B,C) の全 (B,C) 組み合わせ
    # 実装簡略化: P(A in top3) = P(A=1) + P(A=2) + P(A=3)
    # P(A=1) = w_A
    # P(A=2) = Σ_B≠A w_B × (w_A / (1 - w_B))
    # P(A=3) = Σ_B,C distinct ≠A w_B × (w_C/(1-w_B)) × (w_A/(1-w_B-w_C))
    result = {}
    for a in horses:
        w_a = strengths[a]
        # P(A=1)
        p1 = w_a
        # P(A=2)
        p2 = 0.0
        for b in horses:
            if b == a:
                continue
            w_b = strengths[b]
            denom = 1.0 - w_b
            if denom > 0:
                p2 += w_b * w_a / denom
        # P(A=3)
        p3 = 0.0
        for b in horses:
            if b == a:
                continue
            w_b = strengths[b]
            d1 = 1.0 - w_b
            if d1 <= 0:
                continue
            for c in horses:
                if c == a or c == b:
                    continue
                w_c = strengths[c]
                d2 = 1.0 - w_b - w_c
                if d2 <= 0:
                    continue
                p3 += w_b * (w_c / d1) * (w_a / d2)
        result[a] = p1 + p2 + p3
    return result


def main():
    print("[1/3] 特徴量構築...")
    df = build_all_features()

    # テスト期間 (直近20%) に絞る
    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * 0.8)
    test_df = df_sorted.iloc[split_idx:].copy()
    test_df = test_df[test_df["odds"].notna() & (test_df["odds"] > 0)].copy()

    print(f"[2/3] {test_df['race_id'].nunique()} レース × {len(test_df)} 頭を処理...")

    records = []
    for race_id, g in test_df.groupby("race_id"):
        if len(g) < 3:
            continue
        market_top3 = market_implied_top3(g)
        if not market_top3:
            continue
        for _, row in g.iterrows():
            post = int(row["post_number"])
            if post not in market_top3:
                continue
            odds = float(row["odds"])
            old = min(1.0, 3.0 / odds)  # 現行式
            new = market_top3[post]     # 新計算
            records.append({
                "race_id": race_id,
                "post": post,
                "odds": odds,
                "finish_position": row.get("finish_position"),
                "is_hit_top3": 1 if (pd.notna(row.get("finish_position")) and row["finish_position"] <= 3) else 0,
                "old_market_prob": old,
                "new_market_prob": new,
                "diff": old - new,
            })

    out = pd.DataFrame(records)
    print("[3/3] 集計...")

    # オッズ帯別の比較
    out["odds_bin"] = pd.cut(
        out["odds"],
        bins=[0, 1.5, 2, 3, 5, 10, 20, 50, 100, 1000],
        labels=["~1.5", "1.5~2", "2~3", "3~5", "5~10", "10~20", "20~50", "50~100", "100+"],
    )

    summary = out.groupby("odds_bin", observed=True).agg(
        n=("odds", "size"),
        actual_top3_rate=("is_hit_top3", "mean"),
        old_prob=("old_market_prob", "mean"),
        new_prob=("new_market_prob", "mean"),
    )
    summary["old_error"] = summary["old_prob"] - summary["actual_top3_rate"]
    summary["new_error"] = summary["new_prob"] - summary["actual_top3_rate"]
    summary["actual_top3_rate"] *= 100
    summary["old_prob"] *= 100
    summary["new_prob"] *= 100
    summary["old_error"] *= 100
    summary["new_error"] *= 100

    print("\n" + "=" * 80)
    print("  オッズ帯別: 現行式 vs 新計算 vs 実績 top3 率")
    print("=" * 80)
    print(summary.to_string(float_format=lambda x: f"{x:7.1f}"))

    # 全体誤差（RMSE）
    actual = out["is_hit_top3"].values
    rmse_old = np.sqrt(np.mean((out["old_market_prob"].values - actual) ** 2))
    rmse_new = np.sqrt(np.mean((out["new_market_prob"].values - actual) ** 2))
    # Log-loss (clip で計算安定化)
    eps = 1e-6
    old_p = np.clip(out["old_market_prob"].values, eps, 1 - eps)
    new_p = np.clip(out["new_market_prob"].values, eps, 1 - eps)
    ll_old = -np.mean(actual * np.log(old_p) + (1 - actual) * np.log(1 - old_p))
    ll_new = -np.mean(actual * np.log(new_p) + (1 - actual) * np.log(1 - new_p))

    print("\n" + "=" * 80)
    print("  全体精度 (vs 実績の top3 フラグ)")
    print("=" * 80)
    print(f"  RMSE    : 現行 {rmse_old:.4f}  /  新 {rmse_new:.4f}  "
          f"(改善 {(rmse_old - rmse_new) / rmse_old * 100:+.1f}%)")
    print(f"  LogLoss : 現行 {ll_old:.4f}  /  新 {ll_new:.4f}  "
          f"(改善 {(ll_old - ll_new) / ll_old * 100:+.1f}%)")

    # 差分の分布
    print("\n" + "=" * 80)
    print("  現行 − 新 の差分分布 (正 = 現行が過大評価)")
    print("=" * 80)
    print(f"  平均  : {out['diff'].mean()*100:+.2f}pt")
    print(f"  中央値: {out['diff'].median()*100:+.2f}pt")
    print(f"  P10   : {out['diff'].quantile(0.10)*100:+.2f}pt")
    print(f"  P25   : {out['diff'].quantile(0.25)*100:+.2f}pt")
    print(f"  P75   : {out['diff'].quantile(0.75)*100:+.2f}pt")
    print(f"  P90   : {out['diff'].quantile(0.90)*100:+.2f}pt")

    out.to_csv("market_impl_diagnosis.csv", index=False, encoding="utf-8-sig")
    print("\n→ 明細を market_impl_diagnosis.csv に保存")


if __name__ == "__main__":
    main()
