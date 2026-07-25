"""当日保存した予測（prediction_snapshots）で実運用の性能を測定する

バックテストとの決定的な違い:
  バックテスト = build_features 経路で「後から」計算した予測を評価
  本スクリプト = その日に実際に出した予測（当日オッズ込み）をそのまま評価
                 → 真の out-of-sample 性能

測定内容:
  1. AUC / Log Loss  … バックテスト値(v8: 0.7901)と比較
  2. キャリブレーション … AI確率が実際の的中率と一致しているか
  3. AI順位別の複勝率  … 「1位の複勝率が60%前後しかない」の検証
  4. 日付別のブレ      … サンプル数による揺らぎの範囲
"""

import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from config.settings import DB_PATH


def load_live_data() -> pd.DataFrame:
    """保存済み予測 + 実結果 を結合して返す"""
    with sqlite3.connect(DB_PATH) as conn:
        snap = pd.read_sql_query("SELECT * FROM prediction_snapshots", conn)
        res = pd.read_sql_query(
            "SELECT race_id, post_number, finish_position, popularity FROM results", conn
        )
    snap["race_id"] = snap["race_id"].astype(str)
    res["race_id"] = res["race_id"].astype(str)

    df = snap.merge(res, on=["race_id", "post_number"], how="inner")
    df = df[df["finish_position"].notna() & (df["finish_position"] > 0)].copy()
    df["is_top3"] = (df["finish_position"] <= 3).astype(int)
    df["is_win"] = (df["finish_position"] == 1).astype(int)
    # レース内のAI順位
    df["ai_rank"] = df.groupby("race_id")["pred_prob"].rank(
        method="first", ascending=False
    ).astype(int)
    return df


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項分布の95%信頼区間（Wilson score interval）"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - hw) / d * 100, (c + hw) / d * 100)


def main():
    df = load_live_data()
    if len(df) == 0:
        print("保存済み予測がありません。予測ページで『💾 この予測をDBに保存』を実行してください。")
        return

    models = df["model_name"].unique().tolist()
    print("=" * 78)
    print("  当日予測の実運用性能（保存済みスナップショット）")
    print("=" * 78)
    print(f"  対象: {df['race_date'].nunique()}開催日 / {df['race_id'].nunique()}レース / "
          f"{len(df):,}頭  ({df['race_date'].min()} 〜 {df['race_date'].max()})")
    print(f"  モデル: {', '.join(models)}")
    print()

    # ── 1. AUC / Log Loss ──
    y = df["is_top3"]
    p = df["pred_prob"].clip(0.001, 0.999)
    auc = roc_auc_score(y, p)
    ll = log_loss(y, p)
    print("─" * 78)
    print("  【1】全体精度  ※ バックテスト値: v8 AUC=0.7901 / LogLoss=0.4296")
    print("─" * 78)
    print(f"  実運用 AUC     : {auc:.4f}   （バックテスト比 {auc - 0.7901:+.4f}）")
    print(f"  実運用 LogLoss : {ll:.4f}   （バックテスト比 {ll - 0.4296:+.4f}）")
    print(f"  実際の3着内率  : {y.mean() * 100:.1f}%   （AI確率の平均: {p.mean() * 100:.1f}%）")

    # ── 2. キャリブレーション ──
    print()
    print("─" * 78)
    print("  【2】キャリブレーション（AI確率 vs 実際の的中率）")
    print("─" * 78)
    bins = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.01]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"]
    df["_bin"] = pd.cut(df["pred_prob"], bins=bins, labels=labels, right=False)
    print(f"  {'確率帯':<8} {'頭数':>6} {'AI予測':>8} {'実際':>8} {'差':>8}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for lb in labels:
        sub = df[df["_bin"] == lb]
        if len(sub) == 0:
            continue
        pred = sub["pred_prob"].mean() * 100
        act = sub["is_top3"].mean() * 100
        print(f"  {lb:<8} {len(sub):>6} {pred:>7.1f}% {act:>7.1f}% {act - pred:>+7.1f}pt")

    # ── 3. AI順位別 ──
    print()
    print("─" * 78)
    print("  【3】AI順位別の成績  ※ バックテスト値: 1位=65.4% / 2位=50.0% / 3位=41.1%")
    print("─" * 78)
    bt = {1: 65.4, 2: 50.0, 3: 41.1, 4: 33.5, 5: 24.7}
    print(f"  {'AI順位':<6} {'頭数':>5} {'成績':>12} {'複勝率':>7} {'95%信頼区間':>16} {'BT値':>7} {'判定':>8}")
    print(f"  {'-'*6} {'-'*5} {'-'*12} {'-'*7} {'-'*16} {'-'*7} {'-'*8}")
    for rank in range(1, 6):
        sub = df[df["ai_rank"] == rank]
        if len(sub) == 0:
            continue
        f = sub["finish_position"]
        n1, n2, n3 = int((f == 1).sum()), int((f == 2).sum()), int((f == 3).sum())
        n_out = len(sub) - n1 - n2 - n3
        k, n = n1 + n2 + n3, len(sub)
        rate = k / n * 100
        lo, hi = _wilson_ci(k, n)
        b = bt.get(rank)
        # バックテスト値が信頼区間に入っているか
        verdict = "整合" if b is not None and lo <= b <= hi else "乖離"
        print(f"  {rank}位   {n:>5} {f'{n1}-{n2}-{n3}-{n_out}':>12} {rate:>6.1f}% "
              f"{f'[{lo:.1f} - {hi:.1f}]':>16} {b:>6.1f}% {verdict:>8}")

    # ── 4. 日付別のブレ ──
    print()
    print("─" * 78)
    print("  【4】開催日別のブレ（AI1位の複勝率）")
    print("─" * 78)
    r1 = df[df["ai_rank"] == 1]
    print(f"  {'日付':<12} {'レース':>6} {'複勝率':>7} {'95%信頼区間':>16}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*16}")
    for d, g in r1.groupby("race_date"):
        k, n = int(g["is_top3"].sum()), len(g)
        lo, hi = _wilson_ci(k, n)
        print(f"  {d:<12} {n:>6} {k/n*100:>6.1f}% {f'[{lo:.1f} - {hi:.1f}]':>16}")

    k, n = int(r1["is_top3"].sum()), len(r1)
    lo, hi = _wilson_ci(k, n)
    print(f"  {'合計':<12} {n:>6} {k/n*100:>6.1f}% {f'[{lo:.1f} - {hi:.1f}]':>16}")

    # ── 結論 ──
    print()
    print("=" * 78)
    print("  判定")
    print("=" * 78)
    if lo <= 65.4 <= hi:
        print(f"  ✅ AI1位の複勝率 {k/n*100:.1f}% は、バックテスト値 65.4% と統計的に整合")
        print(f"     （95%信頼区間 [{lo:.1f}% - {hi:.1f}%] に 65.4% が含まれる）")
        print("     → 現在の乖離はサンプル数不足による揺らぎ。モデルは想定通り機能している")
    elif hi < 65.4:
        print(f"  ⚠️ AI1位の複勝率 {k/n*100:.1f}% は、バックテスト値 65.4% を有意に下回る")
        print(f"     （95%信頼区間 [{lo:.1f}% - {hi:.1f}%] に 65.4% が入らない）")
        print("     → 実運用性能がバックテストより低い可能性。要調査")
    else:
        print(f"  🎉 AI1位の複勝率 {k/n*100:.1f}% はバックテスト値を有意に上回る")
    print(f"\n  ※ サンプル {n} レース。信頼区間を ±3pt に狭めるには約400レース必要")
    print("=" * 78)


if __name__ == "__main__":
    main()
