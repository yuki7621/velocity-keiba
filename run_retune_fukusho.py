"""新しい市場暗黙確率（Plackett-Luce）での複勝バリューベット再チューニング

旧「3/odds」式を新計算に置き換えた後、
オッズ帯別 × エッジ閾値別に ROI をスイープして
本命帯で見落とされていたスイートスポットを探す。
"""

import numpy as np
import pandas as pd

from src.features.build_features import build_all_features
from src.model.train import load_model, prepare_dataset, get_available_features
from src.betting.market_implied import add_market_top3_column


def _realized_fukusho_payout(df: pd.DataFrame) -> pd.Series:
    in_top3 = df["finish_position"] <= 3
    if "fukusho_odds_actual" in df.columns:
        actual = df["fukusho_odds_actual"]
        fallback = (df["odds"] * 0.3).clip(lower=1.1)
        payout = np.where(
            in_top3,
            np.where((actual.notna()) & (actual > 0), actual, fallback),
            0.0,
        )
    else:
        fallback = (df["odds"] * 0.3).clip(lower=1.1)
        payout = np.where(in_top3, fallback, 0.0)
    return pd.Series(payout, index=df.index)


def main():
    print("[1/3] 特徴量構築 & モデル読込...")
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    model = load_model("lightgbm_v6")

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    test_df["pred_prob"] = model.predict_proba(test_df[features])[:, 1]
    test_df = test_df[test_df["odds"].notna() & (test_df["odds"] > 0)].copy()

    print(f"[2/3] 市場暗黙確率を計算... ({test_df['race_id'].nunique()} races)")
    test_df = add_market_top3_column(test_df, out_col="market_prob_new")
    # オッズ帯極端時のフォールバック
    test_df["market_prob_new"] = test_df["market_prob_new"].fillna(
        (3.0 / test_df["odds"]).clip(upper=1.0)
    )
    test_df["market_prob_old"] = (3.0 / test_df["odds"]).clip(upper=1.0)
    test_df["edge_new"] = test_df["pred_prob"] - test_df["market_prob_new"]
    test_df["edge_old"] = test_df["pred_prob"] - test_df["market_prob_old"]
    test_df["payout"] = _realized_fukusho_payout(test_df)

    print("[3/3] スイープ...")

    # オッズ帯定義
    bands = [
        ("全体",      1.0, 10000),
        ("~3倍",      1.0,  3.0),
        ("3~5倍",     3.0,  5.0),
        ("5~10倍",    5.0, 10.0),
        ("10~20倍",  10.0, 20.0),
        ("20~50倍",  20.0, 50.0),
        ("50倍~",    50.0, 10000),
    ]

    def _roi_table(df_bets, total_races_df):
        """edge_new / edge_old の閾値スイープで ROI を返す"""
        rows = []
        for th in [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
            new_sub = df_bets[df_bets["edge_new"] >= th]
            old_sub = df_bets[df_bets["edge_old"] >= th]
            rows.append({
                "threshold": th,
                "new_bets": len(new_sub),
                "new_hit": new_sub["finish_position"].le(3).sum(),
                "new_hit_rate": new_sub["finish_position"].le(3).mean() * 100 if len(new_sub) else 0,
                "new_roi": new_sub["payout"].sum() / len(new_sub) * 100 if len(new_sub) else 0,
                "old_bets": len(old_sub),
                "old_hit_rate": old_sub["finish_position"].le(3).mean() * 100 if len(old_sub) else 0,
                "old_roi": old_sub["payout"].sum() / len(old_sub) * 100 if len(old_sub) else 0,
            })
        return pd.DataFrame(rows)

    for label, lo, hi in bands:
        sub = test_df[(test_df["odds"] >= lo) & (test_df["odds"] < hi)]
        if len(sub) < 100:
            continue
        print("\n" + "=" * 80)
        print(f"  オッズ帯: {label}  (対象 {len(sub)} 頭)")
        print("=" * 80)
        tbl = _roi_table(sub, None)
        tbl_disp = tbl.copy()
        for c in ["new_hit_rate", "new_roi", "old_hit_rate", "old_roi"]:
            tbl_disp[c] = tbl_disp[c].apply(lambda x: f"{x:6.1f}")
        print(tbl_disp.to_string(index=False))

        # 最良閾値（新計算ベース）
        best = tbl.loc[tbl["new_roi"].idxmax()]
        if best["new_bets"] >= 30:
            print(f"\n  → 新計算の最良: edge>={best['threshold']} "
                  f"bets={int(best['new_bets'])} hit={best['new_hit_rate']:.1f}% "
                  f"ROI={best['new_roi']:.1f}%")

    # 全体の比較サマリー
    print("\n\n" + "=" * 80)
    print("  旧式 vs 新式 の最良ROI 比較")
    print("=" * 80)
    summary_rows = []
    for label, lo, hi in bands:
        sub = test_df[(test_df["odds"] >= lo) & (test_df["odds"] < hi)]
        if len(sub) < 100:
            continue
        tbl = _roi_table(sub, None)
        tbl = tbl[(tbl["new_bets"] >= 30) & (tbl["old_bets"] >= 30)]
        if len(tbl) == 0:
            continue
        best_new = tbl.loc[tbl["new_roi"].idxmax()]
        best_old = tbl.loc[tbl["old_roi"].idxmax()]
        summary_rows.append({
            "band": label,
            "n": len(sub),
            "old_best_th": best_old["threshold"],
            "old_best_bets": int(best_old["old_bets"]),
            "old_best_roi": best_old["old_roi"],
            "new_best_th": best_new["threshold"],
            "new_best_bets": int(best_new["new_bets"]),
            "new_best_roi": best_new["new_roi"],
            "improvement": best_new["new_roi"] - best_old["old_roi"],
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:7.2f}",
    ))

    summary.to_csv("retune_fukusho_summary.csv", index=False, encoding="utf-8-sig")
    print("\n→ サマリーを retune_fukusho_summary.csv に保存")


if __name__ == "__main__":
    main()
