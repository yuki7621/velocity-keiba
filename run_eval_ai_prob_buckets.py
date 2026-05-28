"""AI確率帯別の的中率・回収率を分析する

「pred_prob が何%以上の馬まで馬券に組み込めばROIがプラスになるか」を
複勝・単勝の両方で確認する。
"""

import pandas as pd

from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


def main():
    print("[1/3] 特徴量構築 & v6 読込...")
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    model = load_model("lightgbm_v6")

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()
    test["pred_prob"] = model.predict_proba(test[features])[:, 1]

    # 有効データのみ
    test = test[test["odds"].notna() & (test["odds"] > 0)].copy()
    test["is_top3"] = (test["finish_position"] <= 3).astype(int)
    test["is_win"] = (test["finish_position"] == 1).astype(int)

    # 複勝オッズ: 実データがあれば使用、なければ概算 (odds * 0.3, 下限 1.1)
    if "fukusho_odds_actual" in test.columns and test["fukusho_odds_actual"].notna().any():
        test["fukusho_odds"] = test["fukusho_odds_actual"]
        fuku_label = "実複勝オッズ"
    else:
        test["fukusho_odds"] = (test["odds"] * 0.3).clip(lower=1.1)
        fuku_label = "概算 (単勝×0.3)"

    print(f"\n対象: {len(test):,} 出走 ({test['date'].min()} 〜 {test['date'].max()})")
    print(f"複勝オッズ: {fuku_label}")

    # ─────────────────────────────────────────
    # AI確率帯別の的中率・ROI
    # ─────────────────────────────────────────
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%",
              "25-30%", "30-40%", "40-50%", "50%+"]
    test["bucket"] = pd.cut(test["pred_prob"], bins=bins, labels=labels, right=False)

    print("\n" + "=" * 80)
    print("【AI確率帯別】出走頭数・複勝率・複勝ROI・単勝ROI")
    print("=" * 80)
    print(f"  {'帯':<8} {'頭数':>6} {'複勝':>6} {'複勝率':>7} {'複勝ROI':>8} "
          f"{'単勝':>5} {'勝率':>6} {'単勝ROI':>8} {'平均odds':>9}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 8} "
          f"{'-' * 5} {'-' * 6} {'-' * 8} {'-' * 9}")

    for label in labels:
        sub = test[test["bucket"] == label]
        if len(sub) == 0:
            continue
        n = len(sub)
        top3 = sub["is_top3"].sum()
        win = sub["is_win"].sum()
        fuku_payout = (sub["is_top3"] * sub["fukusho_odds"]).sum()
        win_payout = (sub["is_win"] * sub["odds"]).sum()
        avg_odds = sub["odds"].mean()
        print(f"  {label:<8} {n:>6} {top3:>6} {top3/n*100:>6.1f}% "
              f"{fuku_payout/n*100:>7.1f}% {win:>5} {win/n*100:>5.1f}% "
              f"{win_payout/n*100:>7.1f}% {avg_odds:>9.1f}")

    # ─────────────────────────────────────────
    # 累積：閾値以上に賭けた場合
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【累積】pred_prob >= 閾値 に複勝・単勝を全部買った場合")
    print("=" * 80)
    print(f"  {'閾値':<7} {'対象数':>7} {'複勝率':>7} {'複勝ROI':>8} "
          f"{'勝率':>6} {'単勝ROI':>8}")
    print(f"  {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 8}")

    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        sub = test[test["pred_prob"] >= th]
        if len(sub) == 0:
            continue
        n = len(sub)
        top3 = sub["is_top3"].sum()
        win = sub["is_win"].sum()
        fuku_payout = (sub["is_top3"] * sub["fukusho_odds"]).sum()
        win_payout = (sub["is_win"] * sub["odds"]).sum()
        print(f"  >={th:<5.2f} {n:>7} {top3/n*100:>6.1f}% "
              f"{fuku_payout/n*100:>7.1f}% {win/n*100:>5.1f}% "
              f"{win_payout/n*100:>7.1f}%")

    # ─────────────────────────────────────────
    # オッズフィルタ併用 (バックテストUIのデフォルト相当)
    # ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("【オッズ3.0〜100.0 内 + AI確率閾値】複勝のみ")
    print("=" * 80)
    print(f"  {'閾値':<7} {'対象数':>7} {'複勝率':>7} {'複勝ROI':>8}")
    print(f"  {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")

    odds_filtered = test[(test["odds"] >= 3.0) & (test["odds"] <= 100.0)]
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        sub = odds_filtered[odds_filtered["pred_prob"] >= th]
        if len(sub) == 0:
            continue
        n = len(sub)
        top3 = sub["is_top3"].sum()
        fuku_payout = (sub["is_top3"] * sub["fukusho_odds"]).sum()
        print(f"  >={th:<5.2f} {n:>7} {top3/n*100:>6.1f}% "
              f"{fuku_payout/n*100:>7.1f}%")

    # ─────────────────────────────────────────
    # 「エッジ」と組み合わせた効果
    # ─────────────────────────────────────────
    from src.betting.market_implied import add_market_top3_column
    test_em = add_market_top3_column(test, out_col="market_prob")
    test_em["market_prob"] = test_em["market_prob"].fillna(
        (3.0 / test_em["odds"]).clip(upper=1.0)
    )
    test_em["edge"] = test_em["pred_prob"] - test_em["market_prob"]

    print("\n" + "=" * 80)
    print("【AI確率 × エッジ】複勝ROI のヒートマップ的内訳")
    print("=" * 80)
    print(f"  {'AI確率':<8} {'エッジ条件':<14} {'対象数':>7} {'複勝率':>7} {'複勝ROI':>8}")
    print(f"  {'-' * 8} {'-' * 14} {'-' * 7} {'-' * 7} {'-' * 8}")

    for prob_th in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for edge_th, edge_lbl in [(-1.0, "全て"), (0.05, "edge>=0.05"),
                                    (0.10, "edge>=0.10"), (0.15, "edge>=0.15")]:
            sub = test_em[(test_em["pred_prob"] >= prob_th)
                          & (test_em["edge"] >= edge_th)]
            if len(sub) < 30:
                continue
            n = len(sub)
            top3 = sub["is_top3"].sum()
            fuku_payout = (sub["is_top3"] * sub["fukusho_odds"]).sum()
            print(f"  >={prob_th:<5.2f}  {edge_lbl:<14} {n:>7} "
                  f"{top3/n*100:>6.1f}% {fuku_payout/n*100:>7.1f}%")
        print()


if __name__ == "__main__":
    main()
