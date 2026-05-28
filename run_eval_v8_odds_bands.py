"""v8 の単勝・複勝 ROI をオッズ帯細分で分析する

目的: v8 で単勝 ROI が控除率20%の壁 (100%) を超える帯がないかを探す。
条件: edge >= 0.10 (v6 vs v8 比較と同じ)
"""

import pandas as pd

from src.betting.market_implied import add_market_top3_column
from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


# 細分化したオッズ帯
ODDS_BANDS = [
    (1.0, 3.0),
    (3.0, 5.0),
    (5.0, 7.0),
    (7.0, 10.0),
    (10.0, 15.0),
    (15.0, 20.0),
    (20.0, 25.0),
    (25.0, 30.0),
    (30.0, 40.0),
    (40.0, 50.0),
    (50.0, 70.0),
    (70.0, 100.0),
    (100.0, 200.0),
    (200.0, 9999.0),
]


def analyze(test: pd.DataFrame, edge_threshold: float, label: str) -> pd.DataFrame:
    """エッジ閾値ごとにオッズ帯別 ROI を返す"""
    bets_pool = test[test["edge"] >= edge_threshold].copy()

    rows = []
    for lo, hi in ODDS_BANDS:
        sub = bets_pool[(bets_pool["odds"] >= lo) & (bets_pool["odds"] < hi)]
        if len(sub) < 30:  # サンプル少ない帯はスキップ
            continue
        n = len(sub)
        is_top3 = (sub["finish_position"] <= 3).astype(int)
        is_win = (sub["finish_position"] == 1).astype(int)
        fuku_payout = (is_top3 * sub["fukusho_odds"]).sum()
        win_payout = (is_win * sub["odds"]).sum()
        rows.append({
            "オッズ帯": f"{lo:.0f}-{hi:.0f}" if hi < 9999 else f"{lo:.0f}+",
            "賭数": n,
            "複勝率": is_top3.mean() * 100,
            "複勝ROI": fuku_payout / n * 100,
            "勝率": is_win.mean() * 100,
            "単勝ROI": win_payout / n * 100,
            "平均odds": sub["odds"].mean(),
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    if len(df) == 0:
        print("  (該当データなし)")
        return
    print(f"  {'オッズ帯':<10} {'賭数':>6} {'複勝率':>7} {'複勝ROI':>9} "
          f"{'勝率':>6} {'単勝ROI':>9} {'平均odds':>9}")
    print(f"  {'-' * 10} {'-' * 6} {'-' * 7} {'-' * 9} {'-' * 6} {'-' * 9} {'-' * 9}")
    for _, r in df.iterrows():
        win_marker = " 🟢" if r["単勝ROI"] >= 100 else ("  " if r["単勝ROI"] >= 90 else " 🔴")
        fuk_marker = " 🟢" if r["複勝ROI"] >= 100 else ("  " if r["複勝ROI"] >= 80 else " 🔴")
        print(f"  {r['オッズ帯']:<10} {int(r['賭数']):>6} "
              f"{r['複勝率']:>6.1f}% {r['複勝ROI']:>7.1f}%{fuk_marker} "
              f"{r['勝率']:>5.1f}% {r['単勝ROI']:>7.1f}%{win_marker} "
              f"{r['平均odds']:>8.1f}")


def main():
    print("[1/3] 特徴量構築 & v8 読込...")
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    model = load_model("lightgbm_v8")

    print("[2/3] テストデータの予測...")
    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()
    test = test[test["odds"].notna() & (test["odds"] > 0)].copy()
    test["pred_prob"] = model.predict_proba(test[features])[:, 1]

    # 複勝オッズ
    if "fukusho_odds_actual" in test.columns and test["fukusho_odds_actual"].notna().any():
        test["fukusho_odds"] = test["fukusho_odds_actual"]
        fuk_label = "実複勝"
    else:
        test["fukusho_odds"] = (test["odds"] * 0.3).clip(lower=1.1)
        fuk_label = "概算"

    # エッジ
    test = add_market_top3_column(test, out_col="market_prob")
    test["market_prob"] = test["market_prob"].fillna(
        (3.0 / test["odds"]).clip(upper=1.0)
    )
    test["edge"] = test["pred_prob"] - test["market_prob"]

    print(f"  テスト期間: {test['date'].min().strftime('%Y-%m-%d')} 〜 "
          f"{test['date'].max().strftime('%Y-%m-%d')} ({len(test):,}行 / 複勝オッズ: {fuk_label})")

    print("[3/3] 集計中...")

    # ── 各エッジ閾値ごとに ──
    for edge_th, label in [
        (0.0, "全件 (エッジ条件なし)"),
        (0.05, "edge >= 0.05"),
        (0.10, "edge >= 0.10  ★前回の主条件"),
        (0.15, "edge >= 0.15"),
    ]:
        result = analyze(test, edge_th, label)
        print_table(result, f"v8 オッズ帯別 ROI [{label}]")

    # ── 単勝 ROI 100% を超えた帯のサマリー ──
    print("\n" + "=" * 90)
    print("  ✨ 単勝 ROI 100% 超え (or それに近い) を達成したセグメント")
    print("=" * 90)
    found = []
    for edge_th, label in [(0.05, "edge>=0.05"), (0.10, "edge>=0.10"), (0.15, "edge>=0.15")]:
        result = analyze(test, edge_th, label)
        for _, r in result.iterrows():
            if r["単勝ROI"] >= 95 or r["複勝ROI"] >= 100:
                found.append({
                    "条件": label,
                    "オッズ帯": r["オッズ帯"],
                    "賭数": int(r["賭数"]),
                    "勝率(%)": round(r["勝率"], 2),
                    "単勝ROI(%)": round(r["単勝ROI"], 1),
                    "複勝率(%)": round(r["複勝率"], 2),
                    "複勝ROI(%)": round(r["複勝ROI"], 1),
                })

    if found:
        df_found = pd.DataFrame(found).sort_values("単勝ROI(%)", ascending=False)
        print(df_found.to_string(index=False))
    else:
        print("  該当なし — 全帯で控除率20%の壁を突破できませんでした")

    # CSV 保存
    csv_rows = []
    for edge_th, label in [(0.0, "全件"), (0.05, "edge>=0.05"), (0.10, "edge>=0.10"), (0.15, "edge>=0.15")]:
        r = analyze(test, edge_th, label)
        r["条件"] = label
        csv_rows.append(r)
    pd.concat(csv_rows, ignore_index=True).to_csv(
        "v8_odds_bands.csv", index=False, encoding="utf-8-sig",
    )
    print("\n→ v8_odds_bands.csv に保存")


if __name__ == "__main__":
    main()
