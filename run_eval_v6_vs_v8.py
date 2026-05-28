"""v6 と v8 のバックテスト指標を横並び比較する

v8 (展開シナジー特徴量付き) が v6 を実際に上回ったかを判定するため、
同じテストデータで AUC / Log Loss / 複勝ROI / 単勝ROI を計算して並べる。
"""

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


def evaluate(model, test: pd.DataFrame, features: list[str]) -> dict:
    """共通の評価ロジック"""
    test = test.copy()
    test["pred_prob"] = model.predict_proba(test[features])[:, 1]
    y = (test["finish_position"] <= 3).astype(int)

    res = {
        "AUC": roc_auc_score(y, test["pred_prob"]),
        "LogLoss": log_loss(y, test["pred_prob"]),
        "n": len(test),
    }

    # 複勝オッズ（実データ優先、無ければ概算）
    if "fukusho_odds_actual" in test.columns and test["fukusho_odds_actual"].notna().any():
        test["fukusho_odds"] = test["fukusho_odds_actual"]
    else:
        test["fukusho_odds"] = (test["odds"] * 0.3).clip(lower=1.1)

    # ── オッズ帯 10-50倍 で edge >= 0.10 のバリューベット ──
    from src.betting.market_implied import add_market_top3_column
    test = add_market_top3_column(test, out_col="market_prob")
    test["market_prob"] = test["market_prob"].fillna(
        (3.0 / test["odds"]).clip(upper=1.0)
    )
    test["edge"] = test["pred_prob"] - test["market_prob"]

    # 複勝ROI (edge>=0.10 ∧ 10<=odds<=50)
    bet_mask = (test["edge"] >= 0.10) & (test["odds"] >= 10) & (test["odds"] <= 50)
    bets = test[bet_mask]
    if len(bets) > 0:
        is_top3 = (bets["finish_position"] <= 3).astype(int)
        payout = (is_top3 * bets["fukusho_odds"]).sum()
        res["fukusho_bets"] = len(bets)
        res["fukusho_hit_rate"] = is_top3.mean() * 100
        res["fukusho_roi"] = payout / len(bets) * 100
    else:
        res["fukusho_bets"] = 0
        res["fukusho_hit_rate"] = 0.0
        res["fukusho_roi"] = 0.0

    # 単勝ROI (同条件)
    if len(bets) > 0:
        is_win = (bets["finish_position"] == 1).astype(int)
        win_payout = (is_win * bets["odds"]).sum()
        res["tansho_hit_rate"] = is_win.mean() * 100
        res["tansho_roi"] = win_payout / len(bets) * 100
    else:
        res["tansho_hit_rate"] = 0.0
        res["tansho_roi"] = 0.0

    return res


def main():
    print("[1/3] 特徴量構築...")
    df = build_all_features()
    df = prepare_dataset(df)
    features = get_available_features(df)
    print(f"  使用可能な特徴量: {len(features)}個")

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:].copy()
    test = test[test["odds"].notna() & (test["odds"] > 0)].copy()
    print(f"  テスト期間: {test['date'].min()} 〜 {test['date'].max()} ({len(test):,}行)")

    print("\n[2/3] v6 を評価...")
    try:
        v6 = load_model("lightgbm_v6")
        r6 = evaluate(v6, test, features)
    except FileNotFoundError:
        print("  ⚠️ lightgbm_v6.pkl が見つかりません")
        return

    print("[3/3] v8 を評価...")
    try:
        v8 = load_model("lightgbm_v8")
        r8 = evaluate(v8, test, features)
    except FileNotFoundError:
        print("  ⚠️ lightgbm_v8.pkl が見つかりません — 先に run_train_v8.py を実行してください")
        return

    # ── 結果を並べて表示 ──
    print("\n" + "=" * 78)
    print("  v6 vs v8  比較結果")
    print("=" * 78)
    print(f"  {'指標':<25} {'v6':>14} {'v8':>14} {'差分':>14}")
    print(f"  {'-' * 25} {'-' * 14} {'-' * 14} {'-' * 14}")

    def _row(label, key, fmt="{:.4f}", higher_is_better=True):
        v6_val = r6[key]
        v8_val = r8[key]
        diff = v8_val - v6_val
        sign = "+" if diff >= 0 else ""
        good = (diff >= 0) if higher_is_better else (diff <= 0)
        marker = "🟢" if good else "🔴"
        print(f"  {label:<25} {fmt.format(v6_val):>14} {fmt.format(v8_val):>14} "
              f"{marker} {sign}{fmt.format(diff):>10}")

    _row("AUC", "AUC", "{:.4f}", higher_is_better=True)
    _row("Log Loss", "LogLoss", "{:.4f}", higher_is_better=False)
    print()
    print("  バリューベット条件: edge>=0.10 ∧ オッズ10-50倍")
    print(f"  賭数 v6={r6['fukusho_bets']} / v8={r8['fukusho_bets']}")
    _row("複勝 的中率(%)", "fukusho_hit_rate", "{:.2f}", higher_is_better=True)
    _row("複勝 ROI(%)", "fukusho_roi", "{:.2f}", higher_is_better=True)
    _row("単勝 的中率(%)", "tansho_hit_rate", "{:.2f}", higher_is_better=True)
    _row("単勝 ROI(%)", "tansho_roi", "{:.2f}", higher_is_better=True)

    print("\n" + "=" * 78)
    if r8["AUC"] > r6["AUC"] and r8["fukusho_roi"] >= r6["fukusho_roi"]:
        print("  ✅ v8 採用推奨: AUC・ROI ともに v6 を上回りました")
    elif r8["AUC"] > r6["AUC"]:
        print("  ⚠️ v8 は AUC は改善したが ROI が下がっています — 慎重に判断")
    else:
        print("  ❌ v8 は v6 に劣ります — v6 を維持推奨（v7 と同じく撤退）")
    print("=" * 78)


if __name__ == "__main__":
    main()
