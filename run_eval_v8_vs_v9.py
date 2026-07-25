"""v8 と v9 のバックテスト指標を横並び比較する

v9 (クラス・馬齢・天候を追加) が v8 を実際に上回ったかを判定するため、
同じテストデータで AUC / Log Loss / 複勝ROI / 単勝ROI を計算して並べる。
"""

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.betting.market_implied import add_market_top3_column
from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


def evaluate(model, test: pd.DataFrame, features: list[str]) -> dict:
    """共通の評価ロジック（run_eval_v6_vs_v8.py と同一条件）"""
    test = test.copy()
    test["pred_prob"] = model.predict_proba(test[features])[:, 1]
    y = (test["finish_position"] <= 3).astype(int)

    res = {
        "AUC": roc_auc_score(y, test["pred_prob"]),
        "LogLoss": log_loss(y, test["pred_prob"]),
        "n": len(test),
    }

    if "fukusho_odds_actual" in test.columns and test["fukusho_odds_actual"].notna().any():
        test["fukusho_odds"] = test["fukusho_odds_actual"]
    else:
        test["fukusho_odds"] = (test["odds"] * 0.3).clip(lower=1.1)

    test = add_market_top3_column(test, out_col="market_prob")
    test["market_prob"] = test["market_prob"].fillna((3.0 / test["odds"]).clip(upper=1.0))
    test["edge"] = test["pred_prob"] - test["market_prob"]

    bets = test[(test["edge"] >= 0.10) & (test["odds"] >= 10) & (test["odds"] <= 50)]
    if len(bets) > 0:
        is_top3 = (bets["finish_position"] <= 3).astype(int)
        is_win = (bets["finish_position"] == 1).astype(int)
        res["bets"] = len(bets)
        res["fukusho_hit_rate"] = is_top3.mean() * 100
        res["fukusho_roi"] = (is_top3 * bets["fukusho_odds"]).sum() / len(bets) * 100
        res["tansho_hit_rate"] = is_win.mean() * 100
        res["tansho_roi"] = (is_win * bets["odds"]).sum() / len(bets) * 100
    else:
        res.update({"bets": 0, "fukusho_hit_rate": 0.0, "fukusho_roi": 0.0,
                    "tansho_hit_rate": 0.0, "tansho_roi": 0.0})
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

    print("\n[2/3] v8 を評価...")
    try:
        r_old = evaluate(load_model("lightgbm_v8"), test, features)
    except FileNotFoundError:
        print("  ⚠️ lightgbm_v8.pkl が見つかりません")
        return

    print("[3/3] v9 を評価...")
    try:
        r_new = evaluate(load_model("lightgbm_v9"), test, features)
    except FileNotFoundError:
        print("  ⚠️ lightgbm_v9.pkl が見つかりません — 先に run_train_v9.py を実行してください")
        return

    print("\n" + "=" * 78)
    print("  v8 vs v9  比較結果")
    print("=" * 78)
    print(f"  {'指標':<25} {'v8':>14} {'v9':>14} {'差分':>14}")
    print(f"  {'-' * 25} {'-' * 14} {'-' * 14} {'-' * 14}")

    def _row(label, key, fmt="{:.4f}", higher_is_better=True):
        a, b = r_old[key], r_new[key]
        diff = b - a
        good = (diff >= 0) if higher_is_better else (diff <= 0)
        marker = "🟢" if good else "🔴"
        sign = "+" if diff >= 0 else ""
        print(f"  {label:<25} {fmt.format(a):>14} {fmt.format(b):>14} "
              f"{marker} {sign}{fmt.format(diff):>10}")

    _row("AUC", "AUC", "{:.4f}", True)
    _row("Log Loss", "LogLoss", "{:.4f}", False)
    print()
    print("  バリューベット条件: edge>=0.10 ∧ オッズ10-50倍")
    print(f"  賭数 v8={r_old['bets']} / v9={r_new['bets']}")
    _row("複勝 的中率(%)", "fukusho_hit_rate", "{:.2f}", True)
    _row("複勝 ROI(%)", "fukusho_roi", "{:.2f}", True)
    _row("単勝 的中率(%)", "tansho_hit_rate", "{:.2f}", True)
    _row("単勝 ROI(%)", "tansho_roi", "{:.2f}", True)

    print("\n" + "=" * 78)
    if r_new["AUC"] > r_old["AUC"] and r_new["fukusho_roi"] >= r_old["fukusho_roi"]:
        print("  ✅ v9 採用推奨: AUC・ROI ともに v8 を上回りました")
    elif r_new["AUC"] > r_old["AUC"]:
        print("  ⚠️ v9 は AUC は改善したが ROI が下がっています — 慎重に判断")
    else:
        print("  ❌ v9 は v8 に劣ります — v8 を維持推奨（v7 と同じく撤退）")
    print("=" * 78)


if __name__ == "__main__":
    main()
