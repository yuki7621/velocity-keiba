"""期待値ベースのバックテスト実行スクリプト"""

from src.features.build_features import build_all_features
from src.model.train import load_model
from src.evaluation.backtest import (
    run_backtest,
    run_ev_backtest,
    run_value_bet_backtest,
)


def main():
    print("=== 特徴量構築 ===")
    df = build_all_features()

    print("\n=== モデル読み込み ===")
    model = load_model("lightgbm_v1")
    print("OK\n")

    # ─────────────────────────────────────
    # 旧方式（比較用）
    # ─────────────────────────────────────
    print("=" * 55)
    print("  【旧】確率閾値方式（ベースライン）")
    print("=" * 55)
    run_backtest(model, df, threshold=0.4)
    print()

    # ─────────────────────────────────────
    # 新方式1: 期待値(EV)ベース
    # ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  【新1】期待値(EV)方式 — パラメータ探索")
    print("=" * 55)
    for ev_th in [0.6, 0.8, 1.0, 1.2, 1.5]:
        run_ev_backtest(model, df, ev_threshold=ev_th, min_odds=3.0, max_odds=150.0)
        print()

    # ─────────────────────────────────────
    # 新方式2: バリューベット（エッジ）
    # ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  【新2】バリューベット方式 — パラメータ探索")
    print("=" * 55)
    for edge in [0.05, 0.10, 0.15, 0.20, 0.25]:
        run_value_bet_backtest(model, df, edge_threshold=edge, min_odds=3.0, max_odds=100.0)
        print()

    # ─────────────────────────────────────
    # オッズ帯別の分析
    # ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("  【分析】オッズ帯別のEV方式パフォーマンス")
    print("=" * 55)
    odds_ranges = [
        (3.0, 10.0, "中穴 (3〜10倍)"),
        (10.0, 30.0, "穴 (10〜30倍)"),
        (30.0, 100.0, "大穴 (30〜100倍)"),
    ]
    for min_o, max_o, label in odds_ranges:
        print(f"\n--- {label} ---")
        run_ev_backtest(model, df, ev_threshold=0.8, min_odds=min_o, max_odds=max_o)


if __name__ == "__main__":
    main()
