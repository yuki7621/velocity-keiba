"""v5 と v6 を同じデータで比較する"""

import pickle
from pathlib import Path
from src.features.build_features import build_all_features
from src.evaluation.backtest import run_ev_backtest, run_value_bet_backtest, run_backtest


def load(name):
    with open(Path("models") / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    print("=" * 60)
    print("  v5 vs v6 比較バックテスト")
    print("=" * 60)

    print("[Step 1] 特徴量構築...")
    df = build_all_features()

    for ver in ["v5", "v6"]:
        print("\n" + "#" * 60)
        print(f"#  {ver}")
        print("#" * 60)
        model = load(f"lightgbm_{ver}")

        print(f"\n--- [{ver}] 確率閾値 0.4 ---")
        run_backtest(model, df, threshold=0.4)

        print(f"\n--- [{ver}] EV方式 ---")
        for ev_th in [1.0, 1.2, 1.5]:
            run_ev_backtest(model, df, ev_threshold=ev_th, min_odds=3.0, max_odds=150.0)
            print()

        print(f"\n--- [{ver}] バリューベット ---")
        for edge in [0.10, 0.15, 0.20]:
            run_value_bet_backtest(model, df, edge_threshold=edge, min_odds=3.0, max_odds=100.0)
            print()


if __name__ == "__main__":
    main()
