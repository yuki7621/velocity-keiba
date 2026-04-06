"""v3モデル: 馬場傾向対応版の学習・評価スクリプト"""

from src.features.build_features import build_all_features
from src.model.train import train_model, save_model
from src.evaluation.backtest import run_ev_backtest, run_value_bet_backtest, run_backtest


def main():
    # 1. 特徴量構築（馬場傾向含む）
    print("=" * 55)
    print("  特徴量構築 (v3: 馬場傾向対応)")
    print("=" * 55)
    df = build_all_features()

    # 2. モデル学習
    print("\n" + "=" * 55)
    print("  モデル学習 (v3)")
    print("=" * 55)
    model = train_model(df)
    save_model(model, "lightgbm_v3")

    # 3. バックテスト
    print("\n" + "=" * 55)
    print("  バックテスト: 確率閾値方式 (ベースライン)")
    print("=" * 55)
    run_backtest(model, df, threshold=0.4)

    print("\n" + "=" * 55)
    print("  バックテスト: 期待値方式")
    print("=" * 55)
    for ev_th in [0.8, 1.0, 1.2, 1.5]:
        run_ev_backtest(model, df, ev_threshold=ev_th, min_odds=3.0, max_odds=150.0)
        print()

    print("\n" + "=" * 55)
    print("  バックテスト: バリューベット方式")
    print("=" * 55)
    for edge in [0.05, 0.10, 0.15, 0.20]:
        run_value_bet_backtest(model, df, edge_threshold=edge, min_odds=3.0, max_odds=100.0)
        print()


if __name__ == "__main__":
    main()
