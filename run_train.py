"""モデル学習の実行スクリプト"""

from src.features.build_features import build_all_features
from src.model.train import train_model, save_model
from src.evaluation.backtest import run_backtest


def main():
    # 1. 特徴量構築
    print("=== 特徴量構築 ===")
    df = build_all_features()

    # 2. モデル学習
    print("\n=== モデル学習 ===")
    model = train_model(df)

    # 3. モデル保存
    save_model(model)

    # 4. バックテスト
    print("\n=== バックテスト ===")
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        run_backtest(model, df, threshold=threshold)
        print()


if __name__ == "__main__":
    main()
