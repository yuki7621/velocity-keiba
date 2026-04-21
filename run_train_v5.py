"""v5モデル: Classifier + Ranker アンサンブル + 調教師特徴量"""

from src.features.build_features import build_all_features
from src.model.train import train_model, save_model
from src.evaluation.backtest import run_ev_backtest, run_value_bet_backtest, run_backtest


def main():
    print("=" * 60)
    print("  v5モデル学習")
    print("=" * 60)
    print("  改善点:")
    print("    1. 調教師特徴量 (trainer_win_rate 等 4特徴量)")
    print("    2. LightGBM Ranker をアンサンブル（的中率向上）")
    print("    3. 予測確率フィルターをUIに追加")
    print()

    # 1. 特徴量構築
    print("[Step 1] 特徴量構築...")
    df = build_all_features()

    # 2. モデル学習
    print("\n[Step 2] モデル学習 (Classifier + Ranker アンサンブル)")
    model = train_model(df, calibrate=True, ranker_weight=0.3)
    save_model(model, "lightgbm_v5")

    # 3. バックテスト
    print("\n" + "=" * 60)
    print("  バックテスト: 確率閾値方式")
    print("=" * 60)
    run_backtest(model, df, threshold=0.4)

    print("\n" + "=" * 60)
    print("  バックテスト: 期待値方式")
    print("=" * 60)
    for ev_th in [0.8, 1.0, 1.2, 1.5]:
        run_ev_backtest(model, df, ev_threshold=ev_th, min_odds=3.0, max_odds=150.0)
        print()

    print("\n" + "=" * 60)
    print("  バックテスト: バリューベット方式")
    print("=" * 60)
    for edge in [0.05, 0.10, 0.15, 0.20]:
        run_value_bet_backtest(model, df, edge_threshold=edge, min_odds=3.0, max_odds=100.0)
        print()

    print("\n" + "=" * 60)
    print("  学習完了")
    print("=" * 60)
    print("  → モデル診断で v5 を選択してください")
    print("  → 買い目推奨の「的中率優先フィルター」を試してください")


if __name__ == "__main__":
    main()
