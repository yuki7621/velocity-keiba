"""v6モデル: v5 + 血統特徴量（種牡馬・母父）"""

from src.features.build_features import build_all_features
from src.model.train import train_model, save_model
from src.evaluation.backtest import run_ev_backtest, run_value_bet_backtest, run_backtest


def main():
    print("=" * 60)
    print("  v6モデル学習")
    print("=" * 60)
    print("  改善点:")
    print("    1. 血統特徴量を追加")
    print("       - sire_top3_rate / sire_surface_top3 / sire_distcat_top3")
    print("       - dam_sire_top3_rate / dam_sire_surface_top3")
    print("    2. v5 と同じ Classifier + Ranker アンサンブル構成を維持")
    print()

    # 1. 特徴量構築
    print("[Step 1] 特徴量構築...")
    df = build_all_features()

    # 2. モデル学習
    print("\n[Step 2] モデル学習 (Classifier + Ranker アンサンブル)")
    model = train_model(df, calibrate=True, ranker_weight=0.3)
    save_model(model, "lightgbm_v6")

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
    print("  学習完了 (v6)")
    print("=" * 60)
    print("  → v5 と比較してバックテスト結果を確認してください")
    print("  → 通常レース精度に改善が無ければ撤退（v5に戻す）")


if __name__ == "__main__":
    main()
