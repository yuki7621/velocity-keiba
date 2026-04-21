"""v7モデル: v6 + 脚質 & 休養パターン特徴量"""

from src.features.build_features import build_all_features
from src.model.train import train_model, save_model
from src.evaluation.backtest import run_ev_backtest, run_value_bet_backtest, run_backtest


def main():
    print("=" * 60)
    print("  v7モデル学習")
    print("=" * 60)
    print("  改善点:")
    print("    1. 脚質分類 (逃げ/先行/差し/追込)")
    print("       - horse_style_nige/senko/sashi/oikomi_rate")
    print("       - horse_main_style / main_style_course_fit")
    print("    2. 休養パターン詳細化")
    print("       - is_consecutive / is_short_break / is_long_break / rest_days_log")
    print("    3. v6 の血統特徴量を継続使用")
    print()

    # 1. 特徴量構築
    print("[Step 1] 特徴量構築...")
    df = build_all_features()

    # 2. モデル学習
    print("\n[Step 2] モデル学習 (Classifier + Ranker アンサンブル)")
    model = train_model(df, calibrate=True, ranker_weight=0.3)
    save_model(model, "lightgbm_v7")

    # 3. バックテスト (実払戻ベース)
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
    for edge in [0.10, 0.15, 0.20, 0.25]:
        run_value_bet_backtest(model, df, edge_threshold=edge, min_odds=3.0, max_odds=100.0)
        print()

    print("\n" + "=" * 60)
    print("  学習完了 (v7)")
    print("=" * 60)
    print("  → v6 と比較してバックテスト結果を確認してください")
    print("  → v6 から改善がなければ脚質特徴量の重み調整 or 撤退")


if __name__ == "__main__":
    main()
