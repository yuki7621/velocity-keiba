"""v4モデル: キャリブレーション + 交互作用特徴量 + ハイパーパラメータ調整"""

from src.features.build_features import build_all_features
from src.model.train import train_model, save_model
from src.evaluation.backtest import run_ev_backtest, run_value_bet_backtest, run_backtest


def main():
    # 1. 特徴量構築
    print("=" * 55)
    print("  特徴量構築 (v4: 交互作用特徴量追加)")
    print("=" * 55)
    df = build_all_features()

    # 2. モデル学習（キャリブレーション付き）
    print("\n" + "=" * 55)
    print("  モデル学習 (v4: キャリブレーション対応)")
    print("=" * 55)
    print("  改善点:")
    print("    1. Isotonic Regression による確率キャリブレーション")
    print("    2. 高相関特徴量の交互作用 (馬×騎手シナジー等)")
    print("    3. ハイパーパラメータ調整 (過学習抑制)")
    print()

    model = train_model(df, calibrate=True)
    save_model(model, "lightgbm_v4")

    # 3. バックテスト（v3との比較用に同じ条件で実行）
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

    print("\n" + "=" * 55)
    print("  v4 改善まとめ")
    print("=" * 55)
    print("  ✓ 確率キャリブレーション: 予測確率の過大評価を補正")
    print("  ✓ 交互作用特徴量: 馬×騎手のシナジー、距離適性×騎手など")
    print("  ✓ ハイパーパラメータ: 正則化強化・葉ノード削減で過学習抑制")
    print()
    print("  → 上のバックテスト結果をv3と比較してください")
    print("  → 回収率が改善していれば、予測・買い目推奨でv4を選択可能")


if __name__ == "__main__":
    main()
