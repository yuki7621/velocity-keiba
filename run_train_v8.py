"""v8モデル: v6 + 展開シナジー特徴量（レース全体の脚質構成 × 自分の脚質）

v7 が劣化した原因への対処:
  v7 は「自分の脚質」単体特徴量を追加したが、レース全体のメンバー構成を
  考慮していなかったため逆効果になった（例: 逃げ馬が他にも多いとハイペース
  で共倒れ → 逃げ馬の評価を上げると外す）。

v8 のアプローチ:
  v6 の全特徴量 + v7 の単体脚質 + 「レース構成 × 自分の脚質」の交互作用
  を追加。単体脚質を消さず、文脈情報と組み合わせて学習させる。

追加特徴量 (11個):
  - race_n_nigema / senko / sashi / oikomi   : レース内の主脚質別頭数
  - race_pace_pressure                        : 全馬の (逃げ率+先行率) 平均
  - race_avg_main_style                       : レース内主脚質コードの平均
  - is_lone_nigema                            : 自分が唯一の逃げ馬 (楽逃げ)
  - pace_pressure_x_self_{nige,senko,oikomi}  : ハイペース × 自分の脚質
  - n_nigema_x_self_nige                      : 逃げ馬多 × 自分も逃げ = 共倒れ

採否方針:
  v8 のバックテスト ROI が v6 を超えなければ採用しない（v7 と同じ撤退基準）。
"""

from src.evaluation.backtest import run_backtest, run_ev_backtest, run_value_bet_backtest
from src.features.build_features import build_all_features
from src.model.train import save_model, train_model


def main():
    print("=" * 60)
    print("  v8モデル学習")
    print("=" * 60)
    print("  改善点:")
    print("    1. 展開シナジー特徴量を追加（レース構成 × 自分の脚質）")
    print("       - race_n_nigema / race_pace_pressure / is_lone_nigema")
    print("       - pace_pressure_x_self_nige / oikomi / senko")
    print("       - n_nigema_x_self_nige")
    print("    2. v6 の全特徴量 + v7 の単体脚質 を維持")
    print("    3. v5 と同じ Classifier + Ranker アンサンブル構成")
    print()

    # 1. 特徴量構築
    print("[Step 1] 特徴量構築...")
    df = build_all_features()

    # 2. モデル学習
    print("\n[Step 2] モデル学習 (Classifier + Ranker アンサンブル)")
    model = train_model(df, calibrate=True, ranker_weight=0.3)
    save_model(model, "lightgbm_v8")

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
    print("  学習完了 (v8)")
    print("=" * 60)
    print("  → v6 と比較してバックテスト結果を確認してください")
    print("  → AUC・ROI ともに v6 を超えなければ採用しない（v6 に戻す）")


if __name__ == "__main__":
    main()
