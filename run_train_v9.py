"""v9モデル: v8 + クラス・馬齢・天候（競馬の基本要素）

背景:
  90特徴量を監査したところ、競馬予想の最も基本的な要素が欠落していた:
    - レースクラス（新馬/未勝利/1勝…）と 前走からの昇級・降級
    - 馬齢
    - 性別
    - 天候・出走頭数
  さらに prize が全件NULL・trainer_id が5.6%しか無く、
  賞金系2特徴量・調教師系4特徴量が実質死んでいた。

追加特徴量 (9個):
  - race_class        : レースクラス (0新馬〜5OP)。固有名レースは NaN
  - is_named_race     : 特別/OP/重賞など固有名レースか
  - prev_race_class   : 前走のクラス
  - class_diff        : 今走 - 前走 のクラス差 (正=昇級 / 負=降級)
  - horse_max_class   : 過去に経験した最高クラス
  - horse_age         : 馬齢 (horse_id 先頭4桁の生年から算出。実測500/500一致)
  - sex_num           : 性別 (再スクレイプ後に有効。未取得期間は NaN)
  - weather_num       : 天候
  - head_count        : 出走頭数

採否方針:
  v8 のバックテスト ROI/AUC を超えなければ採用しない（v7 と同じ撤退基準）。
"""

from src.evaluation.backtest import run_backtest, run_ev_backtest, run_value_bet_backtest
from src.features.build_features import build_all_features
from src.model.train import save_model, train_model


def main():
    print("=" * 60)
    print("  v9モデル学習")
    print("=" * 60)
    print("  改善点:")
    print("    1. レースクラス + 前走クラス差（昇級/降級）を追加")
    print("    2. 馬齢を追加（horse_id の生年から算出）")
    print("    3. 性別・天候・出走頭数を追加")
    print("    4. v8 の全特徴量（展開シナジー等）を維持")
    print()

    # 1. 特徴量構築
    print("[Step 1] 特徴量構築...")
    df = build_all_features()

    # 2. モデル学習
    print("\n[Step 2] モデル学習 (Classifier + Ranker アンサンブル)")
    model = train_model(df, calibrate=True, ranker_weight=0.3)
    save_model(model, "lightgbm_v9")

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
    print("  学習完了 (v9)")
    print("=" * 60)
    print("  → run_eval_v8_vs_v9.py で v8 と比較してください")


if __name__ == "__main__":
    main()
