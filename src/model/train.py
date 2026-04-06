"""モデルの学習と予測 v2"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT


# 学習に使う特徴量カラム (v2: 大幅追加)
FEATURE_COLUMNS = [
    # --- 基本情報 ---
    "post_number",
    "gate_number",
    "weight_carried",
    "horse_weight",
    "weight_change",
    "distance",
    "condition_num",
    "is_inner_gate",
    "weight_ratio",
    "post_position_ratio",

    # --- 馬の過去成績 ---
    "horse_avg_finish_5",
    "horse_avg_last_3f_5",
    "horse_avg_time_5",
    "horse_avg_speed_idx_5",
    "horse_win_rate_5",
    "horse_top3_rate_5",
    "horse_race_count",
    "horse_last_finish",
    "horse_last_speed_idx",
    "horse_best_speed_idx_5",
    "horse_form_trend",
    "horse_last3f_std_5",
    "horse_total_prize",
    "horse_avg_prize",

    # --- 適性 ---
    "horse_dist_top3_rate",
    "horse_dist_race_count",
    "horse_surface_top3_rate",
    "horse_venue_top3_rate",
    "distance_diff",

    # --- ペース・脚質 ---
    "horse_avg_early_pos",
    "horse_avg_pos_change",

    # --- 騎手 ---
    "jockey_win_rate",
    "jockey_top3_rate",
    "jockey_recent_top3_20",
    "jockey_venue_top3",

    # --- 間隔 ---
    "days_since_last",
    "is_fresh",

    # --- レース内の相対値 ---
    "horse_avg_speed_idx_5_race_zscore",
    "horse_avg_speed_idx_5_race_rank",
    "horse_top3_rate_5_race_zscore",
    "horse_top3_rate_5_race_rank",
    "jockey_win_rate_race_zscore",
    "jockey_win_rate_race_rank",

    # --- 馬場傾向（前日バイアス）---
    "bias_gate",
    "bias_pace",
    "bias_time",
    "bias_last3f",
    "bias_inner_top3",
    "bias_outer_top3",
    "bias_front_top3",
    "bias_closer_top3",

    # --- バイアス × 馬の特性（交互作用）---
    "bias_x_inner",
    "bias_x_pace",
    "bias_x_last3f",
]

# 目的変数: 3着以内かどうか
TARGET_COLUMN = "is_top3"

MODEL_DIR = PROJECT_ROOT / "models"


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """学習用にデータを整形する"""
    df = df.copy()
    df[TARGET_COLUMN] = (df["finish_position"] <= 3).astype(int)

    # 存在しない特徴量列があれば除外
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"  ※ 未生成の特徴量を除外: {missing}")

    # 欠損が多い行を除外 (過去データが不足する初期レコード)
    df = df.dropna(subset=["horse_avg_finish_5", "jockey_win_rate"])
    return df


def get_available_features(df: pd.DataFrame) -> list[str]:
    """DataFrameに存在する特徴量のみ返す"""
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def train_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """LightGBMモデルを学習する"""
    df = prepare_dataset(df)
    features = get_available_features(df)

    # 時系列で分割 (最新20%をテストに)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[features]
    y_test = test_df[TARGET_COLUMN]

    print(f"学習データ: {len(train_df)}件, テストデータ: {len(test_df)}件")
    print(f"特徴量数: {len(features)}")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    accuracy = model.score(X_test, y_test)
    print(f"テストデータ正解率: {accuracy:.4f}")

    # 特徴量重要度を表示
    print("\n--- 特徴量重要度 TOP 15 ---")
    importance = pd.Series(model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)
    for feat, imp in importance.head(15).items():
        print(f"  {feat:40s} {imp:6d}")

    return model


def save_model(model: lgb.LGBMClassifier, name: str = "lightgbm_v2"):
    """モデルを保存する"""
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"モデルを保存しました: {path}")


def load_model(name: str = "lightgbm_v2") -> lgb.LGBMClassifier:
    """モデルを読み込む"""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
