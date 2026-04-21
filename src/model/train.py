"""モデルの学習と予測 v4 — キャリブレーション対応"""

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from config.settings import PROJECT_ROOT


# 学習に使う特徴量カラム (v4: 交互作用追加)
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

    # --- v4: 高相関特徴量の交互作用 ---
    "horse_jockey_synergy",       # 馬の実力 × 騎手の実力
    "horse_dist_jockey",          # 距離適性 × 騎手の実力
    "horse_form_x_jockey_recent", # 馬の近走調子 × 騎手の直近調子
    "horse_speed_x_dist_apt",     # スピード指数 × 距離適性
    "jockey_top3_race_zscore",    # 騎手複勝率のレース内偏差値
    "jockey_top3_race_rank",      # 騎手複勝率のレース内順位

    # --- v5: 調教師特徴量 ---
    "trainer_win_rate",           # 調教師の通算勝率
    "trainer_top3_rate",          # 調教師の通算複勝率
    "trainer_recent_top3_20",     # 調教師の直近20走複勝率
    "trainer_venue_top3",         # 調教師のその競馬場での複勝率

    # --- v6: 血統特徴量 ---
    "sire_top3_rate",             # 種牡馬の通算複勝率
    "sire_surface_top3",          # 種牡馬の芝/ダ別複勝率
    "sire_distcat_top3",          # 種牡馬の短中距離/中長距離別複勝率
    "dam_sire_top3_rate",         # 母父の通算複勝率
    "dam_sire_surface_top3",      # 母父の芝/ダ別複勝率

    # --- v7: 脚質 & 休養パターン ---
    "horse_style_nige_rate",      # 過去の逃げ率
    "horse_style_senko_rate",     # 過去の先行率
    "horse_style_sashi_rate",     # 過去の差し率
    "horse_style_oikomi_rate",    # 過去の追込率
    "horse_main_style",           # 主脚質 (1逃/2先/3差/4追) — カテゴリだがLGBMは数値OK
    "main_style_course_fit",      # 自馬の主脚質 × このコースの過去複勝率
    "is_consecutive",             # 連闘 (7日以内)
    "is_short_break",             # 中1〜2週
    "is_long_break",              # 半年以上の休み明け
    "rest_days_log",              # 休養日数の対数
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


class CalibratedLGBM:
    """
    Classifier + Ranker アンサンブル + Isotonic Calibration モデル (v5)

    - LightGBM Classifier: 個馬単体の top3 確率を推定
    - LightGBM Ranker: レース内の相対順位を直接学習（的中率向上に貢献）
    - アンサンブル: classifier_prob * α + normalized_ranker_score * (1-α)
    - Isotonic Regression: アンサンブルスコアを真の確率に補正
    """

    def __init__(
        self,
        base_model: lgb.LGBMClassifier,
        calibrator: IsotonicRegression = None,
        ranker: lgb.LGBMRanker = None,
        ranker_weight: float = 0.3,
        training_features: list = None,
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.ranker = ranker
        self.ranker_weight = ranker_weight
        # 学習時の特徴量リストを保存（予測時の不一致を防ぐ）
        self.training_features = training_features or list(base_model.feature_name_)

    def __getattr__(self, name):
        """旧モデル(pkl)に training_features がない場合の後方互換"""
        if name == "training_features":
            return list(self.base_model.feature_name_)
        raise AttributeError(name)

    def _align_features(self, X) -> "pd.DataFrame":
        """学習時の特徴量リストに合わせてXを整列・補完する"""
        import pandas as pd
        if hasattr(X, "columns"):
            missing = [f for f in self.training_features if f not in X.columns]
            if missing:
                for col in missing:
                    X = X.copy()
                    X[col] = np.nan
            return X[self.training_features]
        return X

    def _raw_ensemble_score(self, X) -> np.ndarray:
        """Classifier確率 + Rankerスコアのアンサンブルスコアを計算"""
        X = self._align_features(X)
        clf_prob = self.base_model.predict_proba(X)[:, 1]

        if self.ranker is not None:
            ranker_score = self.ranker.predict(X)  # X already aligned above
            # Rankerスコアを[0,1]に正規化（シグモイド変換）
            ranker_score_norm = 1 / (1 + np.exp(-ranker_score / ranker_score.std().clip(0.01)))
            score = (1 - self.ranker_weight) * clf_prob + self.ranker_weight * ranker_score_norm
        else:
            score = clf_prob

        return score

    def predict_proba(self, X):
        """キャリブレーション済み確率を返す（sklearn互換インターフェース）"""
        raw_score = self._raw_ensemble_score(X)
        if self.calibrator is not None:
            calibrated = self.calibrator.predict(raw_score)
            calibrated = np.clip(calibrated, 0.01, 0.99)
        else:
            calibrated = np.clip(raw_score, 0.01, 0.99)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        pred = self.predict(X)
        return (pred == y).mean()

    @property
    def feature_importances_(self):
        return self.base_model.feature_importances_

    @property
    def feature_name_(self):
        return self.base_model.feature_name_


def train_model(
    df: pd.DataFrame,
    calibrate: bool = True,
    ranker_weight: float = 0.3,
) -> CalibratedLGBM:
    """
    Classifier + Ranker アンサンブルモデルを学習する（v5）。

    データ分割:
        - 学習: 0% ~ 70%
        - キャリブレーション: 70% ~ 85%
        - テスト: 85% ~ 100%
    """
    df = prepare_dataset(df)
    features = get_available_features(df)

    # 時系列で3分割
    n = len(df)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    cal_df = df.iloc[train_end:cal_end]
    test_df = df.iloc[cal_end:]

    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_cal = cal_df[features]
    y_cal = cal_df[TARGET_COLUMN]
    X_test = test_df[features]
    y_test = test_df[TARGET_COLUMN]

    print(f"学習データ: {len(train_df)}件, キャリブレーション: {len(cal_df)}件, テスト: {len(test_df)}件")
    print(f"特徴量数: {len(features)}")

    # ── 1. Classifier 学習 ──
    print("\n--- [1/2] LightGBM Classifier 学習 ---")
    base_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=40,
        subsample=0.75,
        colsample_bytree=0.65,
        min_child_samples=80,
        reg_alpha=0.3,
        reg_lambda=0.5,
        min_split_gain=0.01,
        max_bin=200,
        random_state=42,
        verbose=-1,
    )
    base_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )

    # ── 2. Ranker 学習 ──
    print("\n--- [2/2] LightGBM Ranker 学習（レース内順位直接学習）---")
    ranker = _train_ranker(train_df, features)

    # ── 3. アンサンブル → キャリブレーション ──
    calibrator = None
    if calibrate:
        print("\n--- アンサンブル確率キャリブレーション (Isotonic Regression) ---")

        # キャリブレーションデータでアンサンブルスコアを計算
        clf_cal = base_model.predict_proba(X_cal)[:, 1]
        if ranker is not None:
            ranker_cal = ranker.predict(X_cal)
            ranker_cal_norm = 1 / (1 + np.exp(-ranker_cal / ranker_cal.std().clip(0.01)))
            raw_ensemble_cal = (1 - ranker_weight) * clf_cal + ranker_weight * ranker_cal_norm
        else:
            raw_ensemble_cal = clf_cal

        calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        calibrator.fit(raw_ensemble_cal, y_cal)

        # テストデータで比較
        clf_test = base_model.predict_proba(X_test)[:, 1]
        if ranker is not None:
            ranker_test = ranker.predict(X_test)
            ranker_test_norm = 1 / (1 + np.exp(-ranker_test / ranker_test.std().clip(0.01)))
            raw_ensemble_test = (1 - ranker_weight) * clf_test + ranker_weight * ranker_test_norm
        else:
            raw_ensemble_test = clf_test

        cal_test = calibrator.predict(raw_ensemble_test)
        cal_test = np.clip(cal_test, 0.01, 0.99)
        _show_calibration_comparison(y_test, raw_ensemble_test, cal_test)

    model = CalibratedLGBM(base_model, calibrator, ranker, ranker_weight,
                           training_features=features)

    accuracy = model.score(X_test, y_test)
    print(f"\nテストデータ正解率: {accuracy:.4f}")

    # AUC
    from sklearn.metrics import roc_auc_score
    test_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)
    print(f"テストデータAUC: {auc:.4f}")

    # 特徴量重要度を表示
    print("\n--- 特徴量重要度 TOP 20 ---")
    importance = pd.Series(base_model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)
    for feat, imp in importance.head(20).items():
        print(f"  {feat:40s} {imp:6d}")

    return model


def _train_ranker(train_df: pd.DataFrame, features: list) -> lgb.LGBMRanker | None:
    """
    LightGBM Ranker を学習する。

    Rankerはレース内での相対順位を直接最適化するため、
    「このレースで3着以内に来る馬」の識別精度が Classifier より高くなる。

    LambdaMART アルゴリズムを使用。group はレースごとの頭数。
    """
    try:
        # レースごとのグループサイズ（Ranker に必要）
        group_sizes = train_df.groupby("race_id").size().values

        X_rank = train_df[features]
        # Rankerのラベル: 1着=3, 2着=2, 3着=1, それ以外=0
        y_rank = np.where(
            train_df["finish_position"] == 1, 3,
            np.where(train_df["finish_position"] == 2, 2,
            np.where(train_df["finish_position"] == 3, 1, 0))
        )

        ranker = lgb.LGBMRanker(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.3,
            random_state=42,
            verbose=-1,
        )
        ranker.fit(X_rank, y_rank, group=group_sizes)
        print("  Ranker学習完了")
        return ranker
    except Exception as e:
        print(f"  Ranker学習失敗（Classifierのみで継続）: {e}")
        return None


def _show_calibration_comparison(y_true, raw_proba, cal_proba):
    """キャリブレーション前後の予測精度を比較表示する"""
    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    labels = ["0-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.8", "0.8-1.0"]

    print(f"  {'予測範囲':>10} | {'件数':>6} | {'実際':>6} | {'補正前':>6} | {'補正後':>6} | {'改善':>6}")
    print(f"  {'-'*58}")

    for i in range(len(labels)):
        mask = (raw_proba >= bins[i]) & (raw_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        actual = y_true.values[mask].mean()
        raw_mean = raw_proba[mask].mean()
        cal_mean = cal_proba[mask].mean()
        gap_before = abs(raw_mean - actual)
        gap_after = abs(cal_mean - actual)
        improved = "OK" if gap_after < gap_before else ""
        print(f"  {labels[i]:>10} | {mask.sum():>6} | {actual:>6.3f} | {raw_mean:>6.3f} | {cal_mean:>6.3f} | {improved:>6}")


def save_model(model, name: str = "lightgbm_v5"):
    """モデルを保存する（CalibratedLGBM対応）"""
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"モデルを保存しました: {path}")


def load_model(name: str = "lightgbm_v5"):
    """モデルを読み込む（CalibratedLGBM / LGBMClassifier 両対応）"""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
