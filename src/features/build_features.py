"""特徴量エンジニアリング v3 — 馬場傾向対応版"""

import sqlite3
import pandas as pd
import numpy as np
from config.settings import DB_PATH
from src.features.track_bias import add_track_bias_features


# ============================================================
#  データ読み込み
# ============================================================

def load_results(db_path=DB_PATH) -> pd.DataFrame:
    """全結果データを読み込む"""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            r.race_id, r.horse_id, r.jockey_id,
            r.post_number, r.gate_number,
            r.odds, r.popularity, r.weight_carried,
            r.horse_weight, r.weight_change,
            r.finish_position, r.finish_time_sec, r.last_3f,
            r.passing_order, r.prize,
            rc.date, rc.venue, rc.surface, rc.distance,
            rc.condition, rc.grade, rc.head_count,
            h.sire, h.dam_sire
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        LEFT JOIN horses h ON r.horse_id = h.horse_id
        WHERE r.finish_position > 0
        ORDER BY rc.date, r.race_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
#  1. スピード指数
# ============================================================

def add_speed_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    スピード指数を計算する。
    基準タイム（同距離・同馬場・同コンディションの平均タイム）との差を偏差値化。
    """
    df = df.copy()

    # 基準タイム = 同条件（芝/ダート × 距離 × 馬場状態）の平均タイム
    # ※データリーク防止: 各レース時点での累積平均を使う
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # 条件キー
    df["_cond_key"] = df["surface"] + "_" + df["distance"].astype(str) + "_" + df["condition"].fillna("")

    # 各条件グループの累積平均・標準偏差
    group_stats = []
    for _, group in df.groupby("_cond_key"):
        group = group.sort_values("date")
        shifted_time = group["finish_time_sec"].shift(1)
        group["_base_time"] = shifted_time.expanding(min_periods=10).mean()
        group["_base_std"] = shifted_time.expanding(min_periods=10).std()
        group_stats.append(group)

    df = pd.concat(group_stats).sort_values(["date", "race_id"]).reset_index(drop=True)

    # スピード指数 = (基準タイム - 実タイム) / 標準偏差 × 10 + 50
    # タイムが速い(小さい)ほど指数が高い
    df["speed_index"] = np.where(
        df["_base_std"] > 0,
        (df["_base_time"] - df["finish_time_sec"]) / df["_base_std"] * 10 + 50,
        np.nan,
    )

    df.drop(columns=["_cond_key", "_base_time", "_base_std"], inplace=True)
    return df


# ============================================================
#  2. 馬の過去成績（基本 + 条件別）
# ============================================================

def add_horse_history_features(df: pd.DataFrame, n_races: int = 5) -> pd.DataFrame:
    """馬の過去N走の成績から特徴量を作成する"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    features = []
    for _, group in df.groupby("horse_id"):
        group = group.sort_values("date")
        shifted_pos = group["finish_position"].shift(1)

        # --- 基本統計 (過去N走) ---
        for col_name, col in [
            ("avg_finish", "finish_position"),
            ("avg_last_3f", "last_3f"),
            ("avg_time", "finish_time_sec"),
            ("avg_speed_idx", "speed_index"),
        ]:
            group[f"horse_{col_name}_{n_races}"] = (
                group[col].shift(1).rolling(n_races, min_periods=1).mean()
            )

        # 勝率・複勝率
        group[f"horse_win_rate_{n_races}"] = (
            (shifted_pos == 1).rolling(n_races, min_periods=1).mean()
        )
        group[f"horse_top3_rate_{n_races}"] = (
            (shifted_pos <= 3).rolling(n_races, min_periods=1).mean()
        )

        # --- 直近の調子（トレンド）---
        # 最近3走の平均着順 vs 過去5走の平均着順 → 差が負=好調
        avg_3 = group["finish_position"].shift(1).rolling(3, min_periods=2).mean()
        avg_5 = group["finish_position"].shift(1).rolling(5, min_periods=3).mean()
        group["horse_form_trend"] = avg_3 - avg_5  # 負=改善中、正=下降中

        # 前走着順
        group["horse_last_finish"] = group["finish_position"].shift(1)

        # 前走スピード指数
        group["horse_last_speed_idx"] = group["speed_index"].shift(1)

        # 最高スピード指数 (過去N走)
        group[f"horse_best_speed_idx_{n_races}"] = (
            group["speed_index"].shift(1).rolling(n_races, min_periods=1).max()
        )

        # 出走回数
        group["horse_race_count"] = range(len(group))

        # 前走からの間隔（日数）
        group["days_since_last"] = group["date"].diff().dt.days

        # --- 上がり3Fの安定度 ---
        group[f"horse_last3f_std_{n_races}"] = (
            group["last_3f"].shift(1).rolling(n_races, min_periods=2).std()
        )

        # --- 賞金累計（クラス指標）---
        group["horse_total_prize"] = group["prize"].shift(1).expanding().sum()
        group["horse_avg_prize"] = group["prize"].shift(1).expanding().mean()

        features.append(group)

    return pd.concat(features).sort_values(["date", "race_id"]).reset_index(drop=True)


# ============================================================
#  3. 距離適性（高速版）
# ============================================================

def _calc_conditional_top3_rate(df: pd.DataFrame, group_col: str, result_col: str) -> pd.Series:
    """
    馬ごと・条件別の過去複勝率を高速に計算する。
    group_col の値が同じ過去レースのみで複勝率を計算（データリーク防止済み）。
    """
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_is_top3"] = (df["finish_position"] <= 3).astype(float)

    # 馬×条件 でグループ化し、累積で計算
    result = pd.Series(np.nan, index=df.index)

    for _, group in df.groupby(["horse_id", group_col]):
        group = group.sort_values("date")
        # shift(1)で今回を除外、expanding()で累積平均
        cum_top3 = group["_is_top3"].shift(1).expanding(min_periods=1).mean()
        result.iloc[group.index] = cum_top3.values

    df.drop(columns=["_is_top3"], inplace=True)
    return result


def add_distance_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬の距離帯別成績を計算する（高速版）"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # 距離カテゴリ
    df["_dist_cat"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=["sprint", "mile", "middle", "long"],
    ).astype(str)

    # 条件別の複勝率
    df["horse_dist_top3_rate"] = _calc_conditional_top3_rate(df, "_dist_cat", "horse_dist_top3_rate")

    # 同距離帯でのレース数
    count_series = pd.Series(0, index=df.index, dtype=int)
    for _, group in df.groupby(["horse_id", "_dist_cat"]):
        group = group.sort_values("date")
        cum_count = pd.Series(range(len(group)), index=group.index)
        count_series.iloc[group.index] = cum_count.values
    df["horse_dist_race_count"] = count_series

    # 距離変更（前走との差）
    dist_diff = []
    for _, group in df.groupby("horse_id"):
        group = group.sort_values("date")
        dist_diff.append(group["distance"].diff())
    df["distance_diff"] = pd.concat(dist_diff).reindex(df.index)

    df.drop(columns=["_dist_cat"], inplace=True)
    return df


# ============================================================
#  4. 馬場適性（高速版）
# ============================================================

def add_surface_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬の芝/ダート別成績を計算する（高速版）"""
    df["horse_surface_top3_rate"] = _calc_conditional_top3_rate(df, "surface", "horse_surface_top3_rate")
    return df


# ============================================================
#  5. 競馬場適性（高速版）
# ============================================================

def add_venue_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬の競馬場別成績を計算する（高速版）"""
    df["horse_venue_top3_rate"] = _calc_conditional_top3_rate(df, "venue", "horse_venue_top3_rate")
    return df


# ============================================================
#  6. ペース分析（脚質）
# ============================================================

def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """通過順から脚質・ポジション特徴量を作成する"""
    df = df.copy()

    # 通過順をパース (例: "3-3-2-1" → 各コーナーの位置)
    def parse_passing(s):
        if pd.isna(s) or s == "":
            return [np.nan] * 4
        parts = str(s).split("-")
        return [int(p) if p.isdigit() else np.nan for p in parts[:4]] + [np.nan] * (4 - len(parts))

    passing_cols = ["pass_1", "pass_2", "pass_3", "pass_4"]
    passing_df = df["passing_order"].apply(parse_passing).apply(pd.Series)
    passing_df.columns = passing_cols[:passing_df.shape[1]]
    for col in passing_cols:
        if col not in passing_df.columns:
            passing_df[col] = np.nan

    df[passing_cols] = passing_df[passing_cols]

    # 序盤の位置（1コーナー通過順）
    df["early_position"] = df["pass_1"]

    # ポジション変化 (序盤→終盤の上がり幅。負=追い込み型)
    df["position_change"] = df["pass_1"] - df["finish_position"]

    # 馬ごとの平均脚質（過去の通過順平均）
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    features = []
    for _, group in df.groupby("horse_id"):
        group = group.sort_values("date")
        group["horse_avg_early_pos"] = group["early_position"].shift(1).expanding().mean()
        group["horse_avg_pos_change"] = group["position_change"].shift(1).expanding().mean()
        features.append(group)

    return pd.concat(features).sort_values(["date", "race_id"]).reset_index(drop=True)


# ============================================================
#  7. 騎手の特徴量（強化版）
# ============================================================

def add_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手の通算・条件別成績から特徴量を作成する"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # --- 騎手の通算成績 ---
    features = []
    for _, group in df.groupby("jockey_id"):
        group = group.sort_values("date")
        shifted = group["finish_position"].shift(1)

        group["jockey_win_rate"] = (shifted == 1).expanding().mean()
        group["jockey_top3_rate"] = (shifted <= 3).expanding().mean()

        # 騎手の直近20走の調子
        group["jockey_recent_top3_20"] = (
            (shifted <= 3).rolling(20, min_periods=5).mean()
        )

        features.append(group)

    df = pd.concat(features).sort_values(["date", "race_id"]).reset_index(drop=True)

    # --- 騎手 × 競馬場 の相性 ---
    jv_features = []
    for _, group in df.groupby(["jockey_id", "venue"]):
        group = group.sort_values("date")
        shifted = group["finish_position"].shift(1)
        group["jockey_venue_top3"] = (shifted <= 3).expanding().mean()
        jv_features.append(group)

    df = pd.concat(jv_features).sort_values(["date", "race_id"]).reset_index(drop=True)

    return df


# ============================================================
#  8. レースレベル特徴量（レース内での相対値）
# ============================================================

def add_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内での相対的な位置づけを計算する"""
    df = df.copy()

    for col in ["horse_avg_speed_idx_5", "horse_top3_rate_5", "jockey_win_rate", "odds"]:
        if col not in df.columns:
            continue

        # レース内での偏差値（相対的な強さ）
        race_mean = df.groupby("race_id")[col].transform("mean")
        race_std = df.groupby("race_id")[col].transform("std")
        df[f"{col}_race_zscore"] = np.where(
            race_std > 0,
            (df[col] - race_mean) / race_std,
            0,
        )

        # レース内での順位（パーセンタイル）
        df[f"{col}_race_rank"] = df.groupby("race_id")[col].rank(pct=True)

    return df


# ============================================================
#  9. 休み明けパターン
# ============================================================

def add_rest_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """休養パターンの特徴量"""
    df = df.copy()

    # 休み明けカテゴリ
    df["rest_category"] = pd.cut(
        df["days_since_last"],
        bins=[0, 14, 35, 70, 180, 9999],
        labels=["連闘", "通常", "間隔空", "休み明け", "長期休養"],
    )

    # 休み明けフラグ
    df["is_fresh"] = (df["days_since_last"] >= 70).astype(float)

    return df


# ============================================================
#  10. 基本特徴量
# ============================================================

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本的な特徴量を追加する"""
    # 内枠/外枠
    df["is_inner_gate"] = (df["gate_number"] <= 4).astype(int)

    # 馬場状態を数値化
    condition_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    df["condition_num"] = df["condition"].map(condition_map)

    # 斤量と馬体重の比率
    df["weight_ratio"] = np.where(
        df["horse_weight"] > 0,
        df["weight_carried"] / df["horse_weight"] * 100,
        np.nan,
    )

    # 頭数に対する馬番の相対位置
    df["post_position_ratio"] = np.where(
        df["head_count"].fillna(0) > 0,
        df["post_number"].astype(float) / df["head_count"].astype(float),
        np.nan,
    )
    df["post_position_ratio"] = df["post_position_ratio"].astype(float)

    return df


# ============================================================
#  ビルドパイプライン
# ============================================================

def build_all_features(db_path=DB_PATH) -> pd.DataFrame:
    """全特徴量を構築してDataFrameを返す"""
    print("データ読み込み中...")
    df = load_results(db_path)
    print(f"  → {len(df)}件の出走データ")

    print("[1/8] スピード指数を計算中...")
    df = add_speed_index(df)

    print("[2/8] 馬の過去成績特徴量を作成中...")
    df = add_horse_history_features(df)

    print("[3/8] 距離適性を計算中...")
    df = add_distance_aptitude(df)

    print("[4/8] 馬場適性を計算中...")
    df = add_surface_aptitude(df)

    print("[5/8] 競馬場適性を計算中...")
    df = add_venue_aptitude(df)

    print("[6/8] ペース特徴量を作成中...")
    df = add_pace_features(df)

    print("[7/8] 騎手の特徴量を作成中...")
    df = add_jockey_features(df)

    print("[8/9] 基本特徴量・レースレベル特徴量を追加中...")
    df = add_basic_features(df)
    df = add_rest_pattern(df)
    df = add_race_level_features(df)

    print("[9/9] 馬場傾向（前日バイアス）を計算中...")
    df = add_track_bias_features(df, db_path)

    print(f"特徴量構築完了 (列数: {len(df.columns)})")
    return df
