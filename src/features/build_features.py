"""特徴量エンジニアリング v3 — 高速ベクトル化版"""

import sqlite3
import pandas as pd
import numpy as np
from config.settings import DB_PATH
from src.features.track_bias import add_track_bias_features


# ============================================================
#  データ読み込み
# ============================================================

def load_results(db_path=DB_PATH) -> pd.DataFrame:
    """全結果データを読み込む（実複勝払戻・調教師も結合）"""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            r.race_id, r.horse_id, r.jockey_id, r.trainer_id,
            r.post_number, r.gate_number,
            r.odds, r.popularity, r.weight_carried,
            r.horse_weight, r.weight_change,
            r.finish_position, r.finish_time_sec, r.last_3f,
            r.passing_order, r.prize,
            rc.date, rc.venue, rc.surface, rc.distance,
            rc.condition, rc.grade, rc.head_count,
            h.sire, h.dam_sire,
            pf.payout AS fukusho_payout,
            pt.payout AS tansho_payout
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        LEFT JOIN horses h ON r.horse_id = h.horse_id
        LEFT JOIN payouts pf ON r.race_id = pf.race_id
            AND r.post_number = pf.horse_number
            AND pf.bet_type = 'fukusho'
        LEFT JOIN payouts pt ON r.race_id = pt.race_id
            AND r.post_number = pt.horse_number
            AND pt.bet_type = 'tansho'
        WHERE r.finish_position > 0
          -- 障害レース除外（既存DBに残っている場合のセーフティネット）
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND NOT (rc.distance >= 3000 AND (
                rc.title LIKE '%J' OR rc.title LIKE '%JS' OR rc.title LIKE '%GJ'
          ))
          AND rc.surface != '障害'
        ORDER BY rc.date, r.race_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])

    # 実複勝オッズ（100円あたりの払戻 → 倍率に変換）
    # 3着以内で払戻データがある場合: payout / 100
    # 3着以内で払戻データがない場合: 概算 odds * 0.3
    # 4着以下: 0（ハズレ）
    df["fukusho_odds_actual"] = np.where(
        df["finish_position"] <= 3,
        np.where(
            df["fukusho_payout"].notna(),
            df["fukusho_payout"] / 100,               # 実データ
            (df["odds"] * 0.3).clip(lower=1.1),        # 概算フォールバック
        ),
        0,
    )
    df["tansho_payout_actual"] = np.where(
        df["tansho_payout"].notna(),
        df["tansho_payout"] / 100,
        np.nan,
    )

    return df


# ============================================================
#  ヘルパー: グループ内のshift→rolling/expanding を高速に計算
# ============================================================

def _grouped_shift_rolling(df, group_col, value_col, window, agg="mean", min_periods=1):
    """group_col でグループ化し、value_col の shift(1)→rolling を計算"""
    sorted_df = df.sort_values(["date", "race_id"])
    shifted = sorted_df.groupby(group_col)[value_col].shift(1)
    return shifted.groupby(sorted_df[group_col]).rolling(window, min_periods=min_periods).agg(agg).droplevel(0).reindex(df.index)


def _grouped_shift_expanding(df, group_col, value_col, agg="mean"):
    """group_col でグループ化し、value_col の shift(1)→expanding を計算"""
    sorted_df = df.sort_values(["date", "race_id"])
    shifted = sorted_df.groupby(group_col)[value_col].shift(1)
    return shifted.groupby(sorted_df[group_col]).expanding().agg(agg).droplevel(0).reindex(df.index)


# ============================================================
#  1. スピード指数
# ============================================================

def add_speed_index(df: pd.DataFrame) -> pd.DataFrame:
    """スピード指数を計算する（ベクトル化版）"""
    df = df.copy()
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    df["_cond_key"] = df["surface"] + "_" + df["distance"].astype(str) + "_" + df["condition"].fillna("")

    # 各条件グループの累積平均・標準偏差（shift(1)でデータリーク防止）
    shifted = df.groupby("_cond_key")["finish_time_sec"].shift(1)
    df["_base_time"] = shifted.groupby(df["_cond_key"]).expanding(min_periods=10).mean().droplevel(0).reindex(df.index)
    df["_base_std"] = shifted.groupby(df["_cond_key"]).expanding(min_periods=10).std().droplevel(0).reindex(df.index)

    df["speed_index"] = np.where(
        df["_base_std"] > 0,
        (df["_base_time"] - df["finish_time_sec"]) / df["_base_std"] * 10 + 50,
        np.nan,
    )

    df.drop(columns=["_cond_key", "_base_time", "_base_std"], inplace=True)
    return df


# ============================================================
#  2. 馬の過去成績（ベクトル化版）
# ============================================================

def add_horse_history_features(df: pd.DataFrame, n_races: int = 5) -> pd.DataFrame:
    """馬の過去N走の成績から特徴量を作成する（ベクトル化版）"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    g = "horse_id"

    # 基本統計 (過去N走の rolling mean)
    for col_name, col in [
        ("avg_finish", "finish_position"),
        ("avg_last_3f", "last_3f"),
        ("avg_time", "finish_time_sec"),
        ("avg_speed_idx", "speed_index"),
    ]:
        df[f"horse_{col_name}_{n_races}"] = _grouped_shift_rolling(df, g, col, n_races)

    # 勝率・複勝率
    df["_is_win"] = (df["finish_position"] == 1).astype(float)
    df["_is_top3"] = (df["finish_position"] <= 3).astype(float)
    df[f"horse_win_rate_{n_races}"] = _grouped_shift_rolling(df, g, "_is_win", n_races)
    df[f"horse_top3_rate_{n_races}"] = _grouped_shift_rolling(df, g, "_is_top3", n_races)

    # 調子トレンド（3走平均 - 5走平均）
    avg_3 = _grouped_shift_rolling(df, g, "finish_position", 3, min_periods=2)
    avg_5 = _grouped_shift_rolling(df, g, "finish_position", 5, min_periods=3)
    df["horse_form_trend"] = avg_3 - avg_5

    # 前走着順・前走スピード指数
    df["horse_last_finish"] = df.groupby(g)["finish_position"].shift(1)
    df["horse_last_speed_idx"] = df.groupby(g)["speed_index"].shift(1)

    # 最高スピード指数 (過去N走)
    df[f"horse_best_speed_idx_{n_races}"] = _grouped_shift_rolling(df, g, "speed_index", n_races, agg="max")

    # 出走回数
    df["horse_race_count"] = df.groupby(g).cumcount()

    # 前走からの間隔
    df["days_since_last"] = df.groupby(g)["date"].diff().dt.days

    # 上がり3Fの安定度
    df[f"horse_last3f_std_{n_races}"] = _grouped_shift_rolling(df, g, "last_3f", n_races, agg="std", min_periods=2)

    # 賞金累計
    df["horse_total_prize"] = _grouped_shift_expanding(df, g, "prize", agg="sum")
    df["horse_avg_prize"] = _grouped_shift_expanding(df, g, "prize", agg="mean")

    df.drop(columns=["_is_win", "_is_top3"], inplace=True)
    return df


# ============================================================
#  3. 条件別複勝率（ベクトル化版）
# ============================================================

def _calc_conditional_top3_rate(df: pd.DataFrame, group_col: str, result_col: str) -> pd.Series:
    """馬ごと・条件別の過去複勝率を高速に計算（ベクトル化版）"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    is_top3 = (df["finish_position"] <= 3).astype(float)

    # horse_id × group_col で二重グループ化
    composite_key = df["horse_id"].astype(str) + "_" + df[group_col].astype(str)

    shifted = is_top3.groupby(composite_key).shift(1)
    result = shifted.groupby(composite_key).expanding(min_periods=1).mean().droplevel(0).reindex(df.index)
    return result


def add_distance_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬の距離帯別成績を計算する（ベクトル化版）"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    df["_dist_cat"] = pd.cut(
        df["distance"],
        bins=[0, 1400, 1800, 2200, 9999],
        labels=["sprint", "mile", "middle", "long"],
    ).astype(str)

    # 条件別複勝率
    df["horse_dist_top3_rate"] = _calc_conditional_top3_rate(df, "_dist_cat", "horse_dist_top3_rate")

    # 同距離帯でのレース数
    composite_key = df["horse_id"].astype(str) + "_" + df["_dist_cat"]
    df["horse_dist_race_count"] = df.groupby(composite_key).cumcount()

    # 距離変更
    df["distance_diff"] = df.groupby("horse_id")["distance"].diff()

    df.drop(columns=["_dist_cat"], inplace=True)
    return df


# ============================================================
#  4. 馬場適性（ベクトル化版）
# ============================================================

def add_surface_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬の芝/ダート別成績を計算する"""
    df["horse_surface_top3_rate"] = _calc_conditional_top3_rate(df, "surface", "horse_surface_top3_rate")
    return df


# ============================================================
#  5. 競馬場適性（ベクトル化版）
# ============================================================

def add_venue_aptitude(df: pd.DataFrame) -> pd.DataFrame:
    """馬の競馬場別成績を計算する"""
    df["horse_venue_top3_rate"] = _calc_conditional_top3_rate(df, "venue", "horse_venue_top3_rate")
    return df


# ============================================================
#  6. ペース分析（脚質）（ベクトル化版）
# ============================================================

def add_pace_features(df: pd.DataFrame) -> pd.DataFrame:
    """通過順から脚質・ポジション特徴量を作成する"""
    df = df.copy()

    # 通過順をパース
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

    df["early_position"] = df["pass_1"]
    df["position_change"] = df["pass_1"] - df["finish_position"]

    # 馬ごとの平均脚質（ベクトル化）
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["horse_avg_early_pos"] = _grouped_shift_expanding(df, "horse_id", "early_position")
    df["horse_avg_pos_change"] = _grouped_shift_expanding(df, "horse_id", "position_change")

    return df


# ============================================================
#  7. 騎手の特徴量（ベクトル化版）
# ============================================================

def add_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """調教師の通算・条件別成績から特徴量を作成する"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # trainer_id が未取得（NaN）の場合はNaNのまま返す
    if "trainer_id" not in df.columns or df["trainer_id"].isna().all():
        df["trainer_win_rate"] = np.nan
        df["trainer_top3_rate"] = np.nan
        df["trainer_recent_top3_20"] = np.nan
        df["trainer_venue_top3"] = np.nan
        return df

    # NaNのtrainer_idは "__unknown__" で埋めてグループ化可能にする
    df["trainer_id"] = df["trainer_id"].fillna("__unknown__")

    g = "trainer_id"
    df["_t_is_win"] = (df["finish_position"] == 1).astype(float)
    df["_t_is_top3"] = (df["finish_position"] <= 3).astype(float)

    # 通算成績
    df["trainer_win_rate"] = _grouped_shift_expanding(df, g, "_t_is_win")
    df["trainer_top3_rate"] = _grouped_shift_expanding(df, g, "_t_is_top3")

    # 直近20走
    df["trainer_recent_top3_20"] = _grouped_shift_rolling(df, g, "_t_is_top3", 20, min_periods=5)

    # 調教師×競馬場
    composite_key = df["trainer_id"].astype(str) + "_" + df["venue"].astype(str)
    shifted = df["_t_is_top3"].groupby(composite_key).shift(1)
    df["trainer_venue_top3"] = (
        shifted.groupby(composite_key).expanding(min_periods=1).mean()
        .droplevel(0).reindex(df.index)
    )

    # __unknown__ はNaNに戻す
    unknown_mask = df["trainer_id"] == "__unknown__"
    for col in ["trainer_win_rate", "trainer_top3_rate", "trainer_recent_top3_20", "trainer_venue_top3"]:
        df.loc[unknown_mask, col] = np.nan
    df.loc[unknown_mask, "trainer_id"] = np.nan

    df.drop(columns=["_t_is_win", "_t_is_top3"], inplace=True)
    return df


def add_jockey_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手の通算・条件別成績から特徴量を作成する"""
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    g = "jockey_id"
    df["_j_is_win"] = (df["finish_position"] == 1).astype(float)
    df["_j_is_top3"] = (df["finish_position"] <= 3).astype(float)

    # 通算成績
    df["jockey_win_rate"] = _grouped_shift_expanding(df, g, "_j_is_win")
    df["jockey_top3_rate"] = _grouped_shift_expanding(df, g, "_j_is_top3")

    # 直近20走
    df["jockey_recent_top3_20"] = _grouped_shift_rolling(df, g, "_j_is_top3", 20, min_periods=5)

    # 騎手×競馬場
    composite_key = df["jockey_id"].astype(str) + "_" + df["venue"].astype(str)
    shifted = df["_j_is_top3"].groupby(composite_key).shift(1)
    df["jockey_venue_top3"] = shifted.groupby(composite_key).expanding(min_periods=1).mean().droplevel(0).reindex(df.index)

    df.drop(columns=["_j_is_win", "_j_is_top3"], inplace=True)
    return df


# ============================================================
#  8. レースレベル特徴量（レース内での相対値）
# ============================================================

def add_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内での相対的な位置づけを計算する"""
    df = df.copy()

    for col in ["horse_avg_speed_idx_5", "horse_top3_rate_5", "jockey_win_rate", "jockey_top3_rate", "odds"]:
        if col not in df.columns:
            continue

        race_mean = df.groupby("race_id")[col].transform("mean")
        race_std = df.groupby("race_id")[col].transform("std")
        df[f"{col}_race_zscore"] = np.where(
            race_std > 0,
            (df[col] - race_mean) / race_std,
            0,
        )

        df[f"{col}_race_rank"] = df.groupby("race_id")[col].rank(pct=True)

    return df


# ============================================================
#  v4: 高相関特徴量の交互作用
# ============================================================

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    診断レポートで相関の高い特徴量の交互作用を追加する。

    - horse_top3_rate_5 (corr=+0.2824) × jockey_top3_rate (+0.2516) → 馬×騎手のシナジー
    - horse_dist_top3_rate (+0.2513) × jockey_top3_rate → 距離適性×騎手
    - horse_form_trend × jockey_recent_top3_20 → 近走の勢い
    - horse_avg_speed_idx_5 × horse_dist_top3_rate → スピード×距離適性
    """
    df = df.copy()

    # 馬の実力 × 騎手の実力（最も相関の高い2特徴量の積）
    h_top3 = df.get("horse_top3_rate_5", pd.Series(np.nan, index=df.index))
    j_top3 = df.get("jockey_top3_rate", pd.Series(np.nan, index=df.index))
    df["horse_jockey_synergy"] = h_top3 * j_top3

    # 距離適性 × 騎手の実力
    h_dist = df.get("horse_dist_top3_rate", pd.Series(np.nan, index=df.index))
    df["horse_dist_jockey"] = h_dist * j_top3

    # 馬の調子 × 騎手の直近調子
    h_form = df.get("horse_form_trend", pd.Series(0, index=df.index)).fillna(0)
    j_recent = df.get("jockey_recent_top3_20", pd.Series(np.nan, index=df.index))
    df["horse_form_x_jockey_recent"] = h_form * j_recent

    # スピード指数 × 距離適性
    h_speed = df.get("horse_avg_speed_idx_5", pd.Series(np.nan, index=df.index))
    df["horse_speed_x_dist_apt"] = h_speed * h_dist

    return df


# ============================================================
#  血統特徴量（種牡馬・母父の成績集計）
# ============================================================

def add_pedigree_features(df: pd.DataFrame) -> pd.DataFrame:
    """血統（sire / dam_sire）ごとの過去複勝率を特徴量化する。

    リーク防止のため shift(1) → expanding で計算する。
    学習データに sire/dam_sire が NULL の馬は NaN になる（LightGBM が適切にルーティング）。
    """
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    df["_is_top3_ped"] = (df["finish_position"] <= 3).astype(float)
    # sprint/mile (<=1800) か middle/long (>1800) かの2値カテゴリ
    df["_dist_short"] = (df["distance"] <= 1800).astype(int)

    sire = df["sire"].fillna("__unknown__")
    dam_sire = df["dam_sire"].fillna("__unknown__")

    def _expanding_rate(group_key: pd.Series, min_periods: int) -> pd.Series:
        shifted = df["_is_top3_ped"].groupby(group_key).shift(1)
        return (
            shifted.groupby(group_key)
            .expanding(min_periods=min_periods)
            .mean()
            .droplevel(0)
            .reindex(df.index)
        )

    # sire 通算複勝率
    df["sire_top3_rate"] = _expanding_rate(sire, min_periods=30)

    # sire × 馬場（芝/ダート）
    key = sire.astype(str) + "_" + df["surface"].astype(str)
    df["sire_surface_top3"] = _expanding_rate(key, min_periods=20)

    # sire × 距離帯（短中 / 中長）
    key = sire.astype(str) + "_" + df["_dist_short"].astype(str)
    df["sire_distcat_top3"] = _expanding_rate(key, min_periods=20)

    # dam_sire 通算複勝率
    df["dam_sire_top3_rate"] = _expanding_rate(dam_sire, min_periods=30)

    # dam_sire × 馬場
    key = dam_sire.astype(str) + "_" + df["surface"].astype(str)
    df["dam_sire_surface_top3"] = _expanding_rate(key, min_periods=20)

    # 未取得の血統は NaN に戻す（"__unknown__" の集計結果は信頼できない）
    sire_na = df["sire"].isna()
    for col in ["sire_top3_rate", "sire_surface_top3", "sire_distcat_top3"]:
        df.loc[sire_na, col] = np.nan
    dam_na = df["dam_sire"].isna()
    for col in ["dam_sire_top3_rate", "dam_sire_surface_top3"]:
        df.loc[dam_na, col] = np.nan

    df.drop(columns=["_is_top3_ped", "_dist_short"], inplace=True)
    return df


# ============================================================
#  9. 休み明けパターン
# ============================================================

def add_rest_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """休養パターンの特徴量"""
    df = df.copy()

    df["rest_category"] = pd.cut(
        df["days_since_last"],
        bins=[0, 14, 35, 70, 180, 9999],
        labels=["連闘", "通常", "間隔空", "休み明け", "長期休養"],
    )

    df["is_fresh"] = (df["days_since_last"] >= 70).astype(float)

    return df


# ============================================================
#  10. 基本特徴量
# ============================================================

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本的な特徴量を追加する"""
    df["is_inner_gate"] = (df["gate_number"] <= 4).astype(int)

    condition_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    df["condition_num"] = df["condition"].map(condition_map)

    df["weight_ratio"] = np.where(
        df["horse_weight"] > 0,
        df["weight_carried"] / df["horse_weight"] * 100,
        np.nan,
    )

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
    """全特徴量を構築してDataFrameを返す（ベクトル化高速版）"""
    import time

    t0 = time.time()
    print("データ読み込み中...")
    df = load_results(db_path)
    print(f"  → {len(df)}件の出走データ")

    def _step(label):
        elapsed = time.time() - t0
        print(f"{label} ({elapsed:.1f}s)")

    _step("[1/9] スピード指数を計算中...")
    df = add_speed_index(df)

    _step("[2/9] 馬の過去成績特徴量を作成中...")
    df = add_horse_history_features(df)

    _step("[3/9] 距離適性を計算中...")
    df = add_distance_aptitude(df)

    _step("[4/9] 馬場適性を計算中...")
    df = add_surface_aptitude(df)

    _step("[5/9] 競馬場適性を計算中...")
    df = add_venue_aptitude(df)

    _step("[6/9] ペース特徴量を作成中...")
    df = add_pace_features(df)

    _step("[7/11] 騎手の特徴量を作成中...")
    df = add_jockey_features(df)

    _step("[8/11] 調教師の特徴量を作成中...")
    df = add_trainer_features(df)

    _step("[9/11] 基本特徴量・レースレベル特徴量を追加中...")
    df = add_basic_features(df)
    df = add_rest_pattern(df)
    df = add_race_level_features(df)

    _step("[10/12] 馬場傾向（前日バイアス）を計算中...")
    df = add_track_bias_features(df, db_path)

    _step("[11/12] 血統特徴量を計算中...")
    df = add_pedigree_features(df)

    _step("[12/12] 交互作用特徴量を追加中...")
    df = add_interaction_features(df)

    total = time.time() - t0
    print(f"特徴量構築完了 (列数: {len(df.columns)}, 所要時間: {total:.1f}秒)")
    return df
