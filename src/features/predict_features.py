"""
レース前の予測用特徴量を構築する

出馬表（エントリー情報）とDB内の過去データを組み合わせて、
モデルが必要とする全特徴量を計算する。
"""

import sqlite3
import numpy as np
import pandas as pd
from config.settings import DB_PATH
from src.features.track_bias import get_track_bias_for_date


def build_prediction_features(
    entries: list[dict],
    race_info: dict,
    db_path=DB_PATH,
) -> pd.DataFrame:
    """
    出馬表のエントリーからモデル入力用の特徴量DataFrameを構築する。

    Args:
        entries: 出馬表の各馬のエントリー情報
        race_info: レース基本情報 (surface, distance, venue, condition, date)

    Returns:
        予測用の特徴量DataFrame
    """
    conn = sqlite3.connect(db_path)
    rows = []

    for entry in entries:
        horse_id = entry.get("horse_id", "")
        jockey_id = entry.get("jockey_id", "")

        row = {
            "race_id": race_info.get("race_id"),
            "horse_id": horse_id,
            "jockey_id": jockey_id,
            "horse_name": entry.get("horse_name", ""),
            "jockey_name": entry.get("jockey_name", ""),
            "post_number": entry.get("post_number"),
            "gate_number": entry.get("gate_number"),
            "weight_carried": entry.get("weight_carried"),
            "horse_weight": entry.get("horse_weight"),
            "weight_change": entry.get("weight_change"),
            "distance": race_info.get("distance"),
            "surface": race_info.get("surface"),
            "venue": race_info.get("venue"),
            "condition": race_info.get("condition"),
            "date": race_info.get("date"),
            "head_count": race_info.get("head_count"),
        }

        # === 過去データから特徴量を計算 ===
        horse_features = _get_horse_features(conn, horse_id, race_info)
        row.update(horse_features)

        jockey_features = _get_jockey_features(conn, jockey_id, race_info)
        row.update(jockey_features)

        rows.append(row)

    conn.close()

    df = pd.DataFrame(rows)

    # 基本特徴量
    df = _add_basic_features(df)

    # 馬場傾向
    df = _add_track_bias(df, race_info, db_path)

    # レース内相対値
    df = _add_race_level_features(df)

    return df


def _get_horse_features(conn, horse_id: str, race_info: dict, n_races: int = 5) -> dict:
    """馬の過去成績から特徴量を計算する"""
    features = {}

    if not horse_id:
        return _empty_horse_features(n_races)

    # 過去の出走結果を取得
    query = """
        SELECT r.finish_position, r.finish_time_sec, r.last_3f,
               r.passing_order, r.prize, r.odds,
               rc.date, rc.distance, rc.surface, rc.venue, rc.condition
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.horse_id = ?
          AND r.finish_position > 0
        ORDER BY rc.date DESC
    """
    past = pd.read_sql_query(query, conn, params=[horse_id])

    if len(past) == 0:
        return _empty_horse_features(n_races)

    recent = past.head(n_races)

    # 基本統計（過去N走）
    features[f"horse_avg_finish_{n_races}"] = recent["finish_position"].mean()
    features[f"horse_avg_last_3f_{n_races}"] = recent["last_3f"].mean()
    features[f"horse_avg_time_{n_races}"] = recent["finish_time_sec"].mean()
    features[f"horse_win_rate_{n_races}"] = (recent["finish_position"] == 1).mean()
    features[f"horse_top3_rate_{n_races}"] = (recent["finish_position"] <= 3).mean()

    # スピード指数（簡易版: 過去N走のタイムから計算）
    if recent["finish_time_sec"].notna().any():
        times = recent["finish_time_sec"].dropna()
        if len(times) > 0:
            features[f"horse_avg_speed_idx_{n_races}"] = 50 + (times.mean() - times.iloc[0]) / max(times.std(), 0.1) * 10
            features[f"horse_best_speed_idx_{n_races}"] = 50 + (times.mean() - times.min()) / max(times.std(), 0.1) * 10
        else:
            features[f"horse_avg_speed_idx_{n_races}"] = np.nan
            features[f"horse_best_speed_idx_{n_races}"] = np.nan
    else:
        features[f"horse_avg_speed_idx_{n_races}"] = np.nan
        features[f"horse_best_speed_idx_{n_races}"] = np.nan

    # 前走
    features["horse_last_finish"] = past.iloc[0]["finish_position"]
    features["horse_last_speed_idx"] = features.get(f"horse_avg_speed_idx_{n_races}", np.nan)

    # 調子トレンド（最近3走 vs 5走）
    avg_3 = past.head(3)["finish_position"].mean() if len(past) >= 2 else np.nan
    avg_5 = recent["finish_position"].mean()
    features["horse_form_trend"] = avg_3 - avg_5 if not np.isnan(avg_3) else np.nan

    # 上がり3Fの安定度
    features[f"horse_last3f_std_{n_races}"] = recent["last_3f"].std() if len(recent) >= 2 else np.nan

    # 賞金
    features["horse_total_prize"] = past["prize"].sum()
    features["horse_avg_prize"] = past["prize"].mean()

    # 出走回数
    features["horse_race_count"] = len(past)

    # 前走からの間隔
    if len(past) > 0 and race_info.get("date"):
        try:
            race_date = pd.to_datetime(race_info["date"])
            last_date = pd.to_datetime(past.iloc[0]["date"])
            features["days_since_last"] = (race_date - last_date).days
        except Exception:
            features["days_since_last"] = np.nan
    else:
        features["days_since_last"] = np.nan

    # 距離適性
    race_dist = race_info.get("distance", 0)
    dist_cat = _distance_category(race_dist)
    same_dist = past[past["distance"].apply(_distance_category) == dist_cat]
    features["horse_dist_top3_rate"] = (same_dist["finish_position"] <= 3).mean() if len(same_dist) > 0 else np.nan
    features["horse_dist_race_count"] = len(same_dist)

    # 距離変更
    features["distance_diff"] = race_dist - past.iloc[0]["distance"] if len(past) > 0 else np.nan

    # 馬場適性
    race_surface = race_info.get("surface", "")
    same_surface = past[past["surface"] == race_surface]
    features["horse_surface_top3_rate"] = (same_surface["finish_position"] <= 3).mean() if len(same_surface) > 0 else np.nan

    # 競馬場適性
    race_venue = race_info.get("venue", "")
    same_venue = past[past["venue"] == race_venue]
    features["horse_venue_top3_rate"] = (same_venue["finish_position"] <= 3).mean() if len(same_venue) > 0 else np.nan

    # ペース・脚質
    first_pass_list = []
    pos_change_list = []
    for _, r in past.head(n_races).iterrows():
        po = r["passing_order"]
        if pd.notna(po) and po:
            parts = str(po).split("-")
            try:
                fp = int(parts[0])
                first_pass_list.append(fp)
                pos_change_list.append(fp - r["finish_position"])
            except (ValueError, IndexError):
                pass

    features["horse_avg_early_pos"] = np.mean(first_pass_list) if first_pass_list else np.nan
    features["horse_avg_pos_change"] = np.mean(pos_change_list) if pos_change_list else np.nan

    return features


def _get_jockey_features(conn, jockey_id: str, race_info: dict) -> dict:
    """騎手の過去成績から特徴量を計算する"""
    features = {}

    if not jockey_id:
        return _empty_jockey_features()

    query = """
        SELECT r.finish_position, rc.venue
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.jockey_id = ?
          AND r.finish_position > 0
        ORDER BY rc.date DESC
    """
    past = pd.read_sql_query(query, conn, params=[jockey_id])

    if len(past) == 0:
        return _empty_jockey_features()

    features["jockey_win_rate"] = (past["finish_position"] == 1).mean()
    features["jockey_top3_rate"] = (past["finish_position"] <= 3).mean()

    # 直近20走
    recent_20 = past.head(20)
    features["jockey_recent_top3_20"] = (recent_20["finish_position"] <= 3).mean()

    # 競馬場別
    race_venue = race_info.get("venue", "")
    venue_past = past[past["venue"] == race_venue]
    features["jockey_venue_top3"] = (venue_past["finish_position"] <= 3).mean() if len(venue_past) > 0 else np.nan

    return features


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """基本特徴量を追加"""
    df = df.copy()

    # 内枠/外枠
    df["is_inner_gate"] = (df["gate_number"] <= 4).astype(int)

    # 馬場状態数値化
    condition_map = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    df["condition_num"] = df["condition"].map(condition_map)

    # 斤量/馬体重比率
    df["weight_ratio"] = np.where(
        df["horse_weight"].fillna(0) > 0,
        df["weight_carried"].astype(float) / df["horse_weight"].astype(float) * 100,
        np.nan,
    )
    df["weight_ratio"] = df["weight_ratio"].astype(float)

    # 馬番/頭数比率
    df["post_position_ratio"] = np.where(
        df["head_count"].fillna(0) > 0,
        df["post_number"].astype(float) / df["head_count"].astype(float),
        np.nan,
    )
    df["post_position_ratio"] = df["post_position_ratio"].astype(float)

    # 休み明けフラグ
    df["is_fresh"] = (df["days_since_last"] >= 70).astype(float)

    return df


def _add_track_bias(df: pd.DataFrame, race_info: dict, db_path) -> pd.DataFrame:
    """前日の馬場傾向を追加"""
    df = df.copy()
    venue = race_info.get("venue", "")
    date = race_info.get("date", "")

    bias = get_track_bias_for_date(date, venue, db_path)

    df["bias_gate"] = bias["gate_bias"]
    df["bias_pace"] = bias["pace_bias"]
    df["bias_time"] = bias["time_bias"]
    df["bias_last3f"] = bias["last3f_bias"]
    df["bias_inner_top3"] = bias["inner_top3_rate"]
    df["bias_outer_top3"] = bias["outer_top3_rate"]
    df["bias_front_top3"] = bias["front_top3_rate"]
    df["bias_closer_top3"] = bias["closer_top3_rate"]

    # 交互作用
    df["bias_x_inner"] = df["is_inner_gate"] * df["bias_gate"]

    if "horse_avg_early_pos" in df.columns:
        df["bias_x_pace"] = (1.0 / df["horse_avg_early_pos"].clip(lower=1)) * df["bias_pace"]
    else:
        df["bias_x_pace"] = 0.0

    if "horse_avg_last_3f_5" in df.columns:
        race_mean_3f = df["horse_avg_last_3f_5"].mean()
        df["bias_x_last3f"] = (race_mean_3f - df["horse_avg_last_3f_5"]) * df["bias_last3f"]
    else:
        df["bias_x_last3f"] = 0.0

    return df


def _add_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース内での相対値を計算"""
    df = df.copy()

    for col in ["horse_avg_speed_idx_5", "horse_top3_rate_5", "jockey_win_rate"]:
        if col not in df.columns:
            df[f"{col}_race_zscore"] = 0.0
            df[f"{col}_race_rank"] = 0.5
            continue

        col_mean = df[col].mean()
        col_std = df[col].std()
        df[f"{col}_race_zscore"] = np.where(
            col_std > 0,
            (df[col] - col_mean) / col_std,
            0,
        )
        df[f"{col}_race_rank"] = df[col].rank(pct=True)

    # odds関連のレース内順位（オッズが不明な場合はスキップ）
    if "odds" in df.columns and df["odds"].notna().any():
        odds_mean = df["odds"].mean()
        odds_std = df["odds"].std()
        df["odds_race_zscore"] = np.where(
            odds_std > 0, (df["odds"] - odds_mean) / odds_std, 0
        )
        df["odds_race_rank"] = df["odds"].rank(pct=True)

    return df


def _distance_category(dist) -> str:
    """距離をカテゴリ化"""
    try:
        d = int(dist)
    except (ValueError, TypeError):
        return "unknown"
    if d <= 1400:
        return "sprint"
    elif d <= 1800:
        return "mile"
    elif d <= 2200:
        return "middle"
    else:
        return "long"


def _empty_horse_features(n_races: int = 5) -> dict:
    return {
        f"horse_avg_finish_{n_races}": np.nan,
        f"horse_avg_last_3f_{n_races}": np.nan,
        f"horse_avg_time_{n_races}": np.nan,
        f"horse_avg_speed_idx_{n_races}": np.nan,
        f"horse_win_rate_{n_races}": np.nan,
        f"horse_top3_rate_{n_races}": np.nan,
        f"horse_best_speed_idx_{n_races}": np.nan,
        "horse_last_finish": np.nan,
        "horse_last_speed_idx": np.nan,
        "horse_form_trend": np.nan,
        f"horse_last3f_std_{n_races}": np.nan,
        "horse_total_prize": 0,
        "horse_avg_prize": 0,
        "horse_race_count": 0,
        "days_since_last": np.nan,
        "horse_dist_top3_rate": np.nan,
        "horse_dist_race_count": 0,
        "distance_diff": np.nan,
        "horse_surface_top3_rate": np.nan,
        "horse_venue_top3_rate": np.nan,
        "horse_avg_early_pos": np.nan,
        "horse_avg_pos_change": np.nan,
    }


def _empty_jockey_features() -> dict:
    return {
        "jockey_win_rate": np.nan,
        "jockey_top3_rate": np.nan,
        "jockey_recent_top3_20": np.nan,
        "jockey_venue_top3": np.nan,
    }
