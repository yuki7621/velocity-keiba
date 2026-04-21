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


# ============================================================
# スピード指数の参照統計キャッシュ
# ============================================================
# build_features.py と整合するよう、(surface, distance, condition) ごとの
# finish_time_sec の平均と標準偏差を事前計算してメモリに保持する。
# build_features.py は expanding (累積) で計算するが、予測時は直近の
# 特徴量を求めるだけなので、全期間平均で十分な近似となる。
_speed_ref_cache: dict | None = None


def _load_speed_reference(conn) -> dict:
    """
    (surface, distance, condition) → {base_time, base_std} の辞書を返す。
    初回のみDBから計算してキャッシュする。
    """
    global _speed_ref_cache
    if _speed_ref_cache is not None:
        return _speed_ref_cache

    query = """
        SELECT rc.surface, rc.distance, rc.condition,
               r.finish_time_sec
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.finish_time_sec IS NOT NULL
          AND r.finish_time_sec > 0
          AND r.finish_position > 0
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND rc.surface != '障害'
    """
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        _speed_ref_cache = {}
        return _speed_ref_cache

    df["condition"] = df["condition"].fillna("")
    df["_key"] = df["surface"] + "_" + df["distance"].astype(str) + "_" + df["condition"]

    grouped = df.groupby("_key")["finish_time_sec"].agg(["mean", "std", "count"])
    grouped = grouped[grouped["count"] >= 10]  # build_features と揃える

    _speed_ref_cache = {
        k: {"base_time": row["mean"], "base_std": row["std"]}
        for k, row in grouped.iterrows()
    }
    return _speed_ref_cache


def _compute_speed_index(finish_time_sec: float, surface: str, distance: int,
                          condition: str, ref: dict) -> float:
    """build_features.py と同じ式でスピード指数を計算する

    speed_index = (base_time - finish_time) / base_std * 10 + 50
    """
    if pd.isna(finish_time_sec) or not finish_time_sec:
        return np.nan

    cond = condition if condition and not pd.isna(condition) else ""
    key = f"{surface}_{distance}_{cond}"
    stats = ref.get(key)
    if stats is None:
        return np.nan
    base_std = stats["base_std"]
    if base_std is None or pd.isna(base_std) or base_std <= 0:
        return np.nan
    return (stats["base_time"] - finish_time_sec) / base_std * 10 + 50


def clear_speed_reference_cache():
    """キャッシュをクリアする（テスト用）"""
    global _speed_ref_cache, _pedigree_stats_cache
    _speed_ref_cache = None
    _pedigree_stats_cache = None


# ============================================================
# 血統統計のキャッシュ
# ============================================================
# build_features.py の add_pedigree_features() と整合する集計を
# DB全体から事前計算してキャッシュする。
# 予測時は target_date より前の結果だけを使うのが厳密だが、
# 将来レース予測が主用途なので全期間集計でほぼ等価。
_pedigree_stats_cache: dict | None = None


def _load_pedigree_stats(conn) -> dict:
    """種牡馬・母父の成績集計を返す（初回のみDBから計算してキャッシュ）"""
    global _pedigree_stats_cache
    if _pedigree_stats_cache is not None:
        return _pedigree_stats_cache

    query = """
        SELECT h.sire, h.dam_sire, rc.surface, rc.distance,
               r.finish_position
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        LEFT JOIN horses h ON r.horse_id = h.horse_id
        WHERE r.finish_position > 0
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND NOT (rc.distance >= 3000 AND (
                rc.title LIKE '%J' OR rc.title LIKE '%JS' OR rc.title LIKE '%GJ'
          ))
          AND rc.surface != '障害'
    """
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        _pedigree_stats_cache = {
            "sire_overall": {}, "sire_surface": {}, "sire_distcat": {},
            "dam_sire_overall": {}, "dam_sire_surface": {},
        }
        return _pedigree_stats_cache

    df["is_top3"] = (df["finish_position"] <= 3).astype(float)
    df["distcat"] = (df["distance"] <= 1800).astype(int)

    cache = {
        "sire_overall": {},
        "sire_surface": {},
        "sire_distcat": {},
        "dam_sire_overall": {},
        "dam_sire_surface": {},
    }

    sire_df = df[df["sire"].notna() & (df["sire"] != "")]
    for name, grp in sire_df.groupby("sire"):
        if len(grp) >= 30:
            cache["sire_overall"][name] = float(grp["is_top3"].mean())
    for (name, surf), grp in sire_df.groupby(["sire", "surface"]):
        if len(grp) >= 20:
            cache["sire_surface"][(name, surf)] = float(grp["is_top3"].mean())
    for (name, dc), grp in sire_df.groupby(["sire", "distcat"]):
        if len(grp) >= 20:
            cache["sire_distcat"][(name, int(dc))] = float(grp["is_top3"].mean())

    dam_df = df[df["dam_sire"].notna() & (df["dam_sire"] != "")]
    for name, grp in dam_df.groupby("dam_sire"):
        if len(grp) >= 30:
            cache["dam_sire_overall"][name] = float(grp["is_top3"].mean())
    for (name, surf), grp in dam_df.groupby(["dam_sire", "surface"]):
        if len(grp) >= 20:
            cache["dam_sire_surface"][(name, surf)] = float(grp["is_top3"].mean())

    _pedigree_stats_cache = cache
    return cache


def _get_horse_pedigree(conn, horse_id: str) -> tuple[str | None, str | None]:
    """horses テーブルから sire, dam_sire を取得"""
    if not horse_id:
        return None, None
    cur = conn.execute(
        "SELECT sire, dam_sire FROM horses WHERE horse_id = ?",
        (horse_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None, None
    sire = row[0] if row[0] else None
    dam_sire = row[1] if row[1] else None
    return sire, dam_sire


def _get_pedigree_features(conn, horse_id: str, race_info: dict) -> dict:
    """血統特徴量を計算する"""
    sire, dam_sire = _get_horse_pedigree(conn, horse_id)
    stats = _load_pedigree_stats(conn)

    surface = race_info.get("surface", "")
    distance = race_info.get("distance", 0) or 0
    distcat = 1 if distance <= 1800 else 0

    features = {
        "sire_top3_rate": np.nan,
        "sire_surface_top3": np.nan,
        "sire_distcat_top3": np.nan,
        "dam_sire_top3_rate": np.nan,
        "dam_sire_surface_top3": np.nan,
    }

    if sire:
        features["sire_top3_rate"] = stats["sire_overall"].get(sire, np.nan)
        features["sire_surface_top3"] = stats["sire_surface"].get((sire, surface), np.nan)
        features["sire_distcat_top3"] = stats["sire_distcat"].get((sire, distcat), np.nan)

    if dam_sire:
        features["dam_sire_top3_rate"] = stats["dam_sire_overall"].get(dam_sire, np.nan)
        features["dam_sire_surface_top3"] = stats["dam_sire_surface"].get((dam_sire, surface), np.nan)

    return features


def build_prediction_features(
    entries: list[dict],
    race_info: dict,
    db_path=DB_PATH,
    impute_weight: bool = False,
) -> pd.DataFrame:
    """
    出馬表のエントリーからモデル入力用の特徴量DataFrameを構築する。

    Args:
        entries: 出馬表の各馬のエントリー情報
        race_info: レース基本情報 (surface, distance, venue, condition, date)
        impute_weight: True の場合、馬体重が未発表（None/NaN）なら
                       前走の馬体重で補完する（土曜夜の事前予測用）。
                       weight_change は 0 と仮定。

    Returns:
        予測用の特徴量DataFrame。
        impute_weight=True の場合、補完した行には `weight_imputed=True` が付く。
    """
    conn = sqlite3.connect(db_path)
    rows = []

    for entry in entries:
        horse_id = entry.get("horse_id", "")
        jockey_id = entry.get("jockey_id", "")
        trainer_id = entry.get("trainer_id", "")

        hw = entry.get("horse_weight")
        wc = entry.get("weight_change")
        imputed = False

        # 馬体重未発表時の補完
        if impute_weight and (hw is None or (isinstance(hw, float) and np.isnan(hw)) or hw == 0):
            last_hw = _get_last_horse_weight(conn, horse_id, target_date=race_info.get("date"))
            if last_hw is not None:
                hw = last_hw
                wc = 0  # 前走同体重と仮定
                imputed = True

        row = {
            "race_id": race_info.get("race_id"),
            "horse_id": horse_id,
            "jockey_id": jockey_id,
            "trainer_id": trainer_id,
            "horse_name": entry.get("horse_name", ""),
            "jockey_name": entry.get("jockey_name", ""),
            "trainer_name": entry.get("trainer_name", ""),
            "post_number": entry.get("post_number"),
            "gate_number": entry.get("gate_number"),
            "weight_carried": entry.get("weight_carried"),
            "horse_weight": hw,
            "weight_change": wc,
            "weight_imputed": imputed,
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

        trainer_features = _get_trainer_features(conn, trainer_id, race_info)
        row.update(trainer_features)

        pedigree_features = _get_pedigree_features(conn, horse_id, race_info)
        row.update(pedigree_features)

        rows.append(row)

    conn.close()

    df = pd.DataFrame(rows)

    # 基本特徴量
    df = _add_basic_features(df)

    # 馬場傾向
    df = _add_track_bias(df, race_info, db_path)

    # レース内相対値
    df = _add_race_level_features(df)

    # v4: 交互作用特徴量
    df = _add_interaction_features(df)

    return df


def _get_last_horse_weight(conn, horse_id: str, target_date: str | None = None) -> int | None:
    """
    指定した馬の直近の馬体重を返す。見つからない場合は None。

    土曜夜の予測で、まだ当日の馬体重が発表されていない時に
    前走の値で補完するために使う。

    target_date が指定された場合、その日より前のレースのみから検索する（リーク防止）。
    """
    if not horse_id:
        return None

    params: list = [horse_id]
    date_filter = ""
    if target_date:
        date_filter = "AND rc.date < ?"
        params.append(str(target_date))

    query = f"""
        SELECT r.horse_weight
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.horse_id = ?
          AND r.horse_weight IS NOT NULL
          AND r.horse_weight > 0
          {date_filter}
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND NOT (rc.distance >= 3000 AND (
                rc.title LIKE '%J' OR rc.title LIKE '%JS' OR rc.title LIKE '%GJ'
          ))
          AND rc.surface != '障害'
        ORDER BY rc.date DESC
        LIMIT 1
    """
    cur = conn.execute(query, params)
    row = cur.fetchone()
    if row is None:
        return None
    return int(row[0]) if row[0] else None


def _get_horse_features(conn, horse_id: str, race_info: dict, n_races: int = 5) -> dict:
    """馬の過去成績から特徴量を計算する"""
    features = {}

    if not horse_id:
        return _empty_horse_features(n_races)

    # 過去の出走結果を取得（障害レース除外 + 対象レース以前のみ = リーク防止）
    target_date = race_info.get("date")
    params = [horse_id]
    date_filter = ""
    if target_date:
        date_filter = "AND rc.date < ?"
        params.append(str(target_date))

    query = f"""
        SELECT r.finish_position, r.finish_time_sec, r.last_3f,
               r.passing_order, r.prize, r.odds,
               rc.date, rc.distance, rc.surface, rc.venue, rc.condition
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.horse_id = ?
          AND r.finish_position > 0
          {date_filter}
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND NOT (rc.distance >= 3000 AND (
                rc.title LIKE '%J' OR rc.title LIKE '%JS' OR rc.title LIKE '%GJ'
          ))
          AND rc.surface != '障害'
        ORDER BY rc.date DESC
    """
    past = pd.read_sql_query(query, conn, params=params)

    if len(past) == 0:
        return _empty_horse_features(n_races)

    # build_features.py と同じ式でスピード指数を各過去レースで計算
    speed_ref = _load_speed_reference(conn)
    past["speed_index"] = past.apply(
        lambda r: _compute_speed_index(
            r["finish_time_sec"], r["surface"], r["distance"],
            r["condition"], speed_ref,
        ),
        axis=1,
    )

    recent = past.head(n_races)

    # 基本統計（過去N走）
    features[f"horse_avg_finish_{n_races}"] = recent["finish_position"].mean()
    features[f"horse_avg_last_3f_{n_races}"] = recent["last_3f"].mean()
    features[f"horse_avg_time_{n_races}"] = recent["finish_time_sec"].mean()
    features[f"horse_win_rate_{n_races}"] = (recent["finish_position"] == 1).mean()
    features[f"horse_top3_rate_{n_races}"] = (recent["finish_position"] <= 3).mean()

    # スピード指数（build_features.py と整合した式で計算）
    si_recent = recent["speed_index"].dropna()
    if len(si_recent) > 0:
        features[f"horse_avg_speed_idx_{n_races}"] = si_recent.mean()
        features[f"horse_best_speed_idx_{n_races}"] = si_recent.max()
    else:
        features[f"horse_avg_speed_idx_{n_races}"] = np.nan
        features[f"horse_best_speed_idx_{n_races}"] = np.nan

    # 前走
    features["horse_last_finish"] = past.iloc[0]["finish_position"]
    # 前走スピード指数: 最新の(=0番目の)過去レースの speed_index
    features["horse_last_speed_idx"] = past.iloc[0]["speed_index"] if "speed_index" in past.columns else np.nan

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

    target_date = race_info.get("date")
    params = [jockey_id]
    date_filter = ""
    if target_date:
        date_filter = "AND rc.date < ?"
        params.append(str(target_date))

    query = f"""
        SELECT r.finish_position, rc.venue
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.jockey_id = ?
          AND r.finish_position > 0
          {date_filter}
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND NOT (rc.distance >= 3000 AND (
                rc.title LIKE '%J' OR rc.title LIKE '%JS' OR rc.title LIKE '%GJ'
          ))
          AND rc.surface != '障害'
        ORDER BY rc.date DESC
    """
    past = pd.read_sql_query(query, conn, params=params)

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


def _get_trainer_features(conn, trainer_id: str, race_info: dict) -> dict:
    """調教師の過去成績から特徴量を計算する"""
    if not trainer_id:
        return _empty_trainer_features()

    target_date = race_info.get("date")
    params = [trainer_id]
    date_filter = ""
    if target_date:
        date_filter = "AND rc.date < ?"
        params.append(str(target_date))

    query = f"""
        SELECT r.finish_position, rc.venue
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE r.trainer_id = ?
          AND r.finish_position > 0
          {date_filter}
          AND (rc.distance % 100 = 0 OR rc.distance = 1150)
          AND rc.title NOT LIKE '%障害%'
          AND rc.title NOT LIKE '%ジャンプ%'
          AND NOT (rc.distance >= 3000 AND (
                rc.title LIKE '%J' OR rc.title LIKE '%JS' OR rc.title LIKE '%GJ'
          ))
          AND rc.surface != '障害'
        ORDER BY rc.date DESC
    """
    past = pd.read_sql_query(query, conn, params=params)

    if len(past) == 0:
        return _empty_trainer_features()

    features = {}
    features["trainer_win_rate"] = (past["finish_position"] == 1).mean()
    features["trainer_top3_rate"] = (past["finish_position"] <= 3).mean()

    recent_20 = past.head(20)
    features["trainer_recent_top3_20"] = (recent_20["finish_position"] <= 3).mean()

    race_venue = race_info.get("venue", "")
    venue_past = past[past["venue"] == race_venue]
    features["trainer_venue_top3"] = (
        (venue_past["finish_position"] <= 3).mean() if len(venue_past) > 0 else np.nan
    )

    return features


def _empty_trainer_features() -> dict:
    return {
        "trainer_win_rate": np.nan,
        "trainer_top3_rate": np.nan,
        "trainer_recent_top3_20": np.nan,
        "trainer_venue_top3": np.nan,
    }


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

    for col in ["horse_avg_speed_idx_5", "horse_top3_rate_5", "jockey_win_rate", "jockey_top3_rate"]:
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


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """v4: 交互作用特徴量を追加"""
    df = df.copy()

    h_top3 = df.get("horse_top3_rate_5", pd.Series(np.nan, index=df.index))
    j_top3 = df.get("jockey_top3_rate", pd.Series(np.nan, index=df.index))
    df["horse_jockey_synergy"] = h_top3 * j_top3

    h_dist = df.get("horse_dist_top3_rate", pd.Series(np.nan, index=df.index))
    df["horse_dist_jockey"] = h_dist * j_top3

    h_form = df.get("horse_form_trend", pd.Series(0, index=df.index)).fillna(0)
    j_recent = df.get("jockey_recent_top3_20", pd.Series(np.nan, index=df.index))
    df["horse_form_x_jockey_recent"] = h_form * j_recent

    h_speed = df.get("horse_avg_speed_idx_5", pd.Series(np.nan, index=df.index))
    df["horse_speed_x_dist_apt"] = h_speed * h_dist

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
