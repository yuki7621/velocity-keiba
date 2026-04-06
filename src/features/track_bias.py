"""
馬場傾向分析モジュール

土曜日のレース結果から馬場バイアス（傾向）を数値化し、
日曜日の予測に反映させる。

分析する傾向:
  1. 内外バイアス  — 内枠/外枠どちらが有利か
  2. 脚質バイアス  — 逃げ先行有利か、差し追込有利か
  3. タイムバイアス — 時計が速いか遅いか（馬場の軽さ）
  4. 上がりバイアス — 末脚勝負になりやすいか
"""

import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.settings import DB_PATH


def get_race_day_results(
    target_date: str | date,
    venue: str,
    db_path=DB_PATH,
) -> pd.DataFrame:
    """指定日・指定競馬場の全レース結果を取得する"""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            r.race_id, r.horse_id, r.post_number, r.gate_number,
            r.finish_position, r.finish_time_sec, r.last_3f,
            r.passing_order, r.odds, r.popularity,
            rc.surface, rc.distance, rc.condition, rc.head_count
        FROM results r
        JOIN races rc ON r.race_id = rc.race_id
        WHERE rc.date = ?
          AND rc.venue = ?
          AND r.finish_position > 0
        ORDER BY r.race_id, r.finish_position
    """
    df = pd.read_sql_query(query, conn, params=[str(target_date), venue])
    conn.close()
    return df


def get_previous_day(
    target_date: str | date,
    venue: str,
    db_path=DB_PATH,
) -> str | None:
    """
    指定日の直前の開催日を取得する（通常は前日の土曜日）。
    同じ競馬場で直近に開催された日を探す。
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT DISTINCT date FROM races
        WHERE venue = ?
          AND date < ?
        ORDER BY date DESC
        LIMIT 1
    """
    result = conn.execute(query, [venue, str(target_date)]).fetchone()
    conn.close()
    return result[0] if result else None


def analyze_track_bias(df: pd.DataFrame) -> dict:
    """
    1日分のレース結果から馬場傾向を分析する。

    Returns:
        {
            "gate_bias":     float,  # 正=内枠有利, 負=外枠有利
            "pace_bias":     float,  # 正=先行有利, 負=差し有利
            "time_bias":     float,  # 正=高速馬場, 負=タフな馬場
            "last3f_bias":   float,  # 正=上がり重要, 負=上がり関係薄い
            "inner_top3_rate": float, # 内枠(1-4)の複勝率
            "outer_top3_rate": float, # 外枠(5-8)の複勝率
            "front_top3_rate": float, # 先行馬の複勝率
            "closer_top3_rate": float, # 差し追込馬の複勝率
            "n_races":       int,    # 分析レース数
        }
    """
    if len(df) == 0:
        return _empty_bias()

    # ─── 1. 内外バイアス ───
    # 内枠(枠番1-4) vs 外枠(枠番5-8) の複勝率を比較
    df["is_top3"] = (df["finish_position"] <= 3).astype(int)
    inner = df[df["gate_number"] <= 4]
    outer = df[df["gate_number"] >= 5]

    inner_top3 = inner["is_top3"].mean() if len(inner) > 0 else 0.0
    outer_top3 = outer["is_top3"].mean() if len(outer) > 0 else 0.0

    # gate_bias: 正=内有利、負=外有利 (差を標準化)
    gate_bias = inner_top3 - outer_top3

    # ─── 2. 脚質バイアス（逃げ先行 vs 差し追込）───
    # 1コーナー通過3番手以内 = 先行馬
    def parse_first_pass(s):
        if pd.isna(s) or s == "":
            return np.nan
        parts = str(s).split("-")
        try:
            return int(parts[0])
        except (ValueError, IndexError):
            return np.nan

    df["first_pass"] = df["passing_order"].apply(parse_first_pass)

    # 先行馬 = 1コーナー通過が頭数の1/3以内
    df["is_front"] = df["first_pass"] <= (df["head_count"] / 3)

    front = df[df["is_front"] == True]
    closer = df[df["is_front"] == False]

    front_top3 = front["is_top3"].mean() if len(front) > 0 else 0.0
    closer_top3 = closer["is_top3"].mean() if len(closer) > 0 else 0.0

    pace_bias = front_top3 - closer_top3

    # ─── 3. タイムバイアス（馬場の速さ）───
    # 各レースの勝ちタイムが同距離の平均より速いか遅いか
    # → 同距離の基準タイムとの差を平均
    winners = df[df["finish_position"] == 1].copy()
    if len(winners) > 0 and "finish_time_sec" in winners.columns:
        # 距離あたりのスピード（秒/m）で正規化
        winners["sec_per_m"] = winners["finish_time_sec"] / winners["distance"]
        time_bias = -winners["sec_per_m"].mean()  # 速い=正
        # 正規化（おおよそ-1~1の範囲に）
        time_bias = (time_bias + 0.068) * 100  # 芝1600mで約96秒→0.06秒/m
    else:
        time_bias = 0.0

    # ─── 4. 上がり3Fバイアス ───
    # 上がり3Fが速い馬ほど好走しているか
    # → 3着以内の馬の上がり3F平均 vs 4着以下の平均
    top3 = df[df["finish_position"] <= 3]
    bottom = df[df["finish_position"] > 3]

    top3_last3f = top3["last_3f"].mean() if len(top3) > 0 else 0.0
    bottom_last3f = bottom["last_3f"].mean() if len(bottom) > 0 else 0.0

    # 差が大きい = 上がりが重要（末脚勝負になりやすい馬場）
    last3f_bias = bottom_last3f - top3_last3f  # 正=上がり重要

    n_races = df["race_id"].nunique()

    return {
        "gate_bias": round(gate_bias, 4),
        "pace_bias": round(pace_bias, 4),
        "time_bias": round(time_bias, 4),
        "last3f_bias": round(last3f_bias, 4),
        "inner_top3_rate": round(inner_top3, 4),
        "outer_top3_rate": round(outer_top3, 4),
        "front_top3_rate": round(front_top3, 4),
        "closer_top3_rate": round(closer_top3, 4),
        "n_races": n_races,
    }


def get_track_bias_for_date(
    target_date: str | date,
    venue: str,
    db_path=DB_PATH,
) -> dict:
    """
    指定日の予測に使う馬場傾向を取得する。
    直前の開催日（通常は土曜）のデータから分析する。
    """
    prev_day = get_previous_day(target_date, venue, db_path)
    if prev_day is None:
        print(f"  ※ {venue}の前日データが見つかりません。バイアス=0で計算します。")
        return _empty_bias()

    df = get_race_day_results(prev_day, venue, db_path)
    if len(df) == 0:
        print(f"  ※ {venue} {prev_day}のレース結果が空です。")
        return _empty_bias()

    bias = analyze_track_bias(df)
    print(f"  {venue} 前日({prev_day}): {bias['n_races']}R分析 "
          f"| 内外={bias['gate_bias']:+.3f} "
          f"| 脚質={bias['pace_bias']:+.3f} "
          f"| 時計={bias['time_bias']:+.3f} "
          f"| 上がり={bias['last3f_bias']:+.3f}")
    return bias


def _empty_bias() -> dict:
    """バイアスなし（デフォルト値）"""
    return {
        "gate_bias": 0.0,
        "pace_bias": 0.0,
        "time_bias": 0.0,
        "last3f_bias": 0.0,
        "inner_top3_rate": 0.0,
        "outer_top3_rate": 0.0,
        "front_top3_rate": 0.0,
        "closer_top3_rate": 0.0,
        "n_races": 0,
    }


# ============================================================
#  特徴量としてDataFrameに追加する関数
# ============================================================

def add_track_bias_features(df: pd.DataFrame, db_path=DB_PATH) -> pd.DataFrame:
    """
    各レースの「前日の馬場傾向」を特徴量として追加する。
    学習時: 全レースに対して自動計算
    予測時: 指定日の前日データを使う
    """
    df = df.copy()
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)

    # date + venue の組み合わせごとにバイアスを計算（キャッシュ）
    bias_cache = {}
    bias_columns = [
        "bias_gate", "bias_pace", "bias_time", "bias_last3f",
        "bias_inner_top3", "bias_outer_top3",
        "bias_front_top3", "bias_closer_top3",
    ]
    for col in bias_columns:
        df[col] = np.nan

    unique_combos = df[["date", "venue"]].drop_duplicates()
    total = len(unique_combos)

    for idx, (_, row) in enumerate(unique_combos.iterrows()):
        dt = row["date"]
        venue = row["venue"]
        cache_key = (str(dt.date() if hasattr(dt, 'date') else dt), venue)

        if cache_key not in bias_cache:
            bias = get_track_bias_for_date(cache_key[0], venue, db_path)
            bias_cache[cache_key] = bias

        bias = bias_cache[cache_key]
        mask = (df["date"] == dt) & (df["venue"] == venue)

        df.loc[mask, "bias_gate"] = bias["gate_bias"]
        df.loc[mask, "bias_pace"] = bias["pace_bias"]
        df.loc[mask, "bias_time"] = bias["time_bias"]
        df.loc[mask, "bias_last3f"] = bias["last3f_bias"]
        df.loc[mask, "bias_inner_top3"] = bias["inner_top3_rate"]
        df.loc[mask, "bias_outer_top3"] = bias["outer_top3_rate"]
        df.loc[mask, "bias_front_top3"] = bias["front_top3_rate"]
        df.loc[mask, "bias_closer_top3"] = bias["closer_top3_rate"]

        if (idx + 1) % 200 == 0:
            print(f"    馬場傾向: {idx + 1}/{total} 日程処理済み")

    # ─── バイアスと馬の特性の交互作用 ───
    # 内枠の馬 × 内枠有利バイアス → 大きいほど有利
    df["bias_x_inner"] = df["is_inner_gate"] * df["bias_gate"]

    # 先行馬 × 先行有利バイアス
    if "horse_avg_early_pos" in df.columns:
        # avg_early_posが小さい=先行馬 → 反転して掛ける
        df["bias_x_pace"] = (1.0 / df["horse_avg_early_pos"].clip(lower=1)) * df["bias_pace"]
    else:
        df["bias_x_pace"] = 0.0

    # 末脚型の馬 × 上がり重要バイアス
    if "horse_avg_last_3f_5" in df.columns:
        # last3fが速い(小さい)馬 × 上がり重要バイアスが大きい → 有利
        race_mean_3f = df.groupby("race_id")["horse_avg_last_3f_5"].transform("mean")
        df["bias_x_last3f"] = (race_mean_3f - df["horse_avg_last_3f_5"]) * df["bias_last3f"]
    else:
        df["bias_x_last3f"] = 0.0

    return df
