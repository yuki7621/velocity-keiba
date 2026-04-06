"""スクレイピング結果をDBに保存する"""

import sqlite3
from config.settings import DB_PATH


def save_race_data(data: dict, db_path=DB_PATH):
    """1レース分のデータをDBに保存する"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    race = data["race_info"]

    # レース情報を保存
    cur.execute("""
        INSERT OR REPLACE INTO races
        (race_id, date, venue, title, surface, distance, weather, condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        race.get("race_id"),
        race.get("date"),
        race.get("venue"),
        race.get("title"),
        race.get("surface"),
        race.get("distance"),
        race.get("weather"),
        race.get("condition"),
    ))

    # 出走結果を保存
    for row in data["results"]:
        # 馬を保存
        if row.get("horse_id"):
            cur.execute("""
                INSERT OR IGNORE INTO horses (horse_id, name)
                VALUES (?, ?)
            """, (row["horse_id"], row.get("horse_name", "")))

        # 騎手を保存
        if row.get("jockey_id"):
            cur.execute("""
                INSERT OR IGNORE INTO jockeys (jockey_id, name)
                VALUES (?, ?)
            """, (row["jockey_id"], row.get("jockey_name", "")))

        # 結果を保存
        cur.execute("""
            INSERT OR REPLACE INTO results
            (race_id, horse_id, jockey_id, post_number, gate_number,
             odds, popularity, weight_carried, horse_weight, weight_change,
             finish_position, finish_time, finish_time_sec, last_3f, passing_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("race_id"),
            row.get("horse_id"),
            row.get("jockey_id"),
            row.get("post_number"),
            row.get("gate_number"),
            row.get("odds"),
            row.get("popularity"),
            row.get("weight_carried"),
            row.get("horse_weight"),
            row.get("weight_change"),
            row.get("finish_position"),
            row.get("finish_time"),
            row.get("finish_time_sec"),
            row.get("last_3f"),
            row.get("passing_order"),
        ))

    conn.commit()
    conn.close()
