"""スクレイピング結果をDBに保存する"""

import sqlite3
from config.settings import DB_PATH


def is_jump_race(race_info: dict) -> bool:
    """障害レースかどうか判定する

    判定基準:
      1. surface が "障害"
      2. distance が 100m 単位でない（1150m札幌新馬戦を除く）
      3. title に「障害」を含む
      4. title に「ジャンプ」を含む
      5. distance >= 3000m かつ title 末尾が J / JS / GJ
    """
    surface = race_info.get("surface") or ""
    if "障害" in surface:
        return True

    distance = race_info.get("distance") or 0
    try:
        distance = int(distance)
    except (ValueError, TypeError):
        distance = 0

    if distance > 0 and distance % 100 != 0 and distance != 1150:
        return True

    title = race_info.get("title") or ""
    if "障害" in title or "ジャンプ" in title:
        return True

    if distance >= 3000:
        t = title.rstrip()
        if t.endswith("J") or t.endswith("JS") or t.endswith("GJ"):
            return True

    return False


def save_race_data(data: dict, db_path=DB_PATH):
    """1レース分のデータをDBに保存する。障害レースはスキップ。"""
    race = data["race_info"]

    # 障害レースは保存しない
    if is_jump_race(race):
        return {"skipped": True, "reason": "jump_race", "race_id": race.get("race_id")}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

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

        # 調教師を保存
        if row.get("trainer_id"):
            cur.execute("""
                INSERT OR IGNORE INTO trainers (trainer_id, name)
                VALUES (?, ?)
            """, (row["trainer_id"], row.get("trainer_name", "")))

        # 結果を保存
        cur.execute("""
            INSERT OR REPLACE INTO results
            (race_id, horse_id, jockey_id, trainer_id, post_number, gate_number,
             odds, popularity, weight_carried, horse_weight, weight_change,
             finish_position, finish_time, finish_time_sec, last_3f, passing_order)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.get("race_id"),
            row.get("horse_id"),
            row.get("jockey_id"),
            row.get("trainer_id"),
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

    # 払戻情報を保存
    for payout in data.get("payouts", []):
        cur.execute("""
            INSERT OR REPLACE INTO payouts
            (race_id, bet_type, horse_number, payout)
            VALUES (?, ?, ?, ?)
        """, (
            payout.get("race_id"),
            payout.get("bet_type"),
            payout.get("horse_number"),
            payout.get("payout"),
        ))

    conn.commit()
    conn.close()
    return {"skipped": False, "race_id": race.get("race_id")}
