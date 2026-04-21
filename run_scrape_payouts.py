"""既存レースの払戻データを一括取得するスクリプト

payoutsテーブルに未登録のレースについて、
db.netkeiba.com から複勝・単勝の払戻金を取得して保存する。

使い方:
  python run_scrape_payouts.py           # 全未取得レース
  python run_scrape_payouts.py --limit 100  # 最新100レースだけ
"""

import argparse
import sqlite3
import time

from config.settings import DB_PATH, SCRAPE_INTERVAL_SEC
from src.db.schema import create_tables
from src.scraper.race_result import _scrape_from_db


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="取得レース数の上限 (0=全件)")
    args = parser.parse_args()

    # payoutsテーブルを確実に作成
    create_tables()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # payoutsテーブルに未登録のレースを取得（新しい順）
    cur.execute("""
        SELECT r.race_id
        FROM races r
        WHERE r.race_id NOT IN (SELECT DISTINCT race_id FROM payouts)
        ORDER BY r.date DESC
    """)
    race_ids = [row[0] for row in cur.fetchall()]

    if args.limit > 0:
        race_ids = race_ids[:args.limit]

    total = len(race_ids)
    print(f"払戻データ未取得レース: {total}件")

    success = 0
    skip = 0
    fail = 0

    for i, race_id in enumerate(race_ids, 1):
        try:
            result = _scrape_from_db(race_id)
            if result is None:
                skip += 1
                continue

            payouts = result.get("payouts", [])
            if not payouts:
                skip += 1
                continue

            # DB保存
            for p in payouts:
                cur.execute("""
                    INSERT OR REPLACE INTO payouts
                    (race_id, bet_type, horse_number, payout)
                    VALUES (?, ?, ?, ?)
                """, (p["race_id"], p["bet_type"], p["horse_number"], p["payout"]))

            conn.commit()
            success += 1

            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] 成功={success}, スキップ={skip}, 失敗={fail}")

        except Exception as e:
            fail += 1
            if i % 50 == 0:
                print(f"  [{i}/{total}] エラー: {e}")

        # スクレイピング間隔
        time.sleep(SCRAPE_INTERVAL_SEC)

    conn.close()
    print(f"\n完了: 成功={success}, スキップ={skip}, 失敗={fail}")


if __name__ == "__main__":
    main()
