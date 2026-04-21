"""
既存DBの全レースについて払戻情報だけ再取得する。

拡張後の scraper で全券種（単勝/複勝/枠連/馬連/ワイド/馬単/三連複/三連単）を
payouts テーブルに追加する。

- レース結果 (results) は再取得しない（結果は変わらないため）
- レース情報 (races) も再取得しない
- 払戻だけ `INSERT OR REPLACE` で上書き

ETA: 約12,000レース × 2秒 ≒ 6〜7時間
途中中断しても冪等（再実行すれば続きから）。
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from config.settings import DB_PATH, SCRAPE_INTERVAL_SEC
from src.scraper.race_result import scrape_race


def get_target_race_ids(conn, skip_completed: bool, since_date: str | None) -> list[str]:
    """再取得対象のレースID一覧を返す。"""
    cur = conn.cursor()
    query = "SELECT race_id FROM races"
    conds = []
    params = []
    if since_date:
        conds.append("date >= ?")
        params.append(since_date)
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY date"
    cur.execute(query, params)
    all_ids = [r[0] for r in cur.fetchall()]

    if not skip_completed:
        return all_ids

    # 既にワイド/馬連/3連複が揃っているレースはスキップ
    cur.execute("""
        SELECT race_id FROM payouts
        WHERE bet_type IN ('wide', 'umaren', 'sanrenpuku')
        GROUP BY race_id
        HAVING COUNT(DISTINCT bet_type) >= 3
    """)
    done_ids = {r[0] for r in cur.fetchall()}
    return [rid for rid in all_ids if rid not in done_ids]


def save_payouts_only(data: dict, conn: sqlite3.Connection):
    """払戻だけDBに保存する（results等は触らない）"""
    cur = conn.cursor()
    for p in data.get("payouts", []):
        hns = p.get("horse_numbers")
        if hns is None and p.get("horse_number") is not None:
            hns = str(p["horse_number"])
        cur.execute("""
            INSERT OR REPLACE INTO payouts
            (race_id, bet_type, horse_number, horse_numbers, payout)
            VALUES (?, ?, ?, ?, ?)
        """, (
            p.get("race_id"),
            p.get("bet_type"),
            p.get("horse_number"),
            hns,
            p.get("payout"),
        ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", type=str, default=None,
                        help="この日付以降のレースだけ再取得 (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true",
                        help="完了済みレースも含めて全部再取得")
    parser.add_argument("--limit", type=int, default=None,
                        help="最大N件で止める（動作確認用）")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    race_ids = get_target_race_ids(conn, skip_completed=not args.all, since_date=args.since)

    if args.limit:
        race_ids = race_ids[: args.limit]

    total = len(race_ids)
    print(f"対象レース: {total} 件")
    if total == 0:
        print("全て完了済みです。")
        return

    est_sec = total * SCRAPE_INTERVAL_SEC
    print(f"想定所要時間: {est_sec / 60:.1f} 分 ({est_sec / 3600:.1f} 時間)")
    print()

    start = time.time()
    success = 0
    fail = 0
    for i, race_id in enumerate(race_ids, 1):
        try:
            data = scrape_race(race_id)
            if data and data.get("payouts"):
                save_payouts_only(data, conn)
                success += 1
            else:
                fail += 1
        except KeyboardInterrupt:
            print("\n中断しました。")
            break
        except Exception as e:
            fail += 1
            print(f"  ⚠️ {race_id}: {e}", file=sys.stderr)

        if i % 50 == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} 成功:{success} 失敗:{fail} "
                  f"経過:{elapsed/60:.1f}分 残り:{eta/60:.1f}分")

        time.sleep(SCRAPE_INTERVAL_SEC)

    conn.close()
    print(f"\n完了: 成功={success}, 失敗={fail}, 所要時間={(time.time()-start)/60:.1f}分")


if __name__ == "__main__":
    main()
