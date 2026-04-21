"""
全馬の血統情報を取得してDBに保存する

使い方:
  python run_scrape_pedigree.py              # 未取得馬のみ取得（推奨）
  python run_scrape_pedigree.py --all        # 既存含め全馬取得（上書き）
  python run_scrape_pedigree.py --limit 100  # 最大100頭のみ

注意:
  - 32,000頭以上あるため、完全実行には約10時間かかります
  - 中断しても --resume で再開可能（デフォルト動作）
  - Ctrl+C で中断しても、それまでの進捗はDBに保存されます
"""

import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import DB_PATH, SCRAPE_INTERVAL_SEC
from src.scraper.horse_pedigree import scrape_pedigree


def get_target_horses(conn, fetch_all: bool, limit: int | None) -> list[str]:
    """取得対象の馬IDリストを返す"""
    if fetch_all:
        query = "SELECT horse_id FROM horses ORDER BY horse_id"
    else:
        query = """
            SELECT horse_id FROM horses
            WHERE sire IS NULL AND dam_sire IS NULL
            ORDER BY horse_id
        """
    if limit:
        query += f" LIMIT {limit}"

    cur = conn.execute(query)
    return [r[0] for r in cur.fetchall()]


def update_pedigree(conn, horse_id: str, sire: str | None, dam_sire: str | None):
    """血統情報をDBに保存"""
    conn.execute(
        "UPDATE horses SET sire = ?, dam_sire = ? WHERE horse_id = ?",
        (sire, dam_sire, horse_id),
    )


def main():
    fetch_all = "--all" in sys.argv
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    conn = sqlite3.connect(DB_PATH)
    horse_ids = get_target_horses(conn, fetch_all, limit)

    total = len(horse_ids)
    if total == 0:
        print("対象の馬がいません。")
        conn.close()
        return

    # 予想所要時間
    eta_sec = total * SCRAPE_INTERVAL_SEC
    eta = timedelta(seconds=int(eta_sec))
    start_time = datetime.now()
    finish_estimate = start_time + eta

    print("=" * 60)
    print(f"血統情報スクレイピング開始")
    print(f"  対象: {total:,} 頭")
    print(f"  モード: {'全馬上書き' if fetch_all else '未取得のみ'}")
    print(f"  間隔: {SCRAPE_INTERVAL_SEC}秒")
    print(f"  推定所要時間: {eta}")
    print(f"  終了予定: {finish_estimate.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    success = 0
    fail = 0

    try:
        for i, hid in enumerate(horse_ids, 1):
            try:
                result = scrape_pedigree(hid)
            except Exception as e:
                result = None

            if result:
                update_pedigree(
                    conn, hid,
                    result.get("sire"),
                    result.get("dam_sire"),
                )
                success += 1
            else:
                # 取得失敗した馬には明示的に空文字を入れて次回スキップ対象に
                update_pedigree(conn, hid, "", "")
                fail += 1

            # 50頭ごとにコミット + 進捗表示
            if i % 50 == 0:
                conn.commit()
                elapsed = datetime.now() - start_time
                remaining = total - i
                avg_per = elapsed / i
                eta_remain = avg_per * remaining
                print(f"  [{i:,}/{total:,}] 成功={success:,} 失敗={fail:,} "
                      f"経過={str(elapsed).split('.')[0]} "
                      f"残り={str(eta_remain).split('.')[0]}")

            time.sleep(SCRAPE_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n\n中断されました。進捗を保存します...")

    conn.commit()
    conn.close()

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print("完了")
    print(f"  成功: {success:,} 頭")
    print(f"  失敗: {fail:,} 頭")
    print(f"  所要時間: {str(elapsed).split('.')[0]}")


if __name__ == "__main__":
    main()
