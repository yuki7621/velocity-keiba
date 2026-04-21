"""
既存DBから障害レースを削除する

判定基準:
  1. distance が 100m 単位でない（1150m札幌新馬戦を除く）
  2. title に「障害」を含む
  3. title に「ジャンプ」を含む
  4. distance >= 3000m かつ title 末尾が J / JS / GJ（スプリングJ、中山GJ等）

使い方:
  python scripts/delete_jump_races.py --dry-run    # 削除対象を表示するだけ
  python scripts/delete_jump_races.py              # 実際に削除
"""

import sqlite3
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DB_PATH


# 障害レース判定 WHERE 句（race_id を返すサブクエリに使用）
JUMP_FILTER = """
    (distance % 100 != 0 AND distance != 1150)
    OR (title LIKE '%障害%')
    OR (title LIKE '%ジャンプ%')
    OR (distance >= 3000 AND (title LIKE '%J' OR title LIKE '%JS' OR title LIKE '%GJ'))
"""


def get_jump_race_ids(conn) -> list[str]:
    """障害レースのrace_idを全て取得"""
    cur = conn.execute(f"SELECT race_id FROM races WHERE {JUMP_FILTER}")
    return [r[0] for r in cur.fetchall()]


def count_affected_rows(conn) -> dict:
    """各テーブルで削除される行数を返す"""
    counts = {}
    for table in ["races", "results", "payouts"]:
        if table == "races":
            q = f"SELECT COUNT(*) FROM races WHERE {JUMP_FILTER}"
        else:
            q = f"""
                SELECT COUNT(*) FROM {table} t
                JOIN races rc ON t.race_id = rc.race_id
                WHERE {JUMP_FILTER.replace('distance', 'rc.distance').replace('title', 'rc.title')}
            """
        counts[table] = conn.execute(q).fetchone()[0]
    return counts


def delete_jump_races(conn) -> dict:
    """障害レースを関連テーブル含めて削除"""
    jump_ids = get_jump_race_ids(conn)
    if not jump_ids:
        return {"races": 0, "results": 0, "payouts": 0}

    # プレースホルダを使った削除（大量の場合はチャンク分割）
    deleted = {"races": 0, "results": 0, "payouts": 0}
    CHUNK = 500

    for i in range(0, len(jump_ids), CHUNK):
        chunk = jump_ids[i:i + CHUNK]
        placeholders = ",".join(["?"] * len(chunk))

        for table in ["payouts", "results", "races"]:  # 外部キー順（子→親）
            cur = conn.execute(
                f"DELETE FROM {table} WHERE race_id IN ({placeholders})",
                chunk,
            )
            deleted[table] += cur.rowcount

    conn.commit()
    return deleted


def main():
    dry_run = "--dry-run" in sys.argv

    print(f"DB: {DB_PATH}")
    print(f"モード: {'DRY-RUN（削除しない）' if dry_run else '実行'}")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # 事前統計
    before_races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    before_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    before_payouts = conn.execute("SELECT COUNT(*) FROM payouts").fetchone()[0]

    counts = count_affected_rows(conn)
    print(f"削除対象:")
    print(f"  races:    {counts['races']:,} レース")
    print(f"  results:  {counts['results']:,} 行")
    print(f"  payouts:  {counts['payouts']:,} 行")

    # サンプル表示
    print("\n=== 削除対象サンプル（最大20件）===")
    cur = conn.execute(f"""
        SELECT race_id, date, venue, surface, distance, title
        FROM races WHERE {JUMP_FILTER}
        ORDER BY date DESC LIMIT 20
    """)
    for r in cur.fetchall():
        print(f"  {r[0]} {r[1]} {r[2]} {r[3]}{r[4]}m {r[5]}")

    if dry_run:
        print("\n※ DRY-RUN モードのため削除は実行されませんでした。")
        print("  実行するには --dry-run を外して再実行してください。")
        conn.close()
        return

    # 実行確認
    print()
    ans = input("本当に削除しますか？ (yes/no): ").strip().lower()
    if ans != "yes":
        print("キャンセルしました。")
        conn.close()
        return

    # 削除実行
    print("\n削除中...")
    deleted = delete_jump_races(conn)

    # VACUUM で容量回収
    print("VACUUM 実行中...")
    conn.execute("VACUUM")

    conn.close()

    # 事後確認
    conn = sqlite3.connect(DB_PATH)
    after_races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    after_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    after_payouts = conn.execute("SELECT COUNT(*) FROM payouts").fetchone()[0]
    conn.close()

    print("\n" + "=" * 60)
    print("削除完了")
    print(f"  races:    {before_races:,} → {after_races:,} (-{deleted['races']:,})")
    print(f"  results:  {before_results:,} → {after_results:,} (-{deleted['results']:,})")
    print(f"  payouts:  {before_payouts:,} → {after_payouts:,} (-{deleted['payouts']:,})")


if __name__ == "__main__":
    main()
