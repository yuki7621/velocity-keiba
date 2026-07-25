"""全レースを再スクレイプして prize / sex / trainer_id を埋める

背景:
  スクレイパーの取りこぼしにより、以下が欠損していた:
    - prize      : 100% NULL   → 賞金系2特徴量が死亡 + クラス厳密判定に必要
    - sex        : 100% NULL   → 性別特徴量が使えない
    - trainer_id : 5.6% のみ   → 調教師系4特徴量がほぼ機能せず
  スクレイパー側は修正済みなので、過去レースを取り直せば全て埋まる。

使い方:
    python run_rescrape_all.py            # 未取得のレースだけ再取得（推奨・中断再開可）
    python run_rescrape_all.py --all      # 全レースを強制的に再取得
    python run_rescrape_all.py --year 2026  # 指定年のみ

特徴:
  - 中断しても再実行すれば続きから（処理済みはスキップ）
  - 1件ごとにコミットするため、途中で止めてもそこまでは保存される
"""

import argparse
import sqlite3
import sys
import time

from config.settings import DB_PATH, SCRAPE_INTERVAL_SEC
from src.scraper.race_result import scrape_race
from src.scraper.storage import save_race_data


def get_target_race_ids(only_missing: bool = True, year: str | None = None) -> list[str]:
    """再取得対象のレースIDを返す"""
    conn = sqlite3.connect(DB_PATH)
    where = []
    params: list = []

    if only_missing:
        # prize / sex / trainer_id のいずれかが未取得のレースを対象にする
        where.append("""
            rc.race_id IN (
                SELECT race_id FROM results
                GROUP BY race_id
                HAVING SUM(CASE WHEN prize IS NOT NULL THEN 1 ELSE 0 END) = 0
                    OR SUM(CASE WHEN sex   IS NOT NULL THEN 1 ELSE 0 END) = 0
                    OR SUM(CASE WHEN trainer_id IS NOT NULL AND trainer_id != '' THEN 1 ELSE 0 END) = 0
            )
        """)
    if year:
        where.append("rc.date LIKE ?")
        params.append(f"{year}%")

    sql = "SELECT rc.race_id FROM races rc"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY rc.date DESC"

    ids = [r[0] for r in conn.execute(sql, params).fetchall()]
    conn.close()
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="全レースを強制再取得")
    ap.add_argument("--year", help="対象年 (例: 2026)")
    ap.add_argument("--limit", type=int, help="最大件数（動作確認用）")
    args = ap.parse_args()

    race_ids = get_target_race_ids(only_missing=not args.all, year=args.year)
    if args.limit:
        race_ids = race_ids[: args.limit]

    total = len(race_ids)
    if total == 0:
        print("再取得が必要なレースはありません。")
        return

    est_min = total * SCRAPE_INTERVAL_SEC / 60
    print("=" * 60)
    print("  全レース再スクレイプ (prize / sex / trainer_id の補完)")
    print("=" * 60)
    print(f"  対象: {total:,} レース")
    print(f"  推定所要時間: 約 {est_min / 60:.1f} 時間 ({est_min:.0f}分)")
    print(f"  間隔: {SCRAPE_INTERVAL_SEC}秒/件")
    print("  ※ 中断しても再実行すれば続きから再開できます")
    print()

    ok = fail = 0
    t0 = time.time()
    for i, race_id in enumerate(race_ids, start=1):
        try:
            data = scrape_race(race_id)
            if data and data.get("results"):
                save_race_data(data)
                ok += 1
            else:
                fail += 1
        except KeyboardInterrupt:
            print("\n中断しました。再実行すれば続きから再開します。")
            sys.exit(0)
        except Exception as e:
            fail += 1
            print(f"  ❌ {race_id}: {e}")

        if i % 50 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            remain = (total - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:,}/{total:,}] 成功{ok:,} 失敗{fail:,} "
                  f"| 経過{elapsed/60:.0f}分 残り約{remain:.0f}分")

        time.sleep(SCRAPE_INTERVAL_SEC)

    print()
    print("=" * 60)
    print(f"  完了: 成功 {ok:,} / 失敗 {fail:,}")
    print("=" * 60)
    print("  次のステップ:")
    print("    1. python run_train_v9.py       # 完全なデータで再学習")
    print("    2. python run_eval_v8_vs_v9.py  # v8 と比較して真価を判定")


if __name__ == "__main__":
    main()
