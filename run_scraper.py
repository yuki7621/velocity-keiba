"""データ収集の実行スクリプト"""

import time
from tqdm import tqdm

from config.settings import SCRAPE_YEAR_START, SCRAPE_YEAR_END, SCRAPE_INTERVAL_SEC
from src.db.schema import create_tables
from src.scraper.race_list import get_all_race_ids
from src.scraper.race_result import scrape_race
from src.scraper.storage import save_race_data


def main():
    # 1. DBテーブルを作成
    print("=== データベース初期化 ===")
    create_tables()

    # 2. レースID一覧を取得
    print(f"\n=== レースID取得 ({SCRAPE_YEAR_START}〜{SCRAPE_YEAR_END}) ===")
    race_ids = get_all_race_ids(SCRAPE_YEAR_START, SCRAPE_YEAR_END)
    print(f"合計 {len(race_ids)} レース")

    # 3. 各レースの結果をスクレイピング
    print("\n=== レース結果スクレイピング ===")
    success = 0
    fail = 0

    for race_id in tqdm(race_ids, desc="スクレイピング中"):
        try:
            data = scrape_race(race_id)
            if data:
                save_race_data(data)
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"\nError {race_id}: {e}")
            fail += 1

        time.sleep(SCRAPE_INTERVAL_SEC)

    print(f"\n完了: 成功={success}, 失敗={fail}")


if __name__ == "__main__":
    main()
