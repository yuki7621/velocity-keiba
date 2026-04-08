"""レース一覧ページからレースIDを収集する"""

import re
import time
import requests
from bs4 import BeautifulSoup
from config.settings import NETKEIBA_BASE_URL, SCRAPE_INTERVAL_SEC, USER_AGENT

# 1ページあたりの表示件数
_PER_PAGE = 20


def _extract_race_ids_from_page(soup: BeautifulSoup) -> list[str]:
    """ページ内のレースIDを抽出する"""
    race_ids = []
    for a_tag in soup.select("a[href*='/race/']"):
        href = a_tag.get("href", "")
        if "/race/" in href:
            parts = href.split("/race/")
            if len(parts) > 1:
                race_id = parts[1].strip("/")
                if race_id.isdigit() and len(race_id) == 12:
                    race_ids.append(race_id)
    return race_ids


def _get_total_count(soup: BeautifulSoup) -> int:
    """検索結果の総件数を取得する (例: '264件中1~20件目')"""
    pager = soup.select_one(".pager")
    if pager:
        m = re.search(r"(\d+)件中", pager.get_text())
        if m:
            return int(m.group(1))
    return 0


def get_race_ids_by_month(year: int, month: int) -> list[str]:
    """指定年月のレースID一覧を全ページ取得する"""
    base_url = (
        f"{NETKEIBA_BASE_URL}/?pid=race_list&word="
        f"&start_year={year}&start_mon={month}"
        f"&end_year={year}&end_mon={month}"
        "&jyo%5B%5D=01&jyo%5B%5D=02&jyo%5B%5D=03&jyo%5B%5D=04"
        "&jyo%5B%5D=05&jyo%5B%5D=06&jyo%5B%5D=07&jyo%5B%5D=08"
        "&jyo%5B%5D=09&jyo%5B%5D=10"
    )

    headers = {"User-Agent": USER_AGENT}
    all_ids = []

    # 1ページ目
    response = requests.get(base_url, headers=headers, timeout=30)
    response.encoding = "EUC-JP"
    soup = BeautifulSoup(response.text, "lxml")

    all_ids.extend(_extract_race_ids_from_page(soup))
    total = _get_total_count(soup)

    if total == 0:
        return sorted(set(all_ids))

    # 2ページ目以降
    total_pages = (total + _PER_PAGE - 1) // _PER_PAGE
    for page in range(2, total_pages + 1):
        time.sleep(SCRAPE_INTERVAL_SEC)
        page_url = f"{base_url}&page={page}"
        response = requests.get(page_url, headers=headers, timeout=30)
        response.encoding = "EUC-JP"
        soup = BeautifulSoup(response.text, "lxml")
        all_ids.extend(_extract_race_ids_from_page(soup))

    return sorted(set(all_ids))


def get_all_race_ids(year_start: int, year_end: int) -> list[str]:
    """指定期間の全レースIDを取得する"""
    all_ids = []
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            print(f"  {year}年{month}月のレースを取得中...")
            ids = get_race_ids_by_month(year, month)
            all_ids.extend(ids)
            print(f"    → {len(ids)}件")
    return all_ids
