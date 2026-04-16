"""
出馬表（レース前のエントリー情報）をスクレイピングする

race.netkeiba.com の出馬表ページから、レース開始前に
出走馬・騎手・枠番・馬番・斤量・馬体重などを取得する。
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from config.settings import USER_AGENT, SCRAPE_INTERVAL_SEC

# 出馬表のベースURL（db.netkeiba.com ではなく race.netkeiba.com）
SHUTUBA_BASE_URL = "https://race.netkeiba.com"


def scrape_race_card(race_id: str) -> dict | None:
    """
    レースIDから出馬表（レース前情報）を取得する。

    Returns:
        {
            "race_info": {race_id, date, venue, surface, distance, condition, ...},
            "entries": [
                {horse_id, horse_name, jockey_id, jockey_name,
                 post_number, gate_number, weight_carried, horse_weight, weight_change, ...},
                ...
            ]
        }
    """
    url = f"{SHUTUBA_BASE_URL}/race/shutuba.html?race_id={race_id}"
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = "EUC-JP"
    if response.status_code != 200:
        print(f"Error: {race_id} status={response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    race_info = _parse_race_info_card(soup, race_id)
    if race_info is None:
        return None

    entries = _parse_entries(soup, race_id)
    race_info["head_count"] = len(entries)

    return {"race_info": race_info, "entries": entries}


def _parse_race_info_card(soup: BeautifulSoup, race_id: str) -> dict | None:
    """出馬表ページからレース情報をパースする"""
    info = {"race_id": race_id}

    # レース名
    race_name_tag = soup.select_one(".RaceName")
    if race_name_tag:
        info["title"] = race_name_tag.get_text(strip=True)
    else:
        info["title"] = ""

    # レースデータ（距離、馬場、天候等）
    race_data = soup.select_one(".RaceData01")
    if race_data:
        text = race_data.get_text(strip=True)
        # 芝/ダート（出馬表では「ダ1800m」のように略記されることがある）
        if "芝" in text:
            info["surface"] = "芝"
        elif "ダート" in text or "ダ" in text:
            info["surface"] = "ダート"
        elif "障" in text:
            info["surface"] = "障害"
        # 距離
        m = re.search(r"(\d{3,5})m", text)
        if m:
            info["distance"] = int(m.group(1))
        # 天候
        m = re.search(r"天候:(\S+)", text)
        if m:
            info["weather"] = m.group(1)
        # 馬場状態
        m = re.search(r"馬場:(\S+)", text)
        if m:
            info["condition"] = m.group(1)

    # 日付・競馬場をrace_idから推定
    from config.settings import VENUE_CODES
    venue_code = race_id[4:6]
    if venue_code in VENUE_CODES:
        info["venue"] = VENUE_CODES[venue_code]
        info["venue_code"] = venue_code

    # 日付はRaceData02から取得を試みる
    race_data2 = soup.select_one(".RaceData02")
    if race_data2:
        date_text = race_data2.get_text(strip=True)
        m = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", date_text)
        if m:
            info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # 日付がまだ無ければtitleタグから
    if "date" not in info:
        title_tag = soup.select_one("title")
        if title_tag:
            title_text = title_tag.get_text()
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_text)
            if m:
                info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    return info


def _parse_entries(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """出馬表テーブルから出走馬情報をパースする"""
    entries = []
    table = soup.select_one("table")
    if table is None:
        return entries

    for tr in table.select("tr"):
        tds = tr.select("td")
        if len(tds) < 9:
            continue

        entry = {"race_id": race_id}

        # [0] 枠番, [1] 馬番
        entry["gate_number"] = _to_int(tds[0].get_text(strip=True))
        entry["post_number"] = _to_int(tds[1].get_text(strip=True))

        if entry["gate_number"] is None or entry["post_number"] is None:
            continue

        # [3] 馬名・馬ID
        horse_link = tds[3].select_one("a")
        if horse_link:
            entry["horse_name"] = horse_link.get_text(strip=True)
            href = horse_link.get("href", "")
            m = re.search(r"/horse/(\w+)", href)
            entry["horse_id"] = m.group(1) if m else ""
        else:
            entry["horse_name"] = tds[3].get_text(strip=True)
            entry["horse_id"] = ""

        # [4] 性齢
        entry["sex_age"] = tds[4].get_text(strip=True)

        # [5] 斤量
        entry["weight_carried"] = _to_float(tds[5].get_text(strip=True))

        # [6] 騎手・騎手ID
        jockey_link = tds[6].select_one("a")
        if jockey_link:
            entry["jockey_name"] = jockey_link.get_text(strip=True)
            href = jockey_link.get("href", "")
            m = re.search(r"/jockey/(?:result/recent/)?(\d+)", href)
            entry["jockey_id"] = m.group(1) if m else ""
        else:
            entry["jockey_name"] = tds[6].get_text(strip=True)
            entry["jockey_id"] = ""

        # [8] 馬体重 (例: "504(+4)")
        if len(tds) > 8:
            weight_text = tds[8].get_text(strip=True)
            m = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
            if m:
                entry["horse_weight"] = int(m.group(1))
                entry["weight_change"] = int(m.group(2))
            else:
                entry["horse_weight"] = None
                entry["weight_change"] = None

        entries.append(entry)

    return entries


def get_upcoming_race_ids(date_str: str = None) -> list[str]:
    """
    指定日（またはトップページ）の開催レースID一覧を取得する。
    date_str: 'YYYYMMDD' 形式
    """
    if date_str:
        url = f"{SHUTUBA_BASE_URL}/top/race_list_sub.html?kaisai_date={date_str}"
    else:
        url = f"{SHUTUBA_BASE_URL}/top/race_list.html"

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = "EUC-JP"

    soup = BeautifulSoup(response.text, "lxml")
    race_ids = []

    for a_tag in soup.select("a[href*='race_id=']"):
        href = a_tag.get("href", "")
        m = re.search(r"race_id=(\d{12})", href)
        if m:
            race_ids.append(m.group(1))

    return sorted(set(race_ids))


def _to_int(s: str) -> int | None:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
