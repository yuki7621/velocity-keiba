"""個別レースページから結果データをスクレイピングする"""

import re
import requests
from bs4 import BeautifulSoup
from config.settings import NETKEIBA_BASE_URL, USER_AGENT


def scrape_race(race_id: str) -> dict | None:
    """
    レースIDから結果を取得する。

    Returns:
        {
            "race_info": {...},      # レース基本情報
            "results": [{...}, ...]  # 各馬の成績
        }
    """
    url = f"{NETKEIBA_BASE_URL}/race/{race_id}/"
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers, timeout=30)
    response.encoding = "EUC-JP"

    if response.status_code != 200:
        print(f"Error: {race_id} status={response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    race_info = _parse_race_info(soup, race_id)
    if race_info is None:
        return None

    results = _parse_results_table(soup, race_id)

    return {"race_info": race_info, "results": results}


def _parse_race_info(soup: BeautifulSoup, race_id: str) -> dict | None:
    """レース情報をパースする"""
    info = {"race_id": race_id}

    # レース名・日付: titleタグから取得 (例: "東京優駿｜2024年5月26日 | 競馬データベース")
    title_tag = soup.select_one("title")
    if title_tag is None:
        return None
    title_text = title_tag.get_text(strip=True)
    # レース名 = "｜" の前の部分
    race_name = title_text.split("｜")[0].split("|")[0].strip()
    info["title"] = race_name
    # 日付を抽出
    info.update(_parse_date_venue(title_text))

    # レース名をracedata ddからも取得（グレード情報付き）
    dd_tag = soup.select_one(".racedata dd")
    if dd_tag:
        dd_text = dd_tag.get_text(strip=True)
        # グレード抽出 (例: "(GI)", "(GII)", "(GIII)")
        m = re.search(r"\((G[I]{1,3}|G[123]|OP|L)\)", dd_text)
        if m:
            info["grade"] = m.group(1)

    # コース情報 (例: "芝左2400m / 天候 : 晴 / 芝 : 良")
    detail_tag = soup.select_one(".racedata dd p span")
    if detail_tag:
        detail_text = detail_tag.get_text(strip=True)
        info.update(_parse_course_detail(detail_text))

    # 競馬場をrace_idから推定
    venue_code = race_id[4:6]
    from config.settings import VENUE_CODES
    if venue_code in VENUE_CODES:
        info["venue"] = VENUE_CODES[venue_code]
        info["venue_code"] = venue_code

    return info


def _parse_course_detail(text: str) -> dict:
    """コース詳細テキストをパースする"""
    result = {}

    # 芝/ダート
    if "芝" in text:
        result["surface"] = "芝"
    elif "ダート" in text:
        result["surface"] = "ダート"
    elif "障" in text:
        result["surface"] = "障害"

    # 距離
    m = re.search(r"(\d{3,5})m", text)
    if m:
        result["distance"] = int(m.group(1))

    # 天候
    m = re.search(r"天候\s*:\s*(\S+)", text)
    if m:
        result["weather"] = m.group(1)

    # 馬場状態
    m = re.search(r"(?:芝|ダート)\s*:\s*(\S+)", text)
    if m:
        result["condition"] = m.group(1)

    return result


def _parse_date_venue(text: str) -> dict:
    """日付・競馬場テキストをパースする"""
    result = {}

    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if m:
        result["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    for venue in ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]:
        if venue in text:
            result["venue"] = venue
            break

    return result


def _parse_results_table(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """成績テーブルをパースする"""
    rows = []
    table = soup.select_one(".race_table_01")
    if table is None:
        return rows

    for tr in table.select("tr")[1:]:  # ヘッダー行をスキップ
        tds = tr.select("td")
        if len(tds) < 13:
            continue

        row = {"race_id": race_id}

        # 着順
        finish_text = tds[0].get_text(strip=True)
        row["finish_position"] = int(finish_text) if finish_text.isdigit() else 0

        # 枠番・馬番
        row["gate_number"] = _to_int(tds[1].get_text(strip=True))
        row["post_number"] = _to_int(tds[2].get_text(strip=True))

        # 馬名・馬ID
        horse_link = tds[3].select_one("a")
        if horse_link:
            row["horse_name"] = horse_link.get_text(strip=True)
            href = horse_link.get("href", "")
            m = re.search(r"/horse/(\w+)", href)
            row["horse_id"] = m.group(1) if m else ""
        else:
            row["horse_name"] = tds[3].get_text(strip=True)
            row["horse_id"] = ""

        # 性齢・斤量
        row["sex_age"] = tds[4].get_text(strip=True)
        row["weight_carried"] = _to_float(tds[5].get_text(strip=True))

        # 騎手・騎手ID
        jockey_link = tds[6].select_one("a")
        if jockey_link:
            row["jockey_name"] = jockey_link.get_text(strip=True)
            href = jockey_link.get("href", "")
            m = re.search(r"/jockey/(\w+)", href)
            row["jockey_id"] = m.group(1) if m else ""
        else:
            row["jockey_name"] = tds[6].get_text(strip=True)
            row["jockey_id"] = ""

        # タイム
        row["finish_time"] = tds[7].get_text(strip=True)
        row["finish_time_sec"] = _time_to_sec(row["finish_time"])

        # 通過順 [14], 上がり3F [15]
        row["passing_order"] = tds[14].get_text(strip=True) if len(tds) > 14 else ""
        row["last_3f"] = _to_float(tds[15].get_text(strip=True)) if len(tds) > 15 else None

        # 単勝オッズ [16], 人気 [17]
        if len(tds) > 16:
            row["odds"] = _to_float(tds[16].get_text(strip=True))
        if len(tds) > 17:
            row["popularity"] = _to_int(tds[17].get_text(strip=True))

        # 馬体重 [18] (例: "480(+4)")
        if len(tds) > 18:
            weight_text = tds[18].get_text(strip=True)
            row.update(_parse_horse_weight(weight_text))

        rows.append(row)

    return rows


def _parse_horse_weight(text: str) -> dict:
    """馬体重テキストをパースする (例: '480(+4)')"""
    result = {"horse_weight": None, "weight_change": None}
    m = re.match(r"(\d+)\(([+-]?\d+)\)", text)
    if m:
        result["horse_weight"] = int(m.group(1))
        result["weight_change"] = int(m.group(2))
    return result


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


def _time_to_sec(time_str: str) -> float | None:
    """タイム文字列を秒に変換 (例: '1:59.8' → 119.8)"""
    m = re.match(r"(\d+):(\d+\.\d+)", time_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None
