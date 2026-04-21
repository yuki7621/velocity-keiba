"""個別レースページから結果データをスクレイピングする"""

import re
import requests
from bs4 import BeautifulSoup
from config.settings import NETKEIBA_BASE_URL, USER_AGENT


def scrape_race(race_id: str) -> dict | None:
    """
    レースIDから結果を取得する。
    1) db.netkeiba.com（従来方式）を試す
    2) 失敗したら race.netkeiba.com/race/result.html（速報）にフォールバック

    Returns:
        {
            "race_info": {...},      # レース基本情報
            "results": [{...}, ...]  # 各馬の成績
        }
    """
    # 1) db.netkeiba.com から取得
    result = _scrape_from_db(race_id)
    if result and result.get("results"):
        return result

    # 2) race.netkeiba.com にフォールバック（直近レース用）
    result = _scrape_from_race_site(race_id)
    if result and result.get("results"):
        return result

    return result  # 結果なし（レース未実施など）でもrace_infoは返す


def _scrape_from_db(race_id: str) -> dict | None:
    """db.netkeiba.com から結果を取得（従来方式）"""
    url = f"{NETKEIBA_BASE_URL}/race/{race_id}/"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.encoding = "EUC-JP"
    except Exception:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "lxml")

    race_info = _parse_race_info(soup, race_id)
    if race_info is None:
        return None

    results = _parse_results_table(soup, race_id)

    # 払戻情報を取得
    payouts = _parse_payouts(soup, race_id)

    return {"race_info": race_info, "results": results, "payouts": payouts}


def _scrape_from_race_site(race_id: str) -> dict | None:
    """race.netkeiba.com/race/result.html から結果を取得（速報ページ）"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.encoding = "EUC-JP"
    except Exception:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # レース情報
    race_info = {"race_id": race_id}

    # タイトル
    title_tag = soup.select_one(".RaceName")
    if title_tag:
        race_info["title"] = title_tag.get_text(strip=True)
    else:
        title_el = soup.select_one("title")
        if title_el:
            race_info["title"] = title_el.get_text(strip=True).split("|")[0].strip()

    # コース情報 (RaceData01)
    race_data = soup.select_one(".RaceData01")
    if race_data:
        text = race_data.get_text(strip=True)
        if "芝" in text:
            race_info["surface"] = "芝"
        elif "ダート" in text or "ダ" in text:
            race_info["surface"] = "ダート"
        elif "障" in text:
            race_info["surface"] = "障害"
        m = re.search(r"(\d{3,5})m", text)
        if m:
            race_info["distance"] = int(m.group(1))
        m = re.search(r"天候:(\S+)", text)
        if m:
            race_info["weather"] = m.group(1)
        m = re.search(r"馬場:(\S+)", text)
        if m:
            race_info["condition"] = m.group(1)

    # 日付 — 複数箇所から探す
    date_found = False

    # 1) RaceData02 から探す
    race_data2 = soup.select_one(".RaceData02")
    if race_data2:
        text2 = race_data2.get_text(strip=True)
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text2)
        if m:
            race_info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            date_found = True

    # 2) ページ全体のテキストから日付パターンを探す
    if not date_found:
        page_text = soup.get_text()
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", page_text)
        if m:
            race_info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            date_found = True

    # 3) titleタグから探す (例: "レース名 | 2026年4月12日")
    if not date_found:
        title_el = soup.select_one("title")
        if title_el:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title_el.get_text())
            if m:
                race_info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                date_found = True

    # 4) meta タグから探す
    if not date_found:
        for meta in soup.select("meta"):
            content = meta.get("content", "")
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", content)
            if m:
                race_info["date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                date_found = True
                break

    # 5) 最終手段: race_idから年を取得し、開催情報から推定できない場合はNoneのまま

    # 競馬場をrace_idから推定
    venue_code = race_id[4:6]
    from config.settings import VENUE_CODES
    if venue_code in VENUE_CODES:
        race_info["venue"] = VENUE_CODES[venue_code]
        race_info["venue_code"] = venue_code

    # 頭数
    head_count_tag = soup.select_one(".RaceData02 span")
    if head_count_tag:
        m = re.search(r"(\d+)頭", head_count_tag.get_text())
        if m:
            race_info["head_count"] = int(m.group(1))

    # 結果テーブル
    results = _parse_result_table_race_site(soup, race_id)

    # 払戻情報
    payouts = _parse_payouts_race_site(soup, race_id)

    if not results:
        return {"race_info": race_info, "results": [], "payouts": payouts}

    if "head_count" not in race_info:
        race_info["head_count"] = len(results)

    return {"race_info": race_info, "results": results, "payouts": payouts}


def _parse_result_table_race_site(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """race.netkeiba.com の結果テーブルをパース"""
    rows = []
    table = soup.select_one(".ResultTableWrap table")
    if table is None:
        # 結果未確定（レース前）
        return rows

    for tr in table.select("tbody tr"):
        tds = tr.select("td")
        if len(tds) < 10:
            continue

        row = {"race_id": race_id}

        # 着順 [0]
        finish_text = tds[0].get_text(strip=True)
        row["finish_position"] = int(finish_text) if finish_text.isdigit() else 0

        # 枠番 [1], 馬番 [2]
        row["gate_number"] = _to_int(tds[1].get_text(strip=True))
        row["post_number"] = _to_int(tds[2].get_text(strip=True))

        # 馬名・馬ID [3]
        horse_link = tds[3].select_one("a")
        if horse_link:
            row["horse_name"] = horse_link.get_text(strip=True)
            href = horse_link.get("href", "")
            m = re.search(r"/horse/(\w+)", href)
            row["horse_id"] = m.group(1) if m else ""
        else:
            row["horse_name"] = tds[3].get_text(strip=True)
            row["horse_id"] = ""

        # 性齢 [4], 斤量 [5]
        row["sex_age"] = tds[4].get_text(strip=True)
        row["weight_carried"] = _to_float(tds[5].get_text(strip=True))

        # 騎手 [6]
        jockey_link = tds[6].select_one("a")
        if jockey_link:
            row["jockey_name"] = jockey_link.get_text(strip=True)
            href = jockey_link.get("href", "")
            m = re.search(r"/jockey/(?:result/recent/)?(\d+)", href)
            row["jockey_id"] = m.group(1) if m else ""
        else:
            row["jockey_name"] = tds[6].get_text(strip=True)
            row["jockey_id"] = ""

        # タイム [7]
        row["finish_time"] = tds[7].get_text(strip=True)
        row["finish_time_sec"] = _time_to_sec(row["finish_time"])

        # 着差 [8] — スキップ

        # 後半のカラムはテーブル構造に依存するので安全に取得
        remaining = tds[9:]

        # 通過順、上がり3F、単勝オッズ、人気、馬体重を順に探す
        for td in remaining:
            text = td.get_text(strip=True)

            # 通過順 (例: "3-3-2-1")
            if re.match(r"^\d+-\d+", text) and "passing_order" not in row:
                row["passing_order"] = text
                continue

            # 上がり3F (例: "34.5")
            if re.match(r"^\d{2}\.\d$", text) and "last_3f" not in row:
                row["last_3f"] = _to_float(text)
                continue

            # 単勝オッズ (例: "3.5")
            if re.match(r"^\d+\.\d$", text) and "odds" not in row:
                row["odds"] = _to_float(text)
                continue

            # 人気 (例: "1")
            if text.isdigit() and int(text) <= 30 and "popularity" not in row and "odds" in row:
                row["popularity"] = _to_int(text)
                continue

            # 馬体重 (例: "480(+4)")
            m = re.match(r"(\d+)\(([+-]?\d+)\)", text)
            if m and "horse_weight" not in row:
                row["horse_weight"] = int(m.group(1))
                row["weight_change"] = int(m.group(2))
                continue

            # 調教師 (リンクを別途探す)
            trainer_link = td.select_one("a[href*='/trainer/']")
            if trainer_link and "trainer_id" not in row:
                row["trainer_name"] = trainer_link.get_text(strip=True)
                href = trainer_link.get("href", "")
                m2 = re.search(r"/trainer/(?:result/recent/)?(\d+)", href)
                row["trainer_id"] = m2.group(1) if m2 else ""

        rows.append(row)

    return rows


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
            m = re.search(r"/jockey/(?:result/recent/)?(\d+)", href)
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

        # 調教師 [22] (例: href="/trainer/result/recent/01126/")
        if len(tds) > 22:
            trainer_link = tds[22].select_one("a")
            if trainer_link:
                row["trainer_name"] = trainer_link.get_text(strip=True)
                href = trainer_link.get("href", "")
                m = re.search(r"/trainer/(?:result/recent/)?(\d+)", href)
                row["trainer_id"] = m.group(1) if m else ""
            else:
                row["trainer_name"] = tds[22].get_text(strip=True)
                row["trainer_id"] = ""

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


def _parse_payouts(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """
    db.netkeiba.com の払戻テーブルから複勝・単勝の払戻を取得する。

    払戻テーブル構造 (.pay_table_01):
        各行: [券種名, 馬番, 払戻金, 人気]
        複勝は3行（1着〜3着の馬番ごとに1行）
    """
    payouts = []

    # 払戻テーブルを探す
    tables = soup.select(".pay_table_01")
    if not tables:
        return payouts

    for table in tables:
        for tr in table.select("tr"):
            th = tr.select_one("th")
            if th is None:
                continue
            bet_type_text = th.get_text(strip=True)

            # 単勝 or 複勝のみ取得
            if "単勝" in bet_type_text:
                bet_type = "tansho"
            elif "複勝" in bet_type_text:
                bet_type = "fukusho"
            else:
                continue

            tds = tr.select("td")
            if len(tds) < 2:
                continue

            # 馬番と払戻金を取得
            # 複勝の場合、1つのセルに複数の馬番・払戻が<br>区切りで入っていることがある
            numbers_td = tds[0]
            payouts_td = tds[1]

            numbers_text = numbers_td.get_text(separator="\n", strip=True).split("\n")
            payouts_text = payouts_td.get_text(separator="\n", strip=True).split("\n")

            for num_str, pay_str in zip(numbers_text, payouts_text):
                num_str = num_str.strip()
                pay_str = pay_str.strip().replace(",", "").replace("円", "")

                if not num_str.isdigit():
                    continue
                try:
                    payout_val = int(pay_str)
                except (ValueError, TypeError):
                    continue

                payouts.append({
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "horse_number": int(num_str),
                    "payout": payout_val,
                })

    return payouts


def _parse_payouts_race_site(soup: BeautifulSoup, race_id: str) -> list[dict]:
    """
    race.netkeiba.com の払戻テーブルから複勝・単勝の払戻を取得する。

    PayTable 構造:
        .Payout_Detail_Table 内の各行
    """
    payouts = []

    # race.netkeiba.com の払戻テーブル
    table = soup.select_one(".FullWrap .Result_Pay_Back table")
    if table is None:
        # 別のセレクタを試す
        table = soup.select_one(".PayTableWrap table")
    if table is None:
        return payouts

    for tr in table.select("tr"):
        th = tr.select_one("th")
        if th is None:
            continue
        bet_type_text = th.get_text(strip=True)

        if "単勝" in bet_type_text:
            bet_type = "tansho"
        elif "複勝" in bet_type_text:
            bet_type = "fukusho"
        else:
            continue

        tds = tr.select("td")
        if len(tds) < 2:
            continue

        numbers_text = tds[0].get_text(separator="\n", strip=True).split("\n")
        payouts_text = tds[1].get_text(separator="\n", strip=True).split("\n")

        for num_str, pay_str in zip(numbers_text, payouts_text):
            num_str = num_str.strip()
            pay_str = pay_str.strip().replace(",", "").replace("円", "")

            if not num_str.isdigit():
                continue
            try:
                payout_val = int(pay_str)
            except (ValueError, TypeError):
                continue

            payouts.append({
                "race_id": race_id,
                "bet_type": bet_type,
                "horse_number": int(num_str),
                "payout": payout_val,
            })

    return payouts


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
