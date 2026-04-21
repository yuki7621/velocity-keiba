"""馬の血統情報（父・母父）を netkeiba から取得する"""

import re
import requests
from bs4 import BeautifulSoup
from config.settings import NETKEIBA_BASE_URL, USER_AGENT


def scrape_pedigree(horse_id: str) -> dict | None:
    """
    horse_id から血統情報を取得する。

    Returns:
        {"horse_id": ..., "sire": "...", "dam_sire": "..."}
        取得失敗時は None
    """
    url = f"{NETKEIBA_BASE_URL}/horse/ped/{horse_id}/"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.encoding = "EUC-JP"
    except Exception:
        return None

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "lxml")

    sire, dam_sire = _parse_blood_table(soup)
    if sire is None and dam_sire is None:
        return None

    return {
        "horse_id": horse_id,
        "sire": sire,
        "dam_sire": dam_sire,
    }


def _parse_blood_table(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """
    血統表 (table.blood_table) から父と母父を抽出する。

    netkeiba の血統表は 5世代×16行 = 32行の構造:
      - Row 0, td[0]  → 父（sire）        [rowspan=16]
      - Row 16, td[0] → 母（dam）         [rowspan=16]
      - Row 16, td[1] → 母父（dam_sire）   [rowspan=8]

    各セル内に最初にある <a> タグの text が馬名。
    """
    table = soup.select_one("table.blood_table")
    if table is None:
        return None, None

    rows = table.select("tr")
    if len(rows) < 32:
        return None, None  # 想定外の構造

    # 父: Row 0, td[0]
    sire = _extract_horse_name(rows[0].select("td"), 0)

    # 母父: Row 16 (32の半分), td[1]
    half_idx = len(rows) // 2
    dam_sire = _extract_horse_name(rows[half_idx].select("td"), 1)

    return sire, dam_sire


def _extract_horse_name(tds, index: int) -> str | None:
    """tds[index] から最初の <a> タグの馬名を抽出"""
    if index >= len(tds):
        return None
    td = tds[index]
    a = td.select_one("a")
    if a is None:
        return _clean_horse_name(td.get_text(strip=True))
    return _clean_horse_name(a.get_text(strip=True))


def _clean_horse_name(name: str) -> str | None:
    """馬名を整形する（括弧内の原語名や記号を除去）

    "ディープインパクト(JPN)" → "ディープインパクト"
    "Top Decile(米)" → "Top Decile"
    "ハービンジャーHarbinger" → "ハービンジャー"（カナ名優先）
    """
    if not name:
        return None
    # 括弧内の原語名や記号を除去
    name = re.sub(r"\s*[\(（][^)）]*[\)）]\s*", "", name)
    name = name.strip()

    # カナ名 + 英語名が連結されているケース: カナが含まれるなら ASCII 部分を削除
    if re.search(r"[ぁ-んァ-ヶー一-龥]", name):
        # カナ/漢字の塊だけ抽出
        m = re.match(r"([ぁ-んァ-ヶー一-龥・]+)", name)
        if m:
            name = m.group(1)

    name = name.strip()
    return name if name else None
