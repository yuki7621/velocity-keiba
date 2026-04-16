"""netkeibaのオッズAPIから現在オッズを取得する"""

import requests
import json
import re
from typing import Optional

from config.settings import USER_AGENT


# netkeibaのオッズAPI
ODDS_API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"

# type値: netkeiba内部の券種コード
TYPE_TANSHO = 1   # 単勝
TYPE_FUKUSHO = 1  # 複勝（単勝と同レスポンスに含まれる）
TYPE_WAKUREN = 3  # 枠連
TYPE_UMAREN = 4   # 馬連
TYPE_WIDE = 5     # ワイド
TYPE_UMATAN = 6   # 馬単
TYPE_SANRENPUKU = 7  # 三連複
TYPE_SANRENTAN = 8   # 三連単


def _fetch_odds_json(race_id: str, odds_type: int) -> Optional[dict]:
    """オッズAPIから生のJSONを取得"""
    params = {
        "race_id": race_id,
        "type": odds_type,
        "action": "init",
    }
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": f"https://race.netkeiba.com/odds/index.html?race_id={race_id}",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    try:
        resp = requests.get(ODDS_API_URL, params=params, headers=headers, timeout=20)
        if resp.status_code != 200:
            return None
        # JSONP形式の場合は剥がす
        text = resp.text.strip()
        m = re.match(r"^[\w\$]+\((.*)\)\s*;?\s*$", text, re.DOTALL)
        if m:
            text = m.group(1)
        return json.loads(text)
    except Exception as e:
        print(f"odds fetch error: {e}")
        return None


def _normalize_odds_key(raw_key: str) -> str:
    """
    netkeibaのオッズキーを正規化する。
    "01" → "1", "0102" → "1-2", "010203" → "1-2-3"
    """
    s = str(raw_key)
    if len(s) == 2:
        # 単勝・複勝: "01" → "1"
        return str(int(s))
    elif len(s) == 4:
        # 馬連・ワイド・馬単: "0102" → "1-2"
        return f"{int(s[:2])}-{int(s[2:])}"
    elif len(s) == 6:
        # 三連複・三連単: "010203" → "1-2-3"
        return f"{int(s[:2])}-{int(s[2:4])}-{int(s[4:])}"
    return s


def _parse_odds_dict(odds_data) -> dict:
    """
    オッズ辞書を {key: float_odds} 形式に正規化する。
    netkeibaのレスポンス形式は券種により異なるため、複数パターンに対応。
    キー例: 単勝 → "1", 馬連 → "1-2", 三連単 → "1-2-3"
    値: list[str] の場合は最初の要素（=現在オッズ）を採用
    """
    result = {}
    if not isinstance(odds_data, dict):
        return result

    for key, val in odds_data.items():
        if isinstance(val, list):
            v = val[0] if val else None
        elif isinstance(val, dict):
            v = val.get("odds") or val.get("0")
        else:
            v = val

        if v is None or v == "" or v == "**.*":
            continue
        try:
            result[_normalize_odds_key(key)] = float(v)
        except (ValueError, TypeError):
            continue

    return result


def fetch_all_odds(race_id: str) -> dict:
    """
    全券種のオッズをまとめて取得する。
    Returns:
        {
            "tansho": {"1": 3.5, "2": 12.0, ...},
            "fukusho": {"1": [1.5, 1.8], ...},  # 複勝は範囲
            "umaren": {"1-2": 23.5, ...},
            "wide": {"1-2": 5.6, ...},
            "umatan": {"1-2": 45.0, ...},
            "sanrenpuku": {"1-2-3": 120.0, ...},
            "sanrentan": {"1-2-3": 800.0, ...},
        }
    """
    result = {
        "tansho": {},
        "fukusho_min": {},
        "fukusho_max": {},
        "umaren": {},
        "wide_min": {},
        "wide_max": {},
        "umatan": {},
        "sanrenpuku": {},
        "sanrentan": {},
    }

    # 単勝・複勝（type=1で両方取れる）
    data = _fetch_odds_json(race_id, 1)
    if data and "data" in data:
        odds = data["data"].get("odds", {})
        # 単勝
        tansho_raw = odds.get("1", {})
        for k, v in tansho_raw.items():
            try:
                nk = _normalize_odds_key(k)
                if isinstance(v, list) and len(v) > 0:
                    result["tansho"][nk] = float(v[0])
                else:
                    result["tansho"][nk] = float(v)
            except (ValueError, TypeError):
                continue
        # 複勝
        fukusho_raw = odds.get("2", {})
        for k, v in fukusho_raw.items():
            try:
                nk = _normalize_odds_key(k)
                if isinstance(v, list) and len(v) >= 2:
                    result["fukusho_min"][nk] = float(v[0])
                    result["fukusho_max"][nk] = float(v[1])
                elif isinstance(v, list) and len(v) == 1:
                    result["fukusho_min"][nk] = float(v[0])
                    result["fukusho_max"][nk] = float(v[0])
            except (ValueError, TypeError):
                continue

    # 馬連
    data = _fetch_odds_json(race_id, TYPE_UMAREN)
    if data and "data" in data:
        odds = data["data"].get("odds", {})
        for type_key, type_odds in odds.items():
            result["umaren"].update(_parse_odds_dict(type_odds))

    # ワイド（範囲）
    data = _fetch_odds_json(race_id, TYPE_WIDE)
    if data and "data" in data:
        odds = data["data"].get("odds", {})
        for type_key, type_odds in odds.items():
            for k, v in type_odds.items():
                try:
                    nk = _normalize_odds_key(k)
                    if isinstance(v, list) and len(v) >= 2:
                        result["wide_min"][nk] = float(v[0])
                        result["wide_max"][nk] = float(v[1])
                    elif isinstance(v, list) and len(v) == 1:
                        result["wide_min"][nk] = float(v[0])
                        result["wide_max"][nk] = float(v[0])
                except (ValueError, TypeError):
                    continue

    # 馬単
    data = _fetch_odds_json(race_id, TYPE_UMATAN)
    if data and "data" in data:
        odds = data["data"].get("odds", {})
        for type_key, type_odds in odds.items():
            result["umatan"].update(_parse_odds_dict(type_odds))

    # 三連複
    data = _fetch_odds_json(race_id, TYPE_SANRENPUKU)
    if data and "data" in data:
        odds = data["data"].get("odds", {})
        for type_key, type_odds in odds.items():
            result["sanrenpuku"].update(_parse_odds_dict(type_odds))

    # 三連単
    data = _fetch_odds_json(race_id, TYPE_SANRENTAN)
    if data and "data" in data:
        odds = data["data"].get("odds", {})
        for type_key, type_odds in odds.items():
            result["sanrentan"].update(_parse_odds_dict(type_odds))

    return result
