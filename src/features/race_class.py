"""レースクラス・馬齢・天候の共通ユーティリティ (v9)

学習(build_features)と予測(predict_features)の両経路から使う共通ロジック。
過去に「両経路で計算がズレて予測が食い違う」事故があったため、
定義は必ずこのモジュール1箇所に集約する。
"""

import re

import numpy as np

# ── レースクラス（数値が大きいほど上位）──
# netkeiba の title からクラスを判定する。
# 注意: 特別・重賞は「根岸ステークス」のような固有名のみでクラス表記が無い
#       （全1225title中685が固有名）。その場合は NaN とし、
#       is_named_race フラグで「特別戦以上」であることだけを伝える。
#       ※ 賞金(prize)を再スクレイプすれば全レースで厳密なクラス判定が可能。
CLASS_SHINBA = 0.0      # 新馬
CLASS_MISHOURI = 1.0    # 未勝利
CLASS_1WIN = 2.0        # 1勝クラス (旧500万下)
CLASS_2WIN = 3.0        # 2勝クラス (旧1000万下)
CLASS_3WIN = 4.0        # 3勝クラス (旧1600万下)
CLASS_OPEN = 5.0        # オープン

_CLASS_RULES = [
    (("新馬", "メイクデビュー"), CLASS_SHINBA),
    (("未勝利", "未出走"), CLASS_MISHOURI),
    (("1勝クラス", "500万下"), CLASS_1WIN),
    (("2勝クラス", "1000万下"), CLASS_2WIN),
    (("3勝クラス", "1600万下"), CLASS_3WIN),
    (("オープン",), CLASS_OPEN),
]

# 固有名を持つレース（特別・オープン・重賞）の判定
_NAMED_RE = re.compile(r"ステークス|Ｓ$|S$|賞|杯|記念|カップ|特別|ダービー|オークス|菊花|皐月|天皇")


def parse_race_class(title) -> float:
    """title からレースクラスを判定する。固有名レースは NaN。"""
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return np.nan
    t = str(title)
    for keywords, value in _CLASS_RULES:
        if any(k in t for k in keywords):
            return value
    return np.nan  # 固有名レース（クラス不明）


def is_named_race(title) -> float:
    """固有名を持つレース（特別/OP/重賞）なら 1.0、平場クラス戦なら 0.0"""
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return np.nan
    return 1.0 if _NAMED_RE.search(str(title)) else 0.0


def horse_age_from_id(horse_id, race_date) -> float:
    """horse_id の先頭4桁(生年) と レース年 から馬齢を算出する。

    netkeiba の horse_id は先頭4桁が生年（例: 2023100926 → 2023年生）。
    実データ500件で「3歳戦=3」「2歳戦=2」の一致を確認済みのため、
    性齢列を再スクレイプしなくても馬齢が得られる。
    """
    try:
        birth_year = int(str(horse_id)[:4])
        race_year = int(str(race_date)[:4])
    except (TypeError, ValueError):
        return np.nan
    age = race_year - birth_year
    # 異常値ガード（競走馬は概ね2〜12歳）
    if age < 1 or age > 20:
        return np.nan
    return float(age)


# ── 天候 ──
# 数値が大きいほど悪天候（含水率が上がる方向）
WEATHER_MAP = {
    "晴": 0.0,
    "曇": 1.0,
    "小雨": 2.0,
    "雨": 3.0,
    "小雪": 4.0,
    "雪": 5.0,
}


def weather_to_num(weather) -> float:
    """天候を数値化する。未知の値は NaN。"""
    if weather is None:
        return np.nan
    return WEATHER_MAP.get(str(weather).strip(), np.nan)


# ── 性別 ──
SEX_MAP = {"牡": 0.0, "牝": 1.0, "セ": 2.0, "せん": 2.0, "騸": 2.0}


def sex_to_num(sex) -> float:
    """性別を数値化する。未取得(None)は NaN。"""
    if sex is None:
        return np.nan
    return SEX_MAP.get(str(sex).strip(), np.nan)
