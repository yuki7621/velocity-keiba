"""AIの3着内確率から各券種の組み合わせ確率を計算する

Plackett-Luce モデルを用いて、各馬の "強さ" を3着内確率から推定し、
順位確率を計算する。
"""

from itertools import combinations, permutations
from typing import Dict, List, Tuple

import numpy as np


def compute_strengths(top3_probs: Dict[int, float]) -> Dict[int, float]:
    """
    3着内確率から各馬の "強さ"（Plackett-Luce のパラメータ）を計算する。

    AI の top3 確率は単調に win 確率と相関するため、
    強さ ≈ p^k の形で正規化する。k=1 で単純比例。
    """
    horses = list(top3_probs.keys())
    raw = np.array([max(0.001, top3_probs[h]) for h in horses])
    # 正規化（合計1）
    strengths = raw / raw.sum()
    return {h: float(s) for h, s in zip(horses, strengths)}


def win_prob(strengths: Dict[int, float]) -> Dict[int, float]:
    """単勝確率（Plackett-Luce の1着確率 = 強さそのもの）"""
    return dict(strengths)


def place_prob(top3_probs: Dict[int, float]) -> Dict[int, float]:
    """複勝確率 = AIの3着内確率をそのまま使用"""
    return dict(top3_probs)


def exacta_prob(strengths: Dict[int, float], horse_a: int, horse_b: int) -> float:
    """馬単 P(A=1着, B=2着) = w_A / S × w_B / (S - w_A)"""
    w_a = strengths.get(horse_a, 0)
    w_b = strengths.get(horse_b, 0)
    total = sum(strengths.values())
    if total - w_a <= 0:
        return 0.0
    return (w_a / total) * (w_b / (total - w_a))


def quinella_prob(strengths: Dict[int, float], horse_a: int, horse_b: int) -> float:
    """馬連 P(A,B が順不同で1-2着) = exacta(A,B) + exacta(B,A)"""
    return exacta_prob(strengths, horse_a, horse_b) + exacta_prob(strengths, horse_b, horse_a)


def trifecta_prob(strengths: Dict[int, float], a: int, b: int, c: int) -> float:
    """三連単 P(A=1, B=2, C=3)"""
    w_a = strengths.get(a, 0)
    w_b = strengths.get(b, 0)
    w_c = strengths.get(c, 0)
    total = sum(strengths.values())
    denom1 = total - w_a
    denom2 = total - w_a - w_b
    if denom1 <= 0 or denom2 <= 0:
        return 0.0
    return (w_a / total) * (w_b / denom1) * (w_c / denom2)


def trio_prob(strengths: Dict[int, float], a: int, b: int, c: int) -> float:
    """三連複 P({A,B,C} = top3 順不同) = sum over 6 permutations"""
    total = 0.0
    for perm in permutations([a, b, c]):
        total += trifecta_prob(strengths, *perm)
    return total


def wide_prob(strengths: Dict[int, float], a: int, b: int) -> float:
    """ワイド P(A,B が両方3着以内)
    = sum over (3 perms where A,B in top3) × P(perm)
    """
    horses = [h for h in strengths.keys() if h != a and h != b]
    total = 0.0
    # A,B が top3 に入る全パターン: (A,B,X), (A,X,B), (B,A,X), (B,X,A), (X,A,B), (X,B,A)
    for x in horses:
        for perm in [(a, b, x), (a, x, b), (b, a, x), (b, x, a), (x, a, b), (x, b, a)]:
            total += trifecta_prob(strengths, *perm)
    return total


# ──────────────────────────────────────────────
# 一括計算（全組み合わせのEV）
# ──────────────────────────────────────────────

def all_tansho_ev(top3_probs: Dict[int, float], odds: Dict[str, float]) -> List[dict]:
    """全頭の単勝EV"""
    strengths = compute_strengths(top3_probs)
    win = win_prob(strengths)
    results = []
    for h, p in win.items():
        o = odds.get(str(h))
        if o is None:
            continue
        ev = p * o
        results.append({
            "type": "単勝", "combo": str(h), "horses": (h,),
            "prob": p, "odds": o, "ev": ev,
        })
    return results


def all_fukusho_ev(top3_probs: Dict[int, float], odds_min: Dict[str, float]) -> List[dict]:
    """全頭の複勝EV（最低オッズで保守的に計算）"""
    results = []
    for h, p in top3_probs.items():
        o = odds_min.get(str(h))
        if o is None:
            continue
        ev = p * o
        results.append({
            "type": "複勝", "combo": str(h), "horses": (h,),
            "prob": p, "odds": o, "ev": ev,
        })
    return results


def all_umaren_ev(top3_probs: Dict[int, float], odds: Dict[str, float]) -> List[dict]:
    """全組み合わせの馬連EV"""
    strengths = compute_strengths(top3_probs)
    horses = sorted(top3_probs.keys())
    results = []
    for a, b in combinations(horses, 2):
        key = f"{a}-{b}"
        o = odds.get(key)
        if o is None:
            continue
        p = quinella_prob(strengths, a, b)
        results.append({
            "type": "馬連", "combo": key, "horses": (a, b),
            "prob": p, "odds": o, "ev": p * o,
        })
    return results


def all_wide_ev(top3_probs: Dict[int, float], odds_min: Dict[str, float]) -> List[dict]:
    """全組み合わせのワイドEV（最低オッズで保守的）"""
    strengths = compute_strengths(top3_probs)
    horses = sorted(top3_probs.keys())
    results = []
    for a, b in combinations(horses, 2):
        key = f"{a}-{b}"
        o = odds_min.get(key)
        if o is None:
            continue
        p = wide_prob(strengths, a, b)
        results.append({
            "type": "ワイド", "combo": key, "horses": (a, b),
            "prob": p, "odds": o, "ev": p * o,
        })
    return results


def all_umatan_ev(top3_probs: Dict[int, float], odds: Dict[str, float]) -> List[dict]:
    """全組み合わせの馬単EV（順序あり）"""
    strengths = compute_strengths(top3_probs)
    horses = sorted(top3_probs.keys())
    results = []
    for a, b in permutations(horses, 2):
        key = f"{a}-{b}"
        o = odds.get(key)
        if o is None:
            continue
        p = exacta_prob(strengths, a, b)
        results.append({
            "type": "馬単", "combo": key, "horses": (a, b),
            "prob": p, "odds": o, "ev": p * o,
        })
    return results


def all_sanrenpuku_ev(top3_probs: Dict[int, float], odds: Dict[str, float]) -> List[dict]:
    """全組み合わせの三連複EV"""
    strengths = compute_strengths(top3_probs)
    horses = sorted(top3_probs.keys())
    results = []
    for a, b, c in combinations(horses, 3):
        key = f"{a}-{b}-{c}"
        o = odds.get(key)
        if o is None:
            continue
        p = trio_prob(strengths, a, b, c)
        results.append({
            "type": "三連複", "combo": key, "horses": (a, b, c),
            "prob": p, "odds": o, "ev": p * o,
        })
    return results


def all_sanrentan_ev(top3_probs: Dict[int, float], odds: Dict[str, float]) -> List[dict]:
    """全組み合わせの三連単EV（数が多いのでEV>=1のみ返す）"""
    strengths = compute_strengths(top3_probs)
    horses = sorted(top3_probs.keys())
    results = []
    for a, b, c in permutations(horses, 3):
        key = f"{a}-{b}-{c}"
        o = odds.get(key)
        if o is None:
            continue
        p = trifecta_prob(strengths, a, b, c)
        ev = p * o
        if ev < 0.5:  # 早期足切り
            continue
        results.append({
            "type": "三連単", "combo": key, "horses": (a, b, c),
            "prob": p, "odds": o, "ev": ev,
        })
    return results
