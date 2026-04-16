"""期待値ベースの買い目推奨と予算配分"""

from typing import Dict, List
import pandas as pd

from src.betting.probability import (
    all_tansho_ev, all_fukusho_ev, all_umaren_ev, all_wide_ev,
    all_umatan_ev, all_sanrenpuku_ev, all_sanrentan_ev,
)


BET_UNIT = 100  # JRA最低単位


def compute_all_bets(
    top3_probs: Dict[int, float],
    odds: dict,
    enabled_types: List[str] = None,
) -> pd.DataFrame:
    """
    全券種の全組み合わせのEVを計算してDataFrameで返す。

    Args:
        top3_probs: {horse_number: AI top3 probability}
        odds: fetch_all_odds() の戻り値
        enabled_types: 計算する券種リスト（None = 全部）
    """
    if enabled_types is None:
        enabled_types = ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]

    all_bets = []

    if "単勝" in enabled_types and odds.get("tansho"):
        all_bets.extend(all_tansho_ev(top3_probs, odds["tansho"]))

    if "複勝" in enabled_types and odds.get("fukusho_min"):
        all_bets.extend(all_fukusho_ev(top3_probs, odds["fukusho_min"]))

    if "馬連" in enabled_types and odds.get("umaren"):
        all_bets.extend(all_umaren_ev(top3_probs, odds["umaren"]))

    if "ワイド" in enabled_types and odds.get("wide_min"):
        all_bets.extend(all_wide_ev(top3_probs, odds["wide_min"]))

    if "馬単" in enabled_types and odds.get("umatan"):
        all_bets.extend(all_umatan_ev(top3_probs, odds["umatan"]))

    if "三連複" in enabled_types and odds.get("sanrenpuku"):
        all_bets.extend(all_sanrenpuku_ev(top3_probs, odds["sanrenpuku"]))

    if "三連単" in enabled_types and odds.get("sanrentan"):
        all_bets.extend(all_sanrentan_ev(top3_probs, odds["sanrentan"]))

    if not all_bets:
        return pd.DataFrame(columns=["type", "combo", "horses", "prob", "odds", "ev"])

    df = pd.DataFrame(all_bets)
    df = df.sort_values("ev", ascending=False).reset_index(drop=True)
    return df


def allocate_budget(
    bets_df: pd.DataFrame,
    budget: int,
    ev_threshold: float = 1.1,
    max_per_type: int = None,
    strategy: str = "greedy_ev",
    kelly_fraction: float = 1.0,
) -> pd.DataFrame:
    """
    予算内で買い目を選択する。

    strategy:
        - "greedy_ev": EV降順に100円ずつ購入
        - "kelly": Kelly基準で配分（小数点切り上げで100円単位）
        - "half_kelly": Half Kelly（Kelly × 0.5）
        - "proportional_ev": EV - 1 に比例して配分

    kelly_fraction: Kelly基準の掛率（1.0=フルKelly, 0.5=ハーフKelly）
    """
    if len(bets_df) == 0 or budget < BET_UNIT:
        return pd.DataFrame()

    # 閾値フィルタ
    candidates = bets_df[bets_df["ev"] >= ev_threshold].copy()
    if len(candidates) == 0:
        return pd.DataFrame()

    # 券種ごとに上位N件まで（指定があれば）
    if max_per_type is not None and max_per_type > 0:
        parts = []
        for _, group in candidates.groupby("type"):
            parts.append(group.nlargest(max_per_type, "ev"))
        candidates = pd.concat(parts, ignore_index=True)
        candidates = candidates.sort_values("ev", ascending=False).reset_index(drop=True)

    if strategy == "greedy_ev":
        return _allocate_greedy(candidates, budget)
    elif strategy == "kelly":
        return _allocate_kelly(candidates, budget, fraction=kelly_fraction)
    elif strategy == "half_kelly":
        return _allocate_kelly(candidates, budget, fraction=0.5)
    elif strategy == "proportional_ev":
        return _allocate_proportional(candidates, budget)
    else:
        return _allocate_greedy(candidates, budget)


def _allocate_greedy(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    """EV降順に100円ずつ購入。最良EVに集中投資ではなく分散して購入。"""
    n_units = budget // BET_UNIT
    if n_units == 0:
        return pd.DataFrame()

    selected = candidates.head(n_units).copy()
    selected["stake"] = BET_UNIT
    selected["expected_return"] = selected["stake"] * selected["ev"]
    return selected.reset_index(drop=True)


def _allocate_kelly(candidates: pd.DataFrame, budget: int, fraction: float = 1.0) -> pd.DataFrame:
    """
    Kelly基準: f* = (p × b - q) / b
    where b = odds - 1, p = win prob, q = 1-p
    各候補の Kelly 割合を計算し、合計100%に正規化して配分。
    fraction: 掛率（1.0=フルKelly, 0.5=ハーフKelly）
    """
    df = candidates.copy()
    df["b"] = df["odds"] - 1
    df["kelly_f"] = ((df["prob"] * df["b"]) - (1 - df["prob"])) / df["b"]
    df["kelly_f"] = (df["kelly_f"] * fraction).clip(lower=0)

    total_f = df["kelly_f"].sum()
    if total_f <= 0:
        return _allocate_greedy(candidates, budget)

    # 正規化して予算配分
    df["raw_stake"] = (df["kelly_f"] / total_f) * budget
    # 100円単位に丸める
    df["stake"] = (df["raw_stake"] // BET_UNIT * BET_UNIT).astype(int)
    df = df[df["stake"] >= BET_UNIT].copy()
    df["expected_return"] = df["stake"] * df["ev"]
    df = df.drop(columns=["b", "kelly_f", "raw_stake"])
    return df.reset_index(drop=True)


def _allocate_proportional(candidates: pd.DataFrame, budget: int) -> pd.DataFrame:
    """エッジ (EV - 1) に比例して配分"""
    df = candidates.copy()
    df["edge"] = (df["ev"] - 1).clip(lower=0)
    total_edge = df["edge"].sum()
    if total_edge <= 0:
        return _allocate_greedy(candidates, budget)

    df["raw_stake"] = (df["edge"] / total_edge) * budget
    df["stake"] = (df["raw_stake"] // BET_UNIT * BET_UNIT).astype(int)
    df = df[df["stake"] >= BET_UNIT].copy()
    df["expected_return"] = df["stake"] * df["ev"]
    df = df.drop(columns=["edge", "raw_stake"])
    return df.reset_index(drop=True)
