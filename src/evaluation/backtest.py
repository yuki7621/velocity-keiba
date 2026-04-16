"""バックテスト: 過去データで仮想馬券購入し回収率を検証する"""

import pandas as pd
import numpy as np
from src.model.train import FEATURE_COLUMNS, TARGET_COLUMN, prepare_dataset


def run_backtest(
    model,
    df: pd.DataFrame,
    bet_type: str = "top3",
    threshold: float = 0.5,
) -> dict:
    """
    確率閾値ベースのバックテスト（旧方式）。

    Args:
        model: 学習済みモデル
        df: 特徴量付きDataFrame
        bet_type: "top3" = 複勝予想
        threshold: この確率以上の馬に賭ける

    Returns:
        結果の辞書 (的中率、回収率など)
    """
    df = prepare_dataset(df)

    # 最新20%をテスト期間に
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[FEATURE_COLUMNS]

    # 予測確率
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    # 閾値以上の馬に賭ける
    bets = test_df[test_df["pred_prob"] >= threshold].copy()

    if len(bets) == 0:
        print("賭ける馬が0頭です。閾値を下げてください。")
        return {}

    total_bets = len(bets)
    hits = bets[bets["finish_position"] <= 3]
    total_hits = len(hits)

    # 複勝の簡易回収率計算
    bets["estimated_payout"] = np.where(
        bets["finish_position"] <= 3,
        bets["odds"] * 0.3,
        0,
    )
    total_payout = bets["estimated_payout"].sum()
    roi = total_payout / total_bets * 100

    hit_rate = total_hits / total_bets * 100

    result = {
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
        "roi": roi,
        "threshold": threshold,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }

    _print_result("バックテスト結果 (確率閾値方式)", result)
    return result


def run_ev_backtest(
    model,
    df: pd.DataFrame,
    ev_threshold: float = 1.0,
    min_odds: float = 3.0,
    max_odds: float = 150.0,
) -> dict:
    """
    期待値(EV)ベースのバックテスト。
    「AIの予測確率 × 配当 > ev_threshold」の馬だけに賭ける。

    考え方:
        オッズ10倍 → 市場は「勝率10%」と評価
        AI予測   → 「勝率25%」
        期待値   = 0.25 × (10 × 0.3) = 0.75  (複勝概算)
        → ev_threshold以上なら賭ける

    Args:
        model: 学習済みモデル
        df: 特徴量付きDataFrame
        ev_threshold: 期待値がこの値以上なら賭ける（1.0超 = 期待利益あり）
        min_odds: 最低オッズ（低すぎる人気馬を除外）
        max_odds: 最大オッズ（高すぎる大穴を除外）
    """
    df = prepare_dataset(df)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[FEATURE_COLUMNS]
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    # オッズが欠損・範囲外の行を除外
    test_df = test_df[test_df["odds"].notna()].copy()
    test_df = test_df[
        (test_df["odds"] >= min_odds) & (test_df["odds"] <= max_odds)
    ].copy()

    # 期待値を計算
    # 複勝概算配当 = 単勝オッズ × 0.3
    # 期待値 = AI予測確率 × 概算配当倍率
    test_df["estimated_fukusho_odds"] = (test_df["odds"] * 0.3).clip(lower=1.1)
    test_df["expected_value"] = test_df["pred_prob"] * test_df["estimated_fukusho_odds"]

    # 期待値が閾値以上の馬に賭ける
    bets = test_df[test_df["expected_value"] >= ev_threshold].copy()

    if len(bets) == 0:
        print(f"EV閾値 {ev_threshold} で賭ける馬が0頭です。")
        return {}

    total_bets = len(bets)
    hits = bets[bets["finish_position"] <= 3]
    total_hits = len(hits)
    hit_rate = total_hits / total_bets * 100

    # 回収率 = 的中した馬の概算配当合計 / 賭け数
    bets["actual_payout"] = np.where(
        bets["finish_position"] <= 3,
        bets["estimated_fukusho_odds"],
        0,
    )
    total_payout = bets["actual_payout"].sum()
    roi = total_payout / total_bets * 100

    # 平均期待値
    avg_ev = bets["expected_value"].mean()
    avg_odds = bets["odds"].mean()
    avg_pred_prob = bets["pred_prob"].mean()

    result = {
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
        "roi": roi,
        "ev_threshold": ev_threshold,
        "min_odds": min_odds,
        "max_odds": max_odds,
        "avg_expected_value": avg_ev,
        "avg_odds": avg_odds,
        "avg_pred_prob": avg_pred_prob,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }

    _print_result("バックテスト結果 (期待値方式)", result)
    return result


def run_value_bet_backtest(
    model,
    df: pd.DataFrame,
    edge_threshold: float = 0.05,
    min_odds: float = 3.0,
    max_odds: float = 100.0,
) -> dict:
    """
    バリューベット戦略のバックテスト。
    「AI予測確率 - 市場の暗黙確率 > edge_threshold」の馬に賭ける。

    考え方:
        オッズ10倍  → 市場の暗黙確率 = 1/10 = 10%
        AI予測     → 25%
        エッジ     = 25% - 10% = +15%
        → edge_threshold以上なら賭ける（市場が過小評価している）

    Args:
        edge_threshold: AIと市場の確率差がこの値以上なら賭ける
        min_odds: 最低オッズ
        max_odds: 最大オッズ
    """
    df = prepare_dataset(df)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[FEATURE_COLUMNS]
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    # オッズフィルタ
    test_df = test_df[test_df["odds"].notna()].copy()
    test_df = test_df[
        (test_df["odds"] >= min_odds) & (test_df["odds"] <= max_odds)
    ].copy()

    # 市場の暗黙確率（複勝の場合、3頭的中なので概算で 3/オッズ に近いが、
    # 単勝オッズからの概算として 1/オッズ を使い、複勝補正する）
    # 複勝的中率の市場推定 ≒ 3 / 単勝オッズ（ただし上限1.0）
    test_df["market_implied_prob"] = (3.0 / test_df["odds"]).clip(upper=1.0)

    # エッジ = AI予測 - 市場評価
    test_df["edge"] = test_df["pred_prob"] - test_df["market_implied_prob"]

    # エッジが閾値以上の馬に賭ける
    bets = test_df[test_df["edge"] >= edge_threshold].copy()

    if len(bets) == 0:
        print(f"エッジ閾値 {edge_threshold} で賭ける馬が0頭です。")
        return {}

    total_bets = len(bets)
    hits = bets[bets["finish_position"] <= 3]
    total_hits = len(hits)
    hit_rate = total_hits / total_bets * 100

    # 回収率
    bets["actual_payout"] = np.where(
        bets["finish_position"] <= 3,
        bets["odds"] * 0.3,
        0,
    )
    total_payout = bets["actual_payout"].sum()
    roi = total_payout / total_bets * 100

    avg_edge = bets["edge"].mean()
    avg_odds = bets["odds"].mean()

    result = {
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
        "roi": roi,
        "edge_threshold": edge_threshold,
        "min_odds": min_odds,
        "max_odds": max_odds,
        "avg_edge": avg_edge,
        "avg_odds": avg_odds,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }

    _print_result("バックテスト結果 (バリューベット方式)", result)
    return result


def _print_result(title: str, result: dict):
    """結果を整形して表示する"""
    print("=" * 55)
    print(f"  {title}")
    print("=" * 55)
    print(f"  テスト期間     : {result.get('test_period', 'N/A')}")
    print(f"  賭け数         : {result['total_bets']}件")
    print(f"  的中数         : {result['total_hits']}件")
    print(f"  的中率         : {result['hit_rate']:.1f}%")
    print(f"  概算回収率     : {result['roi']:.1f}%")
    print("-" * 55)
    if "ev_threshold" in result:
        print(f"  EV閾値         : {result['ev_threshold']}")
        print(f"  オッズ範囲     : {result['min_odds']} ~ {result['max_odds']}")
        print(f"  平均期待値     : {result['avg_expected_value']:.3f}")
        print(f"  平均オッズ     : {result['avg_odds']:.1f}")
        print(f"  平均AI予測確率 : {result['avg_pred_prob']:.3f}")
    elif "edge_threshold" in result:
        print(f"  エッジ閾値     : {result['edge_threshold']}")
        print(f"  オッズ範囲     : {result['min_odds']} ~ {result['max_odds']}")
        print(f"  平均エッジ     : {result['avg_edge']:.3f}")
        print(f"  平均オッズ     : {result['avg_odds']:.1f}")
    elif "threshold" in result:
        print(f"  確率閾値       : {result['threshold']}")
    print("=" * 55)
