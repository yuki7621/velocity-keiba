"""複数頭券種（ワイド・馬連・3連複・馬単・3連単）のバックテスト

AI予測確率から上位N頭を選び、固定戦略で購入した場合のROIを計算する。
払戻は payouts テーブル（実払戻）を使用するため、実運用と同じROIが得られる。
"""

import sqlite3
from itertools import combinations

import pandas as pd

from config.settings import DB_PATH
from src.model.train import prepare_dataset, get_available_features


# ============================================================
# 払戻データローダ
# ============================================================

def _load_combo_payouts(
    race_ids: list,
    bet_type: str,
    db_path=DB_PATH,
) -> dict:
    """対象レースの払戻を辞書で返す

    Returns:
        {race_id: {combo: payout_per_100yen, ...}, ...}
        combo は "3-5" / "3-5-7" 等（保存されている文字列そのまま）
    """
    if not race_ids:
        return {}
    conn = sqlite3.connect(db_path)
    # IN 句にまとめると高速
    placeholders = ",".join(["?"] * len(race_ids))
    query = f"""
        SELECT race_id, horse_numbers, payout
        FROM payouts
        WHERE bet_type = ?
          AND race_id IN ({placeholders})
    """
    cur = conn.cursor()
    cur.execute(query, [bet_type] + list(race_ids))
    result = {}
    for race_id, combo, payout in cur.fetchall():
        if combo is None:
            continue
        result.setdefault(race_id, {})[combo] = payout
    conn.close()
    return result


def _sorted_combo(horses) -> str:
    """整数の馬番リストを昇順結合した文字列にする (例: [5,3] → '3-5')"""
    return "-".join(str(h) for h in sorted(horses))


def _ordered_combo(horses) -> str:
    """順序を保持した文字列 (例: [5,3] → '5-3')"""
    return "-".join(str(h) for h in horses)


# ============================================================
# 予測スコアの準備
# ============================================================

def _prepare_test_with_preds(model, df: pd.DataFrame, test_frac: float = 0.2):
    """テスト用データ + 予測確率 を返す"""
    df = prepare_dataset(df)
    features = get_available_features(df)
    split_idx = int(len(df) * (1 - test_frac))
    test_df = df.iloc[split_idx:].copy()
    X = test_df[features]
    test_df["pred_prob"] = model.predict_proba(X)[:, 1]
    return test_df


# ============================================================
# 各券種のバックテスト
# ============================================================

def run_wide_backtest(
    model,
    df: pd.DataFrame,
    n_pick: int = 2,
    db_path=DB_PATH,
) -> dict:
    """ワイド: 予測上位 n_pick 頭の全組合せをボックス購入

    Args:
        n_pick: 選ぶ頭数 (2=1点買い、3=3点買い)
    """
    test_df = _prepare_test_with_preds(model, df)
    race_ids = test_df["race_id"].unique().tolist()
    payouts_map = _load_combo_payouts(race_ids, "wide", db_path)

    total_bets = 0
    total_payout = 0.0
    total_hits = 0
    total_races = 0

    for race_id, race_group in test_df.groupby("race_id"):
        top_horses = race_group.nlargest(n_pick, "pred_prob")
        if len(top_horses) < 2:
            continue
        total_races += 1
        horse_numbers = top_horses["post_number"].astype(int).tolist()

        combos = list(combinations(horse_numbers, 2))
        race_payouts = payouts_map.get(race_id, {})

        for combo in combos:
            total_bets += 1
            combo_key = _sorted_combo(combo)
            pay = race_payouts.get(combo_key, 0)
            if pay > 0:
                total_payout += pay / 100.0  # 100円単位 → 倍率
                total_hits += 1

    if total_bets == 0:
        return {}

    hit_rate = total_hits / total_bets * 100
    roi = total_payout / total_bets * 100

    result = {
        "bet_type": "wide",
        "strategy": f"上位{n_pick}頭ボックス",
        "total_races": total_races,
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": hit_rate,
        "roi": roi,
        "avg_payout_when_hit": (total_payout / total_hits * 100) if total_hits else 0,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }
    _print(result)
    return result


def run_umaren_backtest(
    model,
    df: pd.DataFrame,
    n_pick: int = 2,
    db_path=DB_PATH,
) -> dict:
    """馬連: 予測上位 n_pick 頭のボックス購入"""
    test_df = _prepare_test_with_preds(model, df)
    race_ids = test_df["race_id"].unique().tolist()
    payouts_map = _load_combo_payouts(race_ids, "umaren", db_path)

    total_bets = 0
    total_payout = 0.0
    total_hits = 0
    total_races = 0

    for race_id, race_group in test_df.groupby("race_id"):
        top_horses = race_group.nlargest(n_pick, "pred_prob")
        if len(top_horses) < 2:
            continue
        total_races += 1
        horse_numbers = top_horses["post_number"].astype(int).tolist()
        combos = list(combinations(horse_numbers, 2))
        race_payouts = payouts_map.get(race_id, {})

        for combo in combos:
            total_bets += 1
            combo_key = _sorted_combo(combo)
            pay = race_payouts.get(combo_key, 0)
            if pay > 0:
                total_payout += pay / 100.0
                total_hits += 1

    if total_bets == 0:
        return {}

    result = {
        "bet_type": "umaren",
        "strategy": f"上位{n_pick}頭ボックス",
        "total_races": total_races,
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": total_hits / total_bets * 100,
        "roi": total_payout / total_bets * 100,
        "avg_payout_when_hit": (total_payout / total_hits * 100) if total_hits else 0,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }
    _print(result)
    return result


def run_sanrenpuku_backtest(
    model,
    df: pd.DataFrame,
    n_pick: int = 3,
    db_path=DB_PATH,
) -> dict:
    """3連複: 予測上位 n_pick 頭のボックス購入"""
    test_df = _prepare_test_with_preds(model, df)
    race_ids = test_df["race_id"].unique().tolist()
    payouts_map = _load_combo_payouts(race_ids, "sanrenpuku", db_path)

    total_bets = 0
    total_payout = 0.0
    total_hits = 0
    total_races = 0

    for race_id, race_group in test_df.groupby("race_id"):
        top_horses = race_group.nlargest(n_pick, "pred_prob")
        if len(top_horses) < 3:
            continue
        total_races += 1
        horse_numbers = top_horses["post_number"].astype(int).tolist()
        combos = list(combinations(horse_numbers, 3))
        race_payouts = payouts_map.get(race_id, {})

        for combo in combos:
            total_bets += 1
            combo_key = _sorted_combo(combo)
            pay = race_payouts.get(combo_key, 0)
            if pay > 0:
                total_payout += pay / 100.0
                total_hits += 1

    if total_bets == 0:
        return {}

    result = {
        "bet_type": "sanrenpuku",
        "strategy": f"上位{n_pick}頭ボックス",
        "total_races": total_races,
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": total_hits / total_bets * 100,
        "roi": total_payout / total_bets * 100,
        "avg_payout_when_hit": (total_payout / total_hits * 100) if total_hits else 0,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }
    _print(result)
    return result


def run_sanrenpuku_formation(
    model,
    df: pd.DataFrame,
    n_head: int = 1,
    n_body: int = 5,
    db_path=DB_PATH,
) -> dict:
    """3連複フォーメーション: 軸 n_head 頭 × 相手 n_body 頭

    軸=1頭、相手=上位5頭 → 相手から2頭選ぶ組合せ × 軸1頭 = 10点
    軸馬が top3 にいて、かつ相手2頭も top3 にいれば的中
    """
    test_df = _prepare_test_with_preds(model, df)
    race_ids = test_df["race_id"].unique().tolist()
    payouts_map = _load_combo_payouts(race_ids, "sanrenpuku", db_path)

    total_bets = 0
    total_payout = 0.0
    total_hits = 0
    total_races = 0

    for race_id, race_group in test_df.groupby("race_id"):
        ranked = race_group.nlargest(n_head + n_body, "pred_prob")
        if len(ranked) < 3:
            continue
        total_races += 1
        head_horses = ranked.head(n_head)["post_number"].astype(int).tolist()
        body_horses = ranked.iloc[n_head: n_head + n_body]["post_number"].astype(int).tolist()
        race_payouts = payouts_map.get(race_id, {})

        for h in head_horses:
            for b1, b2 in combinations(body_horses, 2):
                total_bets += 1
                combo_key = _sorted_combo([h, b1, b2])
                pay = race_payouts.get(combo_key, 0)
                if pay > 0:
                    total_payout += pay / 100.0
                    total_hits += 1

    if total_bets == 0:
        return {}

    result = {
        "bet_type": "sanrenpuku",
        "strategy": f"軸{n_head}×相手{n_body} フォーメーション",
        "total_races": total_races,
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": total_hits / total_bets * 100,
        "roi": total_payout / total_bets * 100,
        "avg_payout_when_hit": (total_payout / total_hits * 100) if total_hits else 0,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }
    _print(result)
    return result


def run_wide_axis(
    model,
    df: pd.DataFrame,
    n_body: int = 5,
    db_path=DB_PATH,
) -> dict:
    """ワイド軸流し: 1番人気AI馬 × 相手 n_body 頭"""
    test_df = _prepare_test_with_preds(model, df)
    race_ids = test_df["race_id"].unique().tolist()
    payouts_map = _load_combo_payouts(race_ids, "wide", db_path)

    total_bets = 0
    total_payout = 0.0
    total_hits = 0
    total_races = 0

    for race_id, race_group in test_df.groupby("race_id"):
        ranked = race_group.nlargest(1 + n_body, "pred_prob")
        if len(ranked) < 2:
            continue
        total_races += 1
        head = int(ranked.iloc[0]["post_number"])
        body_horses = ranked.iloc[1:1 + n_body]["post_number"].astype(int).tolist()
        race_payouts = payouts_map.get(race_id, {})

        for b in body_horses:
            total_bets += 1
            combo_key = _sorted_combo([head, b])
            pay = race_payouts.get(combo_key, 0)
            if pay > 0:
                total_payout += pay / 100.0
                total_hits += 1

    if total_bets == 0:
        return {}

    result = {
        "bet_type": "wide",
        "strategy": f"軸1×相手{n_body} 流し",
        "total_races": total_races,
        "total_bets": total_bets,
        "total_hits": total_hits,
        "hit_rate": total_hits / total_bets * 100,
        "roi": total_payout / total_bets * 100,
        "avg_payout_when_hit": (total_payout / total_hits * 100) if total_hits else 0,
        "test_period": f'{test_df["date"].min()} ~ {test_df["date"].max()}',
    }
    _print(result)
    return result


# ============================================================
# 出力
# ============================================================

def _print(r: dict):
    print("-" * 60)
    print(f"  券種: {r['bet_type']:<12s} 戦略: {r['strategy']}")
    print(f"  テスト期間: {r['test_period']}")
    print(f"  レース数: {r['total_races']}")
    print(f"  賭け点数: {r['total_bets']}   的中: {r['total_hits']}   的中率: {r['hit_rate']:.1f}%")
    print(f"  ROI: {r['roi']:.1f}%   平均的中配当: {r['avg_payout_when_hit']:.1f}%")
