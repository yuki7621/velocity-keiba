"""3連複 BOX 買い目の推奨判定ロジック

バックテスト (run_tune_sanrenpuku.py) で見つけた最適フィルタ:
  - head_count >= 13 (多頭数)
  - 1番人気 in AI Top3 (市場との整合)
  - top3_prob >= 0.20 (AI自信度)

これら3条件をすべて満たすレースで 3連複 BOX 1点を推奨する。
実績ROI: 103.1% (前半102.2% / 後半103.9%) — 前後半ともプラス
"""

from dataclasses import dataclass

import pandas as pd


# 閾値（チューニング結果）
MIN_HEAD_COUNT = 13
MIN_TOP3_PROB = 0.20

# 表示用のメタ情報
STRATEGY_NAME = "3連複BOX (上位3頭)"
EXPECTED_ROI = 103.1  # バックテスト実績 (%)
EXPECTED_HIT_RATE = 8.2  # (%)


@dataclass
class SanrenpukuRecommendation:
    """3連複BOX推奨の評価結果"""
    race_id: str
    is_recommended: bool
    reasons: list  # 条件を満たさなかった理由（空なら全て通過）

    # 指標
    head_count: int
    top1_prob: float
    top2_prob: float
    top3_prob: float
    fav_post: int | None  # 1番人気馬番
    fav_in_top3: bool
    top3_posts: list  # AI Top3 の馬番


def evaluate_race(race_df: pd.DataFrame) -> SanrenpukuRecommendation:
    """単一レースの出馬データから 3連複BOX 推奨可否を評価する

    Args:
        race_df: 1レース分のデータ。pred_prob / post_number / odds 列を持つこと

    Returns:
        SanrenpukuRecommendation
    """
    race_id = str(race_df["race_id"].iloc[0]) if "race_id" in race_df.columns else ""
    head_count = len(race_df)

    # 上位3頭
    sorted_df = race_df.sort_values("pred_prob", ascending=False)
    top3 = sorted_df.head(3)
    top3_posts = [int(x) for x in top3["post_number"].astype(int).tolist()]

    probs = top3["pred_prob"].tolist()
    top1_prob = float(probs[0]) if len(probs) >= 1 else 0.0
    top2_prob = float(probs[1]) if len(probs) >= 2 else 0.0
    top3_prob = float(probs[2]) if len(probs) >= 3 else 0.0

    # 1番人気（最小オッズの馬）
    fav_post: int | None = None
    if "odds" in race_df.columns:
        valid = race_df[race_df["odds"].notna() & (race_df["odds"] > 0)]
        if len(valid) > 0:
            fav_post = int(valid.loc[valid["odds"].idxmin(), "post_number"])
    fav_in_top3 = (fav_post is not None) and (fav_post in top3_posts)

    # フィルタ判定
    reasons = []
    if head_count < MIN_HEAD_COUNT:
        reasons.append(f"頭数不足 ({head_count}頭 < {MIN_HEAD_COUNT})")
    if top3_prob < MIN_TOP3_PROB:
        reasons.append(f"AI自信不足 (3位確率 {top3_prob:.1%} < {MIN_TOP3_PROB:.0%})")
    if not fav_in_top3:
        if fav_post is None:
            reasons.append("オッズ未取得 (1番人気判定不可)")
        else:
            reasons.append(f"1番人気 ({fav_post}番) が AI Top3 に不在")

    is_recommended = len(reasons) == 0

    return SanrenpukuRecommendation(
        race_id=race_id,
        is_recommended=is_recommended,
        reasons=reasons,
        head_count=head_count,
        top1_prob=top1_prob,
        top2_prob=top2_prob,
        top3_prob=top3_prob,
        fav_post=fav_post,
        fav_in_top3=fav_in_top3,
        top3_posts=top3_posts,
    )


def evaluate_all_races(df: pd.DataFrame) -> list[SanrenpukuRecommendation]:
    """複数レースをまとめて評価"""
    results = []
    for race_id, g in df.groupby("race_id"):
        results.append(evaluate_race(g))
    return results
