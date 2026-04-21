"""
日曜日の予測スクリプト

使い方:
  1. 土曜の結果データを取得（run_scraper_today.py を土曜夜に実行）
  2. 日曜朝にこのスクリプトを実行
  3. 出走表を元に予測結果を表示

  python run_predict_sunday.py              # 直近の日曜を自動判定
  python run_predict_sunday.py 2025-04-06   # 日付指定
"""

import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config.settings import DB_PATH, SCRAPE_INTERVAL_SEC
from src.db.schema import create_tables
from src.features.track_bias import get_track_bias_for_date, analyze_track_bias, get_race_day_results
from src.model.train import load_model, FEATURE_COLUMNS, get_available_features
from src.features.build_features import build_all_features


def get_target_date() -> str:
    """予測対象日を取得する"""
    if len(sys.argv) > 1:
        return sys.argv[1]

    # 引数なしの場合: 直近の日曜日
    today = date.today()
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0 and today.weekday() == 6:
        return str(today)
    return str(today + timedelta(days=days_until_sunday))


def show_track_bias_report(target_date: str, venues: list[str]):
    """土曜の馬場傾向レポートを表示する"""
    print("=" * 60)
    print(f"  馬場傾向レポート（{target_date} の前日データ）")
    print("=" * 60)

    for venue in venues:
        bias = get_track_bias_for_date(target_date, venue)

        if bias["n_races"] == 0:
            continue

        print(f"\n  【{venue}】 分析レース数: {bias['n_races']}R")
        print(f"  ─────────────────────────────────")

        # 内外バイアス
        gate_label = "内枠有利 ◎" if bias["gate_bias"] > 0.05 else \
                     "外枠有利 ◎" if bias["gate_bias"] < -0.05 else "フラット"
        print(f"  枠順: {gate_label} "
              f"(内枠複勝率 {bias['inner_top3_rate']:.1%} / 外枠 {bias['outer_top3_rate']:.1%})")

        # 脚質バイアス
        pace_label = "先行有利 ◎" if bias["pace_bias"] > 0.05 else \
                     "差し有利 ◎" if bias["pace_bias"] < -0.05 else "フラット"
        print(f"  脚質: {pace_label} "
              f"(先行複勝率 {bias['front_top3_rate']:.1%} / 差し {bias['closer_top3_rate']:.1%})")

        # 時計
        time_label = "高速馬場" if bias["time_bias"] > 0.1 else \
                     "タフな馬場" if bias["time_bias"] < -0.1 else "標準"
        print(f"  時計: {time_label} (指数: {bias['time_bias']:+.3f})")

        # 上がり
        last3f_label = "末脚勝負" if bias["last3f_bias"] > 0.5 else \
                       "前残り傾向" if bias["last3f_bias"] < 0.3 else "標準"
        print(f"  上がり: {last3f_label} (3着内外の上がり差: {bias['last3f_bias']:.2f}秒)")


def predict_and_display(target_date: str):
    """予測結果を表示する"""
    print("\n" + "=" * 60)
    print(f"  予測結果（{target_date}）")
    print("=" * 60)

    # 全特徴量を構築
    print("\n特徴量を構築中（数分かかります）...")
    df = build_all_features()

    # モデル読み込み（新しいバージョンから順にフォールバック）
    model = None
    for model_name, desc in [
        ("lightgbm_v7", "Rankerアンサンブル + 血統 + 脚質/休養"),
        ("lightgbm_v6", "Rankerアンサンブル + 血統特徴量"),
        ("lightgbm_v5", "Rankerアンサンブル"),
        ("lightgbm_v4", "v4"),
        ("lightgbm_v3", "v3"),
    ]:
        try:
            model = load_model(model_name)
            print(f"モデル {model_name} を読み込みました（{desc}）")
            break
        except FileNotFoundError:
            continue
    if model is None:
        print("エラー: 学習済みモデルが見つかりません。先にrun_train_v6.pyを実行してください。")
        return

    features = get_available_features(df)

    # 対象日のデータを抽出
    target_df = df[df["date"] == target_date].copy()

    if len(target_df) == 0:
        print(f"\n{target_date} のレースデータがDBにありません。")
        print("先にrun_scraper_today.pyで出走表を取得してください。")
        return

    X = target_df[features]
    target_df["pred_prob"] = model.predict_proba(X)[:, 1]

    # 期待値を計算
    target_df["estimated_fukusho_odds"] = (target_df["odds"] * 0.3).clip(lower=1.1)
    target_df["expected_value"] = target_df["pred_prob"] * target_df["estimated_fukusho_odds"]

    # 市場の暗黙確率との差（エッジ）
    target_df["market_implied_prob"] = (3.0 / target_df["odds"]).clip(upper=1.0)
    target_df["edge"] = target_df["pred_prob"] - target_df["market_implied_prob"]

    # レースごとに結果を表示
    for race_id, race_df in target_df.groupby("race_id"):
        race_df = race_df.sort_values("pred_prob", ascending=False)

        venue = race_df["venue"].iloc[0]
        distance = race_df["distance"].iloc[0]
        surface = race_df["surface"].iloc[0]
        condition = race_df["condition"].iloc[0] if pd.notna(race_df["condition"].iloc[0]) else "?"
        race_num = str(race_id)[-2:]  # 末尾2桁 = レース番号

        print(f"\n{'─' * 60}")
        print(f"  {venue} {race_num}R  {surface}{distance}m  馬場:{condition}")
        print(f"{'─' * 60}")
        print(f"  {'馬番':>4} {'馬名':^14} {'AI確率':>6} {'オッズ':>6} {'期待値':>6} {'エッジ':>7}  推奨")
        print(f"  {'─' * 54}")

        for _, row in race_df.iterrows():
            horse_id = row["horse_id"]
            post = int(row["post_number"]) if pd.notna(row["post_number"]) else 0
            prob = row["pred_prob"]
            odds = row["odds"] if pd.notna(row["odds"]) else 0
            ev = row["expected_value"] if pd.notna(row["expected_value"]) else 0
            edge = row["edge"] if pd.notna(row["edge"]) else 0

            # 推奨マーク
            if edge >= 0.15 and ev >= 1.0:
                mark = "★★★"
            elif edge >= 0.10 and ev >= 0.8:
                mark = "★★"
            elif edge >= 0.05:
                mark = "★"
            else:
                mark = ""

            # 馬名はhorse_idで表示（名前はDBから取得）
            print(f"  {post:>4} {horse_id:^14} {prob:>6.1%} {odds:>6.1f} {ev:>6.2f} {edge:>+7.3f}  {mark}")

    # サマリー
    value_bets = target_df[
        (target_df["edge"] >= 0.10) & (target_df["expected_value"] >= 0.8)
    ].sort_values("expected_value", ascending=False)

    print(f"\n{'=' * 60}")
    print(f"  本日のバリューベット: {len(value_bets)}頭")
    print(f"{'=' * 60}")

    if len(value_bets) > 0:
        for _, row in value_bets.head(20).iterrows():
            race_num = str(row["race_id"])[-2:]
            venue = row["venue"]
            post = int(row["post_number"])
            print(f"  {venue}{race_num}R {post}番 "
                  f"AI={row['pred_prob']:.1%} "
                  f"オッズ={row['odds']:.1f} "
                  f"EV={row['expected_value']:.2f} "
                  f"エッジ={row['edge']:+.3f}")


def main():
    target_date = get_target_date()
    print(f"予測対象日: {target_date}")

    # 全10競馬場
    venues = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

    # 1. 馬場傾向レポート
    show_track_bias_report(target_date, venues)

    # 2. 予測
    predict_and_display(target_date)


if __name__ == "__main__":
    main()
