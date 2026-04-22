"""v5/v6/v7 × 複勝/ワイド/馬連/3連複 の総合比較スクリプト

各モデルを読み込み、同じテストデータに対して全券種のバックテストを走らせ、
ROI・的中率の比較表を出力する。
"""

import pandas as pd

from src.features.build_features import build_all_features
from src.model.train import load_model
from src.evaluation.backtest import run_value_bet_backtest
from src.evaluation.multi_bet_backtest import (
    run_wide_backtest,
    run_umaren_backtest,
    run_sanrenpuku_backtest,
    run_sanrenpuku_formation,
    run_wide_axis,
)


MODEL_NAMES = ["lightgbm_v5", "lightgbm_v6", "lightgbm_v7"]


def _row(model_name, strategy, result):
    if not result:
        return {
            "model": model_name, "strategy": strategy,
            "bets": 0, "hits": 0, "hit_rate": 0.0, "roi": 0.0,
        }
    return {
        "model": model_name,
        "strategy": strategy,
        "bets": result.get("total_bets", 0),
        "hits": result.get("total_hits", 0),
        "hit_rate": result.get("hit_rate", 0.0),
        "roi": result.get("roi", 0.0),
    }


def main():
    print("=" * 60)
    print("  モデル×券種 総合比較")
    print("=" * 60)

    df = build_all_features()

    rows = []
    for name in MODEL_NAMES:
        try:
            model = load_model(name)
        except FileNotFoundError:
            print(f"[skip] {name} が見つかりません")
            continue

        print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")

        # 複勝 (バリューベット edge=0.15)
        r = run_value_bet_backtest(model, df, edge_threshold=0.15, min_odds=3.0, max_odds=100.0)
        rows.append({
            "model": name, "strategy": "複勝 edge0.15",
            "bets": r.get("total_bets", 0) if r else 0,
            "hits": r.get("total_hits", 0) if r else 0,
            "hit_rate": r.get("hit_rate", 0.0) if r else 0.0,
            "roi": r.get("roi", 0.0) if r else 0.0,
        })

        # ワイド
        rows.append(_row(name, "ワイド 上位2頭", run_wide_backtest(model, df, n_pick=2)))
        rows.append(_row(name, "ワイド 上位3頭BOX", run_wide_backtest(model, df, n_pick=3)))
        rows.append(_row(name, "ワイド 軸1×相手5", run_wide_axis(model, df, n_body=5)))

        # 馬連
        rows.append(_row(name, "馬連 上位2頭", run_umaren_backtest(model, df, n_pick=2)))
        rows.append(_row(name, "馬連 上位3頭BOX", run_umaren_backtest(model, df, n_pick=3)))

        # 3連複
        rows.append(_row(name, "3連複 上位3頭BOX", run_sanrenpuku_backtest(model, df, n_pick=3)))
        rows.append(_row(name, "3連複 上位4頭BOX", run_sanrenpuku_backtest(model, df, n_pick=4)))
        rows.append(_row(name, "3連複 軸1×相手5", run_sanrenpuku_formation(model, df, n_head=1, n_body=5)))

    # 集計表
    result_df = pd.DataFrame(rows)
    print("\n\n" + "=" * 80)
    print("  総合比較表")
    print("=" * 80)
    # ピボット: 戦略 x モデル -> ROI
    pivot_roi = result_df.pivot(index="strategy", columns="model", values="roi")
    pivot_hit = result_df.pivot(index="strategy", columns="model", values="hit_rate")
    print("\n--- ROI (%) ---")
    print(pivot_roi.to_string(float_format=lambda x: f"{x:6.1f}"))
    print("\n--- 的中率 (%) ---")
    print(pivot_hit.to_string(float_format=lambda x: f"{x:6.1f}"))

    # CSV 保存
    out_path = "model_comparison.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n→ 詳細を {out_path} に保存しました")


if __name__ == "__main__":
    main()
