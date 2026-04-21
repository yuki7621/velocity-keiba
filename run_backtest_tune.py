"""バリューベット パラメータチューニング: min_odds=5.0 で edge を広く探索"""

import pickle
from pathlib import Path
from src.features.build_features import build_all_features
from src.evaluation.backtest import run_value_bet_backtest


def load(name):
    with open(Path("models") / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main():
    print("=" * 60)
    print("  バリューベット チューニング (実払戻ベース)")
    print("  min_odds=5.0 / max_odds=50.0 で edge を広く探索")
    print("=" * 60)

    print("\n[Step 1] 特徴量構築...")
    df = build_all_features()

    results = []

    for ver in ["v5", "v6"]:
        print("\n" + "#" * 60)
        print(f"#  {ver}")
        print("#" * 60)
        model = load(f"lightgbm_{ver}")

        for min_o in [5.0, 7.0, 10.0]:
            for max_o in [30.0, 50.0, 100.0]:
                for edge in [0.15, 0.20, 0.25, 0.30]:
                    print(f"\n--- [{ver}] min_odds={min_o} max_odds={max_o} edge={edge} ---")
                    r = run_value_bet_backtest(
                        model, df,
                        edge_threshold=edge,
                        min_odds=min_o,
                        max_odds=max_o,
                    )
                    if r:
                        results.append({
                            "ver": ver,
                            "min_odds": min_o,
                            "max_odds": max_o,
                            "edge": edge,
                            "bets": r["total_bets"],
                            "hit_rate": r["hit_rate"],
                            "roi": r["roi"],
                            "avg_odds": r["avg_odds"],
                        })

    # サマリー
    print("\n" + "=" * 90)
    print("  サマリー (ROI 降順 TOP20)")
    print("=" * 90)
    print(f"{'Ver':<4} {'min':>5} {'max':>6} {'edge':>5} {'bets':>6} {'hit%':>6} {'ROI%':>7} {'avgO':>6}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["roi"])[:20]:
        print(
            f"{r['ver']:<4} {r['min_odds']:>5.1f} {r['max_odds']:>6.1f} "
            f"{r['edge']:>5.2f} {r['bets']:>6} {r['hit_rate']:>5.1f}% "
            f"{r['roi']:>6.1f}% {r['avg_odds']:>6.1f}"
        )

    print("\n" + "=" * 90)
    print("  ROI 100%超え (利益が出る帯)")
    print("=" * 90)
    profitable = [r for r in results if r["roi"] >= 100.0]
    if profitable:
        print(f"{'Ver':<4} {'min':>5} {'max':>6} {'edge':>5} {'bets':>6} {'hit%':>6} {'ROI%':>7} {'avgO':>6}")
        print("-" * 90)
        for r in sorted(profitable, key=lambda x: -x["roi"]):
            print(
                f"{r['ver']:<4} {r['min_odds']:>5.1f} {r['max_odds']:>6.1f} "
                f"{r['edge']:>5.2f} {r['bets']:>6} {r['hit_rate']:>5.1f}% "
                f"{r['roi']:>6.1f}% {r['avg_odds']:>6.1f}"
            )
    else:
        print("  利益が出る帯はありませんでした。")


if __name__ == "__main__":
    main()
