"""バックテストページ"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.features.build_features import build_all_features
from src.model.train import load_model, FEATURE_COLUMNS, get_available_features, prepare_dataset


@st.cache_data(ttl=3600, show_spinner="特徴量を構築中...")
def _build_features():
    return build_all_features()


@st.cache_resource
def _load_model(name: str):
    return load_model(name)


def render():
    st.header("🧪 バックテスト")

    # ── 設定 ──
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox("モデル", ["lightgbm_v6", "lightgbm_v5", "lightgbm_v4", "lightgbm_v3", "lightgbm_v2", "lightgbm_v1"])
        strategy = st.selectbox("戦略", ["バリューベット（エッジ）", "期待値（EV）"])

    with col2:
        min_odds = st.slider("最低オッズ", 1.0, 20.0, 3.0, 0.5)
        max_odds = st.slider("最大オッズ", 20.0, 200.0, 100.0, 10.0)

    if strategy == "バリューベット（エッジ）":
        thresholds = st.multiselect(
            "エッジ閾値",
            [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
            default=[0.05, 0.10, 0.15, 0.20],
        )
    else:
        thresholds = st.multiselect(
            "EV閾値",
            [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
            default=[0.6, 0.8, 1.0, 1.5],
        )

    if st.button("バックテスト実行", type="primary"):
        _run_backtest(model_name, strategy, thresholds, min_odds, max_odds)


def _run_backtest(model_name, strategy, thresholds, min_odds, max_odds):
    """バックテストを実行して結果を表示する"""
    try:
        model = _load_model(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。")
        return

    df = _build_features()
    df = prepare_dataset(df)
    features = get_available_features(df)

    # 最新20%をテスト
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    X_test = test_df[features]
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]

    # オッズフィルタ
    test_df = test_df[test_df["odds"].notna()].copy()
    test_df = test_df[(test_df["odds"] >= min_odds) & (test_df["odds"] <= max_odds)].copy()

    # 複勝オッズ: 実データがあれば使用、なければ概算
    if "fukusho_odds_actual" in test_df.columns and test_df["fukusho_odds_actual"].notna().any():
        test_df["fukusho_odds"] = test_df["fukusho_odds_actual"]
    else:
        test_df["fukusho_odds"] = np.where(
            test_df["finish_position"] <= 3,
            (test_df["odds"] * 0.3).clip(lower=1.1),
            0,
        )
    test_df["is_hit"] = (test_df["finish_position"] <= 3).astype(int)
    test_df["market_prob"] = (3.0 / test_df["odds"]).clip(upper=1.0)
    test_df["edge"] = test_df["pred_prob"] - test_df["market_prob"]
    # EV計算は概算オッズを使用（レース前に実複勝オッズは不明のため）
    test_df["ev"] = test_df["pred_prob"] * (test_df["odds"] * 0.3).clip(lower=1.1)

    period = f"{test_df['date'].min().strftime('%Y-%m-%d')} 〜 {test_df['date'].max().strftime('%Y-%m-%d')}"
    st.info(f"テスト期間: **{period}**  |  対象オッズ: {min_odds} 〜 {max_odds}")

    # ── 各閾値でバックテスト ──
    results = []
    for th in sorted(thresholds):
        if strategy == "バリューベット（エッジ）":
            bets = test_df[test_df["edge"] >= th]
        else:
            bets = test_df[test_df["ev"] >= th]

        if len(bets) == 0:
            continue

        hits = bets["is_hit"].sum()
        total = len(bets)
        hit_rate = hits / total * 100
        payout = (bets["is_hit"] * bets["fukusho_odds"]).sum()
        roi = payout / total * 100

        results.append({
            "閾値": th,
            "賭け数": total,
            "的中数": hits,
            "的中率": hit_rate,
            "回収率": roi,
        })

    if not results:
        st.warning("条件に合う賭けがありませんでした。閾値を調整してください。")
        return

    result_df = pd.DataFrame(results)

    # ── 結果テーブル ──
    st.subheader("📊 結果一覧")
    display_df = result_df.copy()
    display_df["的中率"] = display_df["的中率"].apply(lambda x: f"{x:.1f}%")
    display_df["回収率"] = display_df["回収率"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── 回収率グラフ ──
    st.subheader("📈 閾値 vs 回収率")
    fig = go.Figure()
    fig.add_scatter(
        x=result_df["閾値"],
        y=result_df["回収率"],
        mode="lines+markers+text",
        text=result_df["回収率"].apply(lambda x: f"{x:.0f}%"),
        textposition="top center",
        line=dict(width=3, color="#FF6B6B"),
        marker=dict(size=10),
    )
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="損益分岐 (100%)")
    fig.update_layout(
        xaxis_title="閾値",
        yaxis_title="回収率 (%)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 賭け数グラフ ──
    st.subheader("📉 閾値 vs 賭け数")
    fig2 = go.Figure()
    fig2.add_bar(
        x=result_df["閾値"].astype(str),
        y=result_df["賭け数"],
        marker_color="#4ECDC4",
        text=result_df["賭け数"],
        textposition="auto",
    )
    fig2.update_layout(
        xaxis_title="閾値",
        yaxis_title="賭け数",
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 月別回収率（最良閾値） ──
    best_th = result_df.loc[result_df["回収率"].idxmax(), "閾値"]
    st.subheader(f"📅 月別回収率（閾値: {best_th}）")

    if strategy == "バリューベット（エッジ）":
        best_bets = test_df[test_df["edge"] >= best_th].copy()
    else:
        best_bets = test_df[test_df["ev"] >= best_th].copy()

    best_bets["month"] = best_bets["date"].dt.to_period("M").astype(str)
    monthly = best_bets.groupby("month").apply(
        lambda g: pd.Series({
            "賭け数": len(g),
            "的中数": g["is_hit"].sum(),
            "回収率": (g["is_hit"] * g["fukusho_odds"]).sum() / len(g) * 100 if len(g) > 0 else 0,
        })
    ).reset_index()

    fig3 = go.Figure()
    fig3.add_bar(
        x=monthly["month"],
        y=monthly["回収率"],
        marker_color=["#FF6B6B" if r < 100 else "#4ECDC4" for r in monthly["回収率"]],
        text=monthly["回収率"].apply(lambda x: f"{x:.0f}%"),
        textposition="auto",
    )
    fig3.add_hline(y=100, line_dash="dash", line_color="gray")
    fig3.update_layout(xaxis_title="月", yaxis_title="回収率 (%)", height=350)
    st.plotly_chart(fig3, use_container_width=True)
