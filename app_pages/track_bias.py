"""馬場傾向ページ — 芝/ダート別に分析"""

import sqlite3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from config.settings import DB_PATH
from src.features.track_bias import (
    get_race_day_results,
    analyze_track_bias,
    get_previous_day,
)


def render():
    st.header("📈 馬場傾向分析")

    if not DB_PATH.exists():
        st.warning("データベースが存在しません。")
        return

    conn = sqlite3.connect(DB_PATH)

    # 競馬場と日付の選択
    col1, col2 = st.columns(2)
    with col1:
        venues = pd.read_sql_query(
            "SELECT DISTINCT venue FROM races WHERE venue IS NOT NULL ORDER BY venue",
            conn,
        )["venue"].tolist()
        venue = st.selectbox("競馬場", venues if venues else ["東京"])

    with col2:
        recent_dates = pd.read_sql_query(
            "SELECT DISTINCT date FROM races WHERE venue = ? ORDER BY date DESC LIMIT 30",
            conn, params=[venue],
        )["date"].tolist()

        if recent_dates:
            target_date = st.selectbox("分析する日付", recent_dates)
        else:
            target_date = st.date_input("分析する日付", value=date.today())
            target_date = str(target_date)

    conn.close()

    if st.button("分析実行", type="primary"):
        _show_bias_analysis(str(target_date), venue)


def _show_bias_analysis(target_date: str, venue: str):
    """馬場傾向を芝/ダート別に分析して表示する"""
    df = get_race_day_results(target_date, venue)

    if len(df) == 0:
        st.warning(f"{venue} {target_date} のレースデータがありません。")
        return

    # 芝/ダートのレース数をカウント
    surfaces = df["surface"].dropna().unique().tolist()
    turf_count = len(df[df["surface"] == "芝"]["race_id"].unique()) if "芝" in surfaces else 0
    dirt_count = len(df[df["surface"] == "ダート"]["race_id"].unique()) if "ダート" in surfaces else 0
    total_count = df["race_id"].nunique()

    st.subheader(f"{venue} {target_date}（全{total_count}R: 芝{turf_count}R / ダート{dirt_count}R）")

    # ── 芝/ダートのタブ ──
    tab_labels = []
    tab_surfaces = []

    if turf_count > 0:
        tab_labels.append(f"🌱 芝（{turf_count}R）")
        tab_surfaces.append("芝")
    if dirt_count > 0:
        tab_labels.append(f"🟤 ダート（{dirt_count}R）")
        tab_surfaces.append("ダート")

    # 障害があれば追加
    obstacle_count = total_count - turf_count - dirt_count
    if obstacle_count > 0:
        tab_labels.append(f"🚧 障害（{obstacle_count}R）")
        tab_surfaces.append("障害")

    if not tab_labels:
        st.warning("分析対象のレースがありません。")
        return

    tabs = st.tabs(tab_labels)

    for tab, surface in zip(tabs, tab_surfaces):
        with tab:
            bias = analyze_track_bias(df, surface=surface)
            _show_surface_bias(bias, df[df["surface"] == surface], surface)


def _show_surface_bias(bias: dict, surface_df: pd.DataFrame, surface: str):
    """特定馬場の傾向を表示"""

    n_races = bias["n_races"]
    if n_races == 0:
        st.info(f"{surface}のレースがありません。")
        return

    # ── メトリクス ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gate_label = "内枠有利" if bias["gate_bias"] > 0.05 else "外枠有利" if bias["gate_bias"] < -0.05 else "フラット"
        st.metric("枠順バイアス", gate_label, f"{bias['gate_bias']:+.3f}")

    with col2:
        pace_label = "先行有利" if bias["pace_bias"] > 0.05 else "差し有利" if bias["pace_bias"] < -0.05 else "フラット"
        st.metric("脚質バイアス", pace_label, f"{bias['pace_bias']:+.3f}")

    with col3:
        time_label = "高速" if bias["time_bias"] > 0.1 else "タフ" if bias["time_bias"] < -0.1 else "標準"
        st.metric("馬場スピード", time_label, f"{bias['time_bias']:+.3f}")

    with col4:
        last3f_label = "末脚勝負" if bias["last3f_bias"] > 0.5 else "前残り" if bias["last3f_bias"] < 0.3 else "標準"
        st.metric("上がり傾向", last3f_label, f"{bias['last3f_bias']:.2f}秒差")

    st.divider()

    # ── 内外バイアス詳細 ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🎯 枠番別複勝率")
        fig = go.Figure()
        fig.add_bar(
            x=["内枠 (1-4)", "外枠 (5-8)"],
            y=[bias["inner_top3_rate"] * 100, bias["outer_top3_rate"] * 100],
            marker_color=["#FF6B6B", "#4ECDC4"],
            text=[f"{bias['inner_top3_rate']:.1%}", f"{bias['outer_top3_rate']:.1%}"],
            textposition="auto",
        )
        fig.update_layout(yaxis_title="複勝率 (%)", height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🏃 脚質別複勝率")
        fig = go.Figure()
        fig.add_bar(
            x=["先行", "差し・追込"],
            y=[bias["front_top3_rate"] * 100, bias["closer_top3_rate"] * 100],
            marker_color=["#FFD93D", "#6BCB77"],
            text=[f"{bias['front_top3_rate']:.1%}", f"{bias['closer_top3_rate']:.1%}"],
            textposition="auto",
        )
        fig.update_layout(yaxis_title="複勝率 (%)", height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # ── レースごとの詳細 ──
    st.markdown("#### 📋 レース別結果")

    if len(surface_df) == 0:
        return

    surface_df = surface_df.copy()
    surface_df["is_top3"] = (surface_df["finish_position"] <= 3).astype(int)

    race_summary = surface_df.groupby("race_id").agg(
        頭数=("horse_id", "count"),
        距離=("distance", "first"),
        馬場状態=("condition", "first"),
        勝ちタイム=("finish_time_sec", "min"),
        上がり3F平均=("last_3f", "mean"),
    ).reset_index()

    race_summary["レース"] = race_summary["race_id"].apply(lambda x: f"{str(x)[-2:]}R")
    race_summary["距離"] = race_summary["距離"].apply(lambda x: f"{int(x)}m" if pd.notna(x) else "-")
    race_summary["勝ちタイム"] = race_summary["勝ちタイム"].apply(
        lambda x: f"{int(x // 60)}:{x % 60:05.2f}" if pd.notna(x) else "-"
    )
    race_summary["上がり3F平均"] = race_summary["上がり3F平均"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    )

    st.dataframe(
        race_summary[["レース", "距離", "頭数", "馬場状態", "勝ちタイム", "上がり3F平均"]],
        use_container_width=True,
        hide_index=True,
    )
