"""予測ページ"""

import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

from config.settings import DB_PATH
from src.features.track_bias import get_track_bias_for_date
from src.features.build_features import build_all_features
from src.model.train import load_model, FEATURE_COLUMNS, get_available_features


@st.cache_data(ttl=3600, show_spinner="特徴量を構築中（初回は数分かかります）...")
def _build_features():
    """特徴量を構築する（キャッシュ付き）"""
    return build_all_features()


@st.cache_resource
def _load_model(name: str):
    """モデルを読み込む（キャッシュ付き）"""
    return load_model(name)


def render():
    st.header("📊 レース予測")

    if not DB_PATH.exists():
        st.warning("データベースが存在しません。「データ更新」ページから取得してください。")
        return

    conn = sqlite3.connect(DB_PATH)

    # ── 設定 ──
    col1, col2, col3 = st.columns(3)
    with col1:
        dates = pd.read_sql_query(
            "SELECT DISTINCT date FROM races ORDER BY date DESC LIMIT 60", conn
        )["date"].tolist()
        if dates:
            target_date = st.selectbox("予測する日付", dates)
        else:
            st.warning("データがありません。")
            conn.close()
            return

    with col2:
        # 選択した日の競馬場
        venues_on_date = pd.read_sql_query(
            "SELECT DISTINCT venue FROM races WHERE date = ? ORDER BY venue", conn,
            params=[target_date],
        )["venue"].tolist()
        venue_filter = st.multiselect("競馬場（空欄で全て）", venues_on_date, default=venues_on_date)

    with col3:
        model_name = st.selectbox("モデル", ["lightgbm_v3", "lightgbm_v2", "lightgbm_v1"])

    conn.close()

    if st.button("予測を実行", type="primary"):
        _run_prediction(target_date, venue_filter, model_name)


def _run_prediction(target_date: str, venues: list[str], model_name: str):
    """予測を実行して結果を表示する"""

    # モデル読み込み
    try:
        model = _load_model(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。先に学習を実行してください。")
        return

    # 特徴量構築
    df = _build_features()
    features = get_available_features(df)

    # 対象日のデータ
    target_df = df[df["date"] == target_date].copy()

    if venues:
        target_df = target_df[target_df["venue"].isin(venues)]

    if len(target_df) == 0:
        st.warning(f"{target_date} のデータがありません。")
        return

    # 予測
    X = target_df[features]
    target_df["pred_prob"] = model.predict_proba(X)[:, 1]

    # 期待値・エッジ
    target_df["fukusho_odds"] = target_df["odds"] * 0.3
    target_df["expected_value"] = target_df["pred_prob"] * target_df["fukusho_odds"]
    target_df["market_prob"] = (3.0 / target_df["odds"]).clip(upper=1.0)
    target_df["edge"] = target_df["pred_prob"] - target_df["market_prob"]

    # 馬名取得
    conn = sqlite3.connect(DB_PATH)
    horse_names = pd.read_sql_query("SELECT horse_id, name FROM horses", conn)
    conn.close()
    target_df = target_df.merge(horse_names, on="horse_id", how="left")

    # ── 馬場傾向サマリー ──
    st.subheader("📈 前日の馬場傾向")
    for venue in target_df["venue"].unique():
        bias = get_track_bias_for_date(target_date, venue)
        if bias["n_races"] > 0:
            cols = st.columns(5)
            cols[0].markdown(f"**{venue}**")
            gate_emoji = "🔴" if bias["gate_bias"] > 0.05 else "🔵" if bias["gate_bias"] < -0.05 else "⚪"
            cols[1].metric("枠順", f"{gate_emoji} {'内' if bias['gate_bias'] > 0.05 else '外' if bias['gate_bias'] < -0.05 else '—'}")
            pace_emoji = "🔴" if bias["pace_bias"] > 0.05 else "🔵" if bias["pace_bias"] < -0.05 else "⚪"
            cols[2].metric("脚質", f"{pace_emoji} {'先行' if bias['pace_bias'] > 0.05 else '差し' if bias['pace_bias'] < -0.05 else '—'}")
            cols[3].metric("時計", f"{'高速' if bias['time_bias'] > 0.1 else 'タフ' if bias['time_bias'] < -0.1 else '標準'}")
            cols[4].metric("上がり", f"{bias['last3f_bias']:.2f}秒差")

    st.divider()

    # ── バリューベットサマリー ──
    value_bets = target_df[
        (target_df["edge"] >= 0.10) & (target_df["expected_value"] >= 0.8)
    ].sort_values("expected_value", ascending=False)

    st.subheader(f"⭐ バリューベット: {len(value_bets)}頭")

    if len(value_bets) > 0:
        display_cols = {
            "venue": "競馬場",
            "race_id": "レースID",
            "post_number": "馬番",
            "name": "馬名",
            "pred_prob": "AI確率",
            "odds": "単勝ｵｯｽﾞ",
            "expected_value": "期待値",
            "edge": "エッジ",
        }
        vb_display = value_bets[list(display_cols.keys())].rename(columns=display_cols).copy()
        vb_display["AI確率"] = vb_display["AI確率"].apply(lambda x: f"{x:.1%}")
        vb_display["期待値"] = vb_display["期待値"].apply(lambda x: f"{x:.2f}")
        vb_display["エッジ"] = vb_display["エッジ"].apply(lambda x: f"{x:+.3f}")
        vb_display["レース"] = vb_display["レースID"].apply(lambda x: f"{str(x)[-2:]}R")
        st.dataframe(
            vb_display[["競馬場", "レース", "馬番", "馬名", "AI確率", "単勝ｵｯｽﾞ", "期待値", "エッジ"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("今回のレースにはバリューベット対象がありませんでした。")

    st.divider()

    # ── レースごとの詳細 ──
    st.subheader("📋 レース別予測")

    for race_id in sorted(target_df["race_id"].unique()):
        race_df = target_df[target_df["race_id"] == race_id].sort_values("pred_prob", ascending=False)

        venue = race_df["venue"].iloc[0]
        distance = int(race_df["distance"].iloc[0])
        surface = race_df["surface"].iloc[0]
        condition = race_df["condition"].iloc[0] if pd.notna(race_df["condition"].iloc[0]) else "?"
        race_num = str(race_id)[-2:]

        n_value = len(race_df[race_df["edge"] >= 0.10])
        value_tag = f" ⭐{n_value}" if n_value > 0 else ""

        with st.expander(f"**{venue} {race_num}R** — {surface}{distance}m ({condition}){value_tag}", expanded=(n_value > 0)):
            display = race_df[[
                "post_number", "name", "pred_prob", "odds",
                "expected_value", "edge", "finish_position",
            ]].copy()

            display.columns = ["馬番", "馬名", "AI確率", "ｵｯｽﾞ", "期待値", "エッジ", "着順"]

            # 推奨マーク
            def _mark(row):
                if row["エッジ"] >= 0.15 and row["期待値"] >= 1.0:
                    return "★★★"
                elif row["エッジ"] >= 0.10 and row["期待値"] >= 0.8:
                    return "★★"
                elif row["エッジ"] >= 0.05:
                    return "★"
                return ""

            display["推奨"] = display.apply(_mark, axis=1)

            display["AI確率"] = display["AI確率"].apply(lambda x: f"{x:.1%}")
            display["ｵｯｽﾞ"] = display["ｵｯｽﾞ"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
            display["期待値"] = display["期待値"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
            display["エッジ"] = display["エッジ"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
            display["馬番"] = display["馬番"].astype(int)
            display["着順"] = display["着順"].astype(int)

            st.dataframe(
                display[["馬番", "馬名", "AI確率", "ｵｯｽﾞ", "期待値", "エッジ", "推奨", "着順"]],
                use_container_width=True,
                hide_index=True,
            )
