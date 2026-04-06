"""DB状況ページ"""

import sqlite3
import streamlit as st
import pandas as pd
from config.settings import DB_PATH


def render():
    st.header("ℹ️ データベース状況")

    if not DB_PATH.exists():
        st.warning("データベースが存在しません。「データ更新」ページからスクレイピングを実行してください。")
        return

    conn = sqlite3.connect(DB_PATH)

    # 基本統計
    col1, col2, col3, col4 = st.columns(4)

    race_count = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    result_count = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    horse_count = conn.execute("SELECT COUNT(*) FROM horses").fetchone()[0]
    jockey_count = conn.execute("SELECT COUNT(*) FROM jockeys").fetchone()[0]

    col1.metric("レース数", f"{race_count:,}")
    col2.metric("出走データ数", f"{result_count:,}")
    col3.metric("馬数", f"{horse_count:,}")
    col4.metric("騎手数", f"{jockey_count:,}")

    # 期間
    date_range = conn.execute("SELECT MIN(date), MAX(date) FROM races").fetchone()
    st.info(f"データ期間: **{date_range[0]}** 〜 **{date_range[1]}**")

    # 年別レース数
    st.subheader("年別レース数")
    yearly = pd.read_sql_query(
        "SELECT substr(date, 1, 4) as year, COUNT(*) as count FROM races GROUP BY year ORDER BY year",
        conn,
    )
    st.bar_chart(yearly.set_index("year"))

    # 競馬場別レース数
    st.subheader("競馬場別レース数")
    venue_counts = pd.read_sql_query(
        "SELECT venue, COUNT(*) as count FROM races WHERE venue IS NOT NULL GROUP BY venue ORDER BY count DESC",
        conn,
    )
    st.dataframe(venue_counts, use_container_width=True)

    # 最新データ
    st.subheader("直近のレース（最新10件）")
    recent = pd.read_sql_query(
        "SELECT date, venue, title, surface, distance, condition FROM races ORDER BY date DESC LIMIT 10",
        conn,
    )
    st.dataframe(recent, use_container_width=True)

    conn.close()
