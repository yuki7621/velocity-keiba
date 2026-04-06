"""データ更新ページ"""

import time
import threading
import streamlit as st
from datetime import date, datetime

from config.settings import DB_PATH, SCRAPE_INTERVAL_SEC
from src.db.schema import create_tables
from src.scraper.race_list import get_race_ids_by_month
from src.scraper.race_result import scrape_race
from src.scraper.storage import save_race_data


def render():
    st.header("🔄 データ更新")

    # DB初期化
    if not DB_PATH.exists():
        if st.button("データベースを初期化"):
            create_tables()
            st.success("データベースを作成しました。")
            st.rerun()
        return

    st.markdown("netkeibaからレースデータをスクレイピングしてDBに保存します。")

    # ── 月単位の更新 ──
    st.subheader("📅 月を指定して更新")

    col1, col2 = st.columns(2)
    with col1:
        target_year = st.selectbox("年", list(range(2025, 2020, -1)), index=0)
    with col2:
        target_month = st.selectbox("月", list(range(1, 13)), index=max(0, date.today().month - 1))

    if st.button("この月のデータを取得", key="monthly"):
        _scrape_month(target_year, target_month)

    st.divider()

    # ── 今日のデータを更新 ──
    st.subheader("📌 今日のレース結果を取得")
    st.markdown("土曜夜に実行すると、当日のレース結果がDBに反映されます。")

    today_str = date.today().strftime("%Y-%m-%d")
    st.info(f"今日の日付: **{today_str}**")

    if st.button("今月のデータを更新", key="today"):
        _scrape_month(date.today().year, date.today().month)


def _scrape_month(year: int, month: int):
    """指定月のレースをスクレイピングする"""
    progress_bar = st.progress(0)
    status = st.empty()
    log_area = st.empty()

    status.text(f"{year}年{month}月のレースID一覧を取得中...")
    try:
        race_ids = get_race_ids_by_month(year, month)
    except Exception as e:
        st.error(f"レースID取得に失敗: {e}")
        return

    if not race_ids:
        st.warning(f"{year}年{month}月のレースが見つかりませんでした。")
        return

    total = len(race_ids)
    status.text(f"{total}レースを取得中...")

    success = 0
    fail = 0
    logs = []

    for i, race_id in enumerate(race_ids):
        try:
            data = scrape_race(race_id)
            if data:
                save_race_data(data)
                success += 1
                venue = data["race_info"].get("venue", "?")
                title = data["race_info"].get("title", "?")
                logs.append(f"✅ {race_id} {venue} {title}")
            else:
                fail += 1
                logs.append(f"❌ {race_id} データ取得失敗")
        except Exception as e:
            fail += 1
            logs.append(f"❌ {race_id} エラー: {e}")

        progress_bar.progress((i + 1) / total)
        status.text(f"進捗: {i + 1}/{total} (成功: {success}, 失敗: {fail})")

        # 最新5件のログを表示
        log_area.text("\n".join(logs[-5:]))

        time.sleep(SCRAPE_INTERVAL_SEC)

    progress_bar.progress(1.0)
    st.success(f"完了！ 成功: {success}件, 失敗: {fail}件")
