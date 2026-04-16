"""データ更新ページ"""

import time
import streamlit as st
from datetime import date, datetime, timedelta

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

    tab_date, tab_month = st.tabs(["📌 日付指定で取得（推奨）", "📅 月単位で取得"])

    with tab_date:
        _render_date_update()

    with tab_month:
        _render_month_update()


# ── 日付指定で取得 ──

def _render_date_update():
    """日付を指定してレース結果を取得する（race.netkeiba.comからID取得）"""

    st.markdown(
        "race.netkeiba.com から指定日のレースIDを取得し、結果をDBに保存します。\n\n"
        "当日・前日のレース結果を素早く取得できます。"
    )

    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input(
            "取得する日付",
            value=date.today(),
            key="update_date",
        )
    with col2:
        # 複数日まとめて取得
        days_back = st.number_input(
            "何日前まで取得する",
            min_value=0, max_value=30, value=0,
            help="0 = 指定日のみ、1 = 指定日と前日、2 = 指定日から2日前まで",
            key="update_days_back",
        )

    if st.button("🔄 レース結果を取得", type="primary", key="btn_date_update"):
        dates = [target_date - timedelta(days=i) for i in range(days_back + 1)]
        _scrape_by_dates(dates)


def _scrape_by_dates(dates: list):
    """日付リストからレースIDを取得してスクレイピングする"""
    from src.scraper.race_card import get_upcoming_race_ids

    all_race_ids = []
    status = st.empty()

    for d in dates:
        date_str = d.strftime("%Y%m%d")
        status.info(f"📡 {d} のレースID一覧を取得中...")
        try:
            ids = get_upcoming_race_ids(date_str)
            all_race_ids.extend(ids)
            status.info(f"📡 {d}: {len(ids)} レース見つかりました")
        except Exception as e:
            st.warning(f"⚠️ {d} のレースID取得に失敗: {e}")
        time.sleep(SCRAPE_INTERVAL_SEC)

    all_race_ids = sorted(set(all_race_ids))

    if not all_race_ids:
        status.empty()
        st.warning("対象日にレースが見つかりませんでした。開催日を確認してください。")
        return

    status.info(f"📡 合計 {len(all_race_ids)} レースの結果を取得中...")
    _scrape_race_ids(all_race_ids)


# ── 月単位で取得 ──

def _render_month_update():
    """月を指定してレースデータを取得する（従来方式）"""

    st.markdown("db.netkeiba.com のレース検索から月単位で取得します。")

    col1, col2 = st.columns(2)
    with col1:
        current_year = date.today().year
        target_year = st.selectbox("年", list(range(current_year, 2020, -1)), index=0)
    with col2:
        target_month = st.selectbox("月", list(range(1, 13)), index=max(0, date.today().month - 1))

    if st.button("この月のデータを取得", key="monthly"):
        _scrape_month(target_year, target_month)


def _scrape_month(year: int, month: int):
    """指定月のレースをスクレイピングする"""
    status = st.empty()
    status.text(f"{year}年{month}月のレースID一覧を取得中...")

    try:
        race_ids = get_race_ids_by_month(year, month)
    except Exception as e:
        st.error(f"レースID取得に失敗: {e}")
        return

    if not race_ids:
        st.warning(f"{year}年{month}月のレースが見つかりませんでした。")
        return

    status.text(f"{len(race_ids)} レースを取得中...")
    _scrape_race_ids(race_ids)


# ── 共通スクレイピング処理 ──

def _scrape_race_ids(race_ids: list[str]):
    """レースIDリストからレース結果をスクレイピングしてDBに保存する"""
    total = len(race_ids)
    progress_bar = st.progress(0)
    status = st.empty()
    log_area = st.empty()

    success = 0
    fail = 0
    skip = 0
    logs = []

    for i, race_id in enumerate(race_ids):
        try:
            data = scrape_race(race_id)
            if data and data.get("results"):
                # dateが取得できなかった場合はスキップ
                if not data["race_info"].get("date"):
                    fail += 1
                    logs.append(f"❌ {race_id} 日付情報を取得できませんでした")
                    progress_bar.progress((i + 1) / total)
                    status.text(f"進捗: {i + 1}/{total} (成功: {success}, スキップ: {skip}, 失敗: {fail})")
                    log_area.text("\n".join(logs[-5:]))
                    time.sleep(SCRAPE_INTERVAL_SEC)
                    continue
                save_race_data(data)
                success += 1
                venue = data["race_info"].get("venue", "?")
                title = data["race_info"].get("title", "?")
                logs.append(f"✅ {race_id} {venue} {title}")
            elif data:
                skip += 1
                logs.append(f"⏭️ {race_id} 結果未確定（レース前 or 中止）")
            else:
                fail += 1
                logs.append(f"❌ {race_id} データ取得失敗")
        except Exception as e:
            fail += 1
            logs.append(f"❌ {race_id} エラー: {e}")

        progress_bar.progress((i + 1) / total)
        status.text(f"進捗: {i + 1}/{total} (成功: {success}, スキップ: {skip}, 失敗: {fail})")
        log_area.text("\n".join(logs[-5:]))

        time.sleep(SCRAPE_INTERVAL_SEC)

    progress_bar.progress(1.0)
    st.success(f"完了！ 成功: {success}件, スキップ: {skip}件, 失敗: {fail}件")
