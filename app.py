"""競馬予想AI — Streamlit アプリ"""

import streamlit as st

st.set_page_config(
    page_title="競馬予想AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# サイドバー: ページ選択
st.sidebar.title("🏇 競馬予想AI")
page = st.sidebar.radio(
    "メニュー",
    ["📊 予測", "📈 馬場傾向", "🔄 データ更新", "🧪 バックテスト", "ℹ️ DB状況"],
)

# ページルーティング
if page == "📊 予測":
    from app_pages.predict import render
    render()
elif page == "📈 馬場傾向":
    from app_pages.track_bias import render
    render()
elif page == "🔄 データ更新":
    from app_pages.data_update import render
    render()
elif page == "🧪 バックテスト":
    from app_pages.backtest import render
    render()
elif page == "ℹ️ DB状況":
    from app_pages.db_status import render
    render()
