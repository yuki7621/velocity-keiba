"""予測ページ — レース開始前の予測に対応"""

import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from config.settings import DB_PATH
from src.features.track_bias import get_track_bias_for_date
from src.features.predict_features import build_prediction_features
from src.model.train import load_model, FEATURE_COLUMNS, get_available_features


# ──────────────────────────────────────────────
# キャッシュ
# ──────────────────────────────────────────────

@st.cache_resource
def _load_model(name: str):
    """モデルを読み込む（キャッシュ付き）"""
    return load_model(name)


# ──────────────────────────────────────────────
# メインページ
# ──────────────────────────────────────────────

def render():
    st.header("📊 レース予測")

    if not DB_PATH.exists():
        st.warning("データベースが存在しません。「データ更新」ページから取得してください。")
        return

    # ── モード選択 ──
    mode = st.radio(
        "予測モード",
        ["🔮 レース前予測（出馬表から）", "📁 DB内データで予測（過去検証用）"],
        horizontal=True,
    )

    if mode == "🔮 レース前予測（出馬表から）":
        _render_pre_race_prediction()
    else:
        _render_db_prediction()


# ══════════════════════════════════════════════
# レース前予測モード
# ══════════════════════════════════════════════

def _render_pre_race_prediction():
    """出馬表をスクレイピングしてレース前の予測を行う"""

    st.markdown(
        "出馬表（race.netkeiba.com）からエントリー情報を取得し、"
        "過去データを使って予測します。**レース開始前**に実行してください。"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        target_date_input = st.date_input(
            "予測する日付",
            value=date.today(),
            key="prerace_date",
        )
        target_date_str = target_date_input.strftime("%Y%m%d")

    with col2:
        model_name = st.selectbox(
            "モデル", ["lightgbm_v3", "lightgbm_v2", "lightgbm_v1"], key="prerace_model"
        )

    with col3:
        all_venues = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
        venue_filter = st.multiselect("競馬場（空欄で全て）", all_venues, key="prerace_venues")

    # ── 出馬表取得 & 予測 ──
    if st.button("🔮 出馬表を取得して予測", type="primary", key="btn_prerace"):
        _run_pre_race_prediction(target_date_str, target_date_input, venue_filter, model_name)

    # session_stateに結果があれば表示
    if "prerace_results" in st.session_state and st.session_state["prerace_results"] is not None:
        result_df = st.session_state["prerace_results"]
        result_date = st.session_state.get("prerace_results_date", str(target_date_input))
        result_model = st.session_state.get("prerace_results_model", model_name)
        _display_pre_race_results(result_df, result_date, result_model)


def _repredict_single_race(race_id: str, target_date: str, model_name: str):
    """単一レースを再取得・再予測してsession_stateを更新する"""
    try:
        model = _load_model(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。")
        return

    # target_date は str なので date オブジェクトに変換
    if isinstance(target_date, str):
        try:
            td = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            td = date.today()
    else:
        td = target_date

    with st.spinner(f"{race_id} を再取得中..."):
        new_feat = _predict_single_race(race_id, td, model)

    if new_feat is None or len(new_feat) == 0:
        st.error(f"⚠️ {race_id} の再予測に失敗しました。")
        return

    # session_state内の該当レースを差し替え
    current = st.session_state.get("prerace_results")
    if current is None:
        st.session_state["prerace_results"] = new_feat
    else:
        # race_idが一致する行を削除して新しい行を追加
        remaining = current[current["race_id"] != race_id]
        st.session_state["prerace_results"] = pd.concat(
            [remaining, new_feat], ignore_index=True
        )

    st.success(f"✅ {race_id} を再予測しました！")


def _predict_single_race(race_id: str, target_date: date, model) -> pd.DataFrame | None:
    """単一レースの出馬表を取得 → 特徴量構築 → 予測。失敗時はNone。"""
    from src.scraper.race_card import scrape_race_card

    try:
        card = scrape_race_card(race_id)
    except Exception as e:
        st.warning(f"⚠️ {race_id}: 出馬表取得エラー ({e})")
        return None

    if card is None:
        return None

    race_info = card["race_info"]
    entries = card["entries"]

    if "date" not in race_info or not race_info["date"]:
        race_info["date"] = str(target_date)

    try:
        feat_df = build_prediction_features(entries, race_info)
    except Exception as e:
        st.warning(f"⚠️ {race_id}: 特徴量構築エラー ({e})")
        return None

    if len(feat_df) == 0:
        return None

    # 欠損カラムは0で埋める
    for mc in [c for c in FEATURE_COLUMNS if c not in feat_df.columns]:
        feat_df[mc] = 0.0

    X = feat_df[FEATURE_COLUMNS].astype(float)
    feat_df["pred_prob"] = model.predict_proba(X)[:, 1]
    feat_df["race_title"] = race_info.get("title", "")

    return feat_df


def _run_pre_race_prediction(
    date_str: str,
    target_date: date,
    venues: list[str],
    model_name: str,
):
    """出馬表を取得し、過去データから特徴量を構築して予測する"""
    import time
    from src.scraper.race_card import get_upcoming_race_ids
    from config.settings import SCRAPE_INTERVAL_SEC

    # 1) モデル読み込み
    try:
        model = _load_model(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。先に学習を実行してください。")
        return

    # 2) 出馬表のレースID一覧を取得
    status = st.empty()
    status.info(f"📡 {date_str} のレースID一覧を取得中...")

    try:
        race_ids = get_upcoming_race_ids(date_str)
    except Exception as e:
        st.error(f"レースID取得に失敗しました: {e}")
        return

    if not race_ids:
        st.warning(f"⚠️ {date_str} のレースが見つかりません。開催日を確認してください。")
        return

    status.info(f"📡 {len(race_ids)} レースが見つかりました。出馬表を取得中...")

    # 3) 各レースの出馬表を取得 & 予測
    progress = st.progress(0)
    all_predictions = []

    for i, race_id in enumerate(race_ids):
        progress.progress((i + 1) / len(race_ids))

        feat_df = _predict_single_race(race_id, target_date, model)
        if feat_df is None:
            time.sleep(SCRAPE_INTERVAL_SEC)
            continue

        # 競馬場フィルタ
        venue = feat_df["venue"].iloc[0] if "venue" in feat_df.columns else ""
        if venues and venue not in venues:
            time.sleep(SCRAPE_INTERVAL_SEC)
            continue

        status.info(f"📡 {venue} {str(race_id)[-2:]}R を処理中...")

        all_predictions.append(feat_df)
        time.sleep(SCRAPE_INTERVAL_SEC)

    progress.progress(1.0)
    status.empty()

    if not all_predictions:
        st.warning("予測可能なレースがありませんでした。")
        return

    # 4) 結果を結合してsession_stateに保存
    result_df = pd.concat(all_predictions, ignore_index=True)
    st.session_state["prerace_results"] = result_df
    st.session_state["prerace_results_date"] = str(target_date)
    st.session_state["prerace_results_model"] = model_name
    st.session_state["prerace_expand_mode"] = "auto"
    st.success(f"✅ {result_df['race_id'].nunique()} レース・{len(result_df)} 頭の予測が完了しました！")


def _display_pre_race_results(df: pd.DataFrame, target_date: str, model_name: str):
    """レース前予測の結果を表示する"""

    # 全レース再予測ボタン（オプション）
    col_a, col_b, col_c = st.columns([1, 1, 4])
    with col_a:
        if st.button("🗑️ 結果クリア", key="btn_clear_results"):
            st.session_state["prerace_results"] = None
            st.rerun()

    # ── 馬場傾向サマリー（芝/ダート別）──
    st.subheader("📈 前日の馬場傾向")
    has_any_bias = False

    from src.features.track_bias import get_race_day_results, get_previous_day

    for venue in sorted(df["venue"].dropna().unique()):
        prev_day = get_previous_day(target_date, venue)
        if prev_day is None:
            continue
        day_results = get_race_day_results(prev_day, venue)
        if len(day_results) == 0:
            continue

        for surface in ["芝", "ダート"]:
            from src.features.track_bias import analyze_track_bias as _analyze
            bias = _analyze(day_results, surface=surface)
            if bias["n_races"] == 0:
                continue
            has_any_bias = True

            surface_icon = "🌱" if surface == "芝" else "🟤"
            cols = st.columns(5)
            cols[0].markdown(f"**{venue} {surface_icon}{surface}** ({bias['n_races']}R)")
            gate_emoji = "🔴" if bias["gate_bias"] > 0.05 else "🔵" if bias["gate_bias"] < -0.05 else "⚪"
            cols[1].metric(
                "枠順",
                f"{gate_emoji} {'内' if bias['gate_bias'] > 0.05 else '外' if bias['gate_bias'] < -0.05 else '—'}",
            )
            pace_emoji = "🔴" if bias["pace_bias"] > 0.05 else "🔵" if bias["pace_bias"] < -0.05 else "⚪"
            cols[2].metric(
                "脚質",
                f"{pace_emoji} {'先行' if bias['pace_bias'] > 0.05 else '差し' if bias['pace_bias'] < -0.05 else '—'}",
            )
            cols[3].metric(
                "時計",
                f"{'高速' if bias['time_bias'] > 0.1 else 'タフ' if bias['time_bias'] < -0.1 else '標準'}",
            )
            cols[4].metric("上がり", f"{bias['last3f_bias']:.2f}秒差")

    if not has_any_bias:
        st.info(
            "前日のレース結果がDBに無いため、馬場傾向データはありません。\n\n"
            "日曜の予測精度を上げるには、土曜のレース結果を「データ更新」で取得してから再実行してください。"
        )

    st.divider()

    # ── 注目馬サマリー（高確率馬のランキング） ──
    st.subheader("⭐ 注目馬ランキング（AI確率 Top 20）")
    top_horses = df.nlargest(20, "pred_prob")

    display_top = top_horses[[
        "venue", "race_id", "post_number", "horse_name", "jockey_name",
        "pred_prob",
    ]].copy()
    display_top["レース"] = display_top["race_id"].apply(lambda x: f"{str(x)[-2:]}R")
    display_top["AI確率"] = display_top["pred_prob"].apply(lambda x: f"{x:.1%}")
    display_top = display_top.rename(columns={
        "venue": "競馬場", "post_number": "馬番",
        "horse_name": "馬名", "jockey_name": "騎手",
    })
    st.dataframe(
        display_top[["競馬場", "レース", "馬番", "馬名", "騎手", "AI確率"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── レースごとの詳細 ──
    st.subheader("📋 レース別予測")

    # 一括展開/畳むボタン
    btn_col_expand, btn_col_collapse, _ = st.columns([1, 1, 4])
    with btn_col_expand:
        if st.button("📂 全て展開", key="btn_expand_all_prerace"):
            st.session_state["prerace_expand_mode"] = "all"
            st.rerun()
    with btn_col_collapse:
        if st.button("📁 全て畳む", key="btn_collapse_all_prerace"):
            st.session_state["prerace_expand_mode"] = "none"
            st.rerun()

    expand_mode = st.session_state.get("prerace_expand_mode", "auto")

    for race_id in sorted(df["race_id"].unique()):
        race_df = df[df["race_id"] == race_id].sort_values("pred_prob", ascending=False)

        venue = race_df["venue"].iloc[0] if pd.notna(race_df["venue"].iloc[0]) else "?"
        distance = int(race_df["distance"].iloc[0]) if pd.notna(race_df["distance"].iloc[0]) else 0
        surface = race_df["surface"].iloc[0] if pd.notna(race_df["surface"].iloc[0]) else "?"
        condition = race_df["condition"].iloc[0] if "condition" in race_df.columns and pd.notna(race_df["condition"].iloc[0]) else "?"
        race_num = str(race_id)[-2:]
        title = race_df["race_title"].iloc[0] if "race_title" in race_df.columns else ""

        # 展開状態の判定
        top_prob = race_df["pred_prob"].iloc[0]
        if expand_mode == "all":
            is_expanded = True
        elif expand_mode == "none":
            is_expanded = False
        else:
            is_expanded = top_prob >= 0.40

        label = f"**{venue} {race_num}R** — {surface}{distance}m ({condition})"
        if title:
            label += f" {title}"

        with st.expander(label, expanded=is_expanded):
            # ── 再予測ボタン ──
            btn_col1, btn_col2 = st.columns([1, 4])
            with btn_col1:
                if st.button("🔄 このレースを再予測", key=f"repredict_{race_id}"):
                    _repredict_single_race(race_id, target_date, model_name)
                    st.rerun()
            with btn_col2:
                # 馬体重の取得状況を表示
                hw_count = race_df["horse_weight"].notna().sum() if "horse_weight" in race_df.columns else 0
                hw_total = len(race_df)
                if hw_count == 0:
                    st.caption("⚠️ 馬体重未公開（発走30分前頃に公開）")
                elif hw_count < hw_total:
                    st.caption(f"⚠️ 馬体重 {hw_count}/{hw_total} 頭のみ取得済み")
                else:
                    st.caption(f"✅ 馬体重 全{hw_total}頭取得済み")

            display = race_df[[
                "post_number", "gate_number", "horse_name", "jockey_name",
                "weight_carried", "pred_prob",
            ]].copy()

            display.columns = ["馬番", "枠番", "馬名", "騎手", "斤量", "AI確率"]

            # 推奨マーク
            def _mark(prob):
                if prob >= 0.50:
                    return "★★★"
                elif prob >= 0.40:
                    return "★★"
                elif prob >= 0.30:
                    return "★"
                return ""

            display["推奨"] = display["AI確率"].apply(_mark)
            display["AI確率"] = display["AI確率"].apply(lambda x: f"{x:.1%}")
            display["馬番"] = display["馬番"].astype(int)
            display["枠番"] = display["枠番"].astype(int)
            display["斤量"] = display["斤量"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "-"
            )

            st.dataframe(
                display[["枠番", "馬番", "馬名", "騎手", "斤量", "AI確率", "推奨"]],
                use_container_width=True,
                hide_index=True,
            )

            # 上位3頭のレーダーチャート的コメント
            top3 = race_df.head(3)
            for _, horse in top3.iterrows():
                name = horse["horse_name"]
                prob = horse["pred_prob"]
                comments = []

                if pd.notna(horse.get("horse_top3_rate_5")) and horse["horse_top3_rate_5"] >= 0.5:
                    comments.append("近走好調")
                if pd.notna(horse.get("horse_dist_top3_rate")) and horse["horse_dist_top3_rate"] >= 0.5:
                    comments.append("距離適性◎")
                if pd.notna(horse.get("horse_venue_top3_rate")) and horse["horse_venue_top3_rate"] >= 0.5:
                    comments.append("コース巧者")
                if pd.notna(horse.get("horse_surface_top3_rate")) and horse["horse_surface_top3_rate"] >= 0.5:
                    comments.append("馬場適性◎")
                if pd.notna(horse.get("jockey_venue_top3")) and horse["jockey_venue_top3"] >= 0.3:
                    comments.append("騎手得意場")
                if pd.notna(horse.get("horse_form_trend")) and horse["horse_form_trend"] < -0.5:
                    comments.append("上昇気流")
                if pd.notna(horse.get("days_since_last")) and horse["days_since_last"] >= 70:
                    comments.append("休み明け")

                comment_str = "、".join(comments) if comments else "—"
                st.caption(f"🏇 **{name}** ({prob:.1%}) — {comment_str}")


# ══════════════════════════════════════════════
# DB内データ予測モード（従来の過去検証用）
# ══════════════════════════════════════════════

def _render_db_prediction():
    """DB内のデータを使った予測（過去レースの検証用）"""

    st.markdown(
        "DB内に保存済みのレースデータで予測を実行します。"
        "過去レースの検証・バックテスト向けです。"
    )

    conn = sqlite3.connect(DB_PATH)

    col1, col2, col3 = st.columns(3)
    with col1:
        dates = pd.read_sql_query(
            "SELECT DISTINCT date FROM races ORDER BY date DESC LIMIT 60", conn
        )["date"].tolist()
        if dates:
            target_date = st.selectbox("予測する日付", dates, key="db_date")
        else:
            st.warning("データがありません。先に「データ更新」で取得してください。")
            conn.close()
            return

    with col2:
        venues_on_date = pd.read_sql_query(
            "SELECT DISTINCT venue FROM races WHERE date = ? ORDER BY venue",
            conn,
            params=[target_date],
        )["venue"].tolist()
        all_venues = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
        venue_options = venues_on_date if venues_on_date else all_venues
        default_venues = venues_on_date if venues_on_date else []
        venue_filter = st.multiselect(
            "競馬場（空欄で全て）", venue_options, default=default_venues, key="db_venues"
        )

    with col3:
        model_name = st.selectbox(
            "モデル", ["lightgbm_v3", "lightgbm_v2", "lightgbm_v1"], key="db_model"
        )

    conn.close()

    if st.button("予測を実行", type="primary", key="btn_db"):
        _run_db_prediction(target_date, venue_filter, model_name)


def _run_db_prediction(target_date: str, venues: list[str], model_name: str):
    """DB内データで予測を実行して結果を表示する（従来方式）"""
    from src.features.build_features import build_all_features

    # モデル読み込み
    try:
        model = _load_model(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。先に学習を実行してください。")
        return

    # 特徴量構築
    with st.spinner("特徴量を構築中（初回は数分かかります）..."):
        df = build_all_features()
    features = get_available_features(df)

    # 対象日のデータ
    target_df = df[df["date"] == target_date].copy()

    if venues:
        target_df = target_df[target_df["venue"].isin(venues)]

    if len(target_df) == 0:
        st.warning(f"{target_date} の出走データがありません。データ取得済みか確認してください。")
        return

    # 予測
    X = target_df[features]
    target_df["pred_prob"] = model.predict_proba(X)[:, 1]

    # 期待値・エッジ
    target_df["fukusho_odds"] = (target_df["odds"] * 0.3).clip(lower=1.1)
    target_df["expected_value"] = target_df["pred_prob"] * target_df["fukusho_odds"]
    target_df["market_prob"] = (3.0 / target_df["odds"]).clip(upper=1.0)
    target_df["edge"] = target_df["pred_prob"] - target_df["market_prob"]

    # 馬名取得
    conn = sqlite3.connect(DB_PATH)
    horse_names = pd.read_sql_query("SELECT horse_id, name FROM horses", conn)
    conn.close()
    target_df = target_df.merge(horse_names, on="horse_id", how="left")

    # ── 馬場傾向サマリー（芝/ダート別）──
    st.subheader("📈 前日の馬場傾向")
    has_any_bias_db = False

    from src.features.track_bias import get_race_day_results, get_previous_day
    from src.features.track_bias import analyze_track_bias as _analyze

    for venue in target_df["venue"].unique():
        prev_day = get_previous_day(target_date, venue)
        if prev_day is None:
            continue
        day_results = get_race_day_results(prev_day, venue)
        if len(day_results) == 0:
            continue

        for surface in ["芝", "ダート"]:
            bias = _analyze(day_results, surface=surface)
            if bias["n_races"] == 0:
                continue
            has_any_bias_db = True

            surface_icon = "🌱" if surface == "芝" else "🟤"
            cols = st.columns(5)
            cols[0].markdown(f"**{venue} {surface_icon}{surface}** ({bias['n_races']}R)")
            gate_emoji = "🔴" if bias["gate_bias"] > 0.05 else "🔵" if bias["gate_bias"] < -0.05 else "⚪"
            cols[1].metric("枠順", f"{gate_emoji} {'内' if bias['gate_bias'] > 0.05 else '外' if bias['gate_bias'] < -0.05 else '—'}")
            pace_emoji = "🔴" if bias["pace_bias"] > 0.05 else "🔵" if bias["pace_bias"] < -0.05 else "⚪"
            cols[2].metric("脚質", f"{pace_emoji} {'先行' if bias['pace_bias'] > 0.05 else '差し' if bias['pace_bias'] < -0.05 else '—'}")
            cols[3].metric("時計", f"{'高速' if bias['time_bias'] > 0.1 else 'タフ' if bias['time_bias'] < -0.1 else '標準'}")
            cols[4].metric("上がり", f"{bias['last3f_bias']:.2f}秒差")

    if not has_any_bias_db:
        st.info(
            "前日のレース結果がDBに無いため、馬場傾向データはありません。\n\n"
            "日曜の予測精度を上げるには、土曜のレース結果を「データ更新」で取得してから再実行してください。"
        )

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

    # 一括展開/畳むボタン
    btn_col_expand, btn_col_collapse, _ = st.columns([1, 1, 4])
    with btn_col_expand:
        if st.button("📂 全て展開", key="btn_expand_all_db"):
            st.session_state["db_expand_mode"] = "all"
            st.rerun()
    with btn_col_collapse:
        if st.button("📁 全て畳む", key="btn_collapse_all_db"):
            st.session_state["db_expand_mode"] = "none"
            st.rerun()

    expand_mode_db = st.session_state.get("db_expand_mode", "auto")

    for race_id in sorted(target_df["race_id"].unique()):
        race_df = target_df[target_df["race_id"] == race_id].sort_values("pred_prob", ascending=False)

        venue = race_df["venue"].iloc[0]
        distance = int(race_df["distance"].iloc[0])
        surface = race_df["surface"].iloc[0]
        condition = race_df["condition"].iloc[0] if pd.notna(race_df["condition"].iloc[0]) else "?"
        race_num = str(race_id)[-2:]

        n_value = len(race_df[race_df["edge"] >= 0.10])
        value_tag = f" ⭐{n_value}" if n_value > 0 else ""

        if expand_mode_db == "all":
            is_expanded = True
        elif expand_mode_db == "none":
            is_expanded = False
        else:
            is_expanded = n_value > 0

        with st.expander(
            f"**{venue} {race_num}R** — {surface}{distance}m ({condition}){value_tag}",
            expanded=is_expanded,
        ):
            display = race_df[[
                "post_number", "name", "pred_prob", "odds",
                "expected_value", "edge", "finish_position",
            ]].copy()

            display.columns = ["馬番", "馬名", "AI確率", "ｵｯｽﾞ", "期待値", "エッジ", "着順"]

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
