"""予測ページ — レース開始前の予測に対応"""

import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime

from config.settings import DB_PATH
from src.model.train import load_model, get_available_features
from src.betting.sanrenpuku_filter import (
    evaluate_race as evaluate_sanrenpuku,
    STRATEGY_NAME as SANRENPUKU_STRATEGY,
    EXPECTED_ROI as SANRENPUKU_ROI,
)


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
            "モデル", ["lightgbm_v8", "lightgbm_v6", "lightgbm_v7", "lightgbm_v5", "lightgbm_v4", "lightgbm_v3", "lightgbm_v2", "lightgbm_v1"], key="prerace_model",
            help="v8 推奨 (展開シナジー特徴量追加で v6 を全項目上回り。v7 は撤退モデル)"
        )

    with col3:
        all_venues = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
        venue_filter = st.multiselect("競馬場（空欄で全て）", all_venues, key="prerace_venues")

    # ── 馬体重補完オプション（土曜夜の事前予測用）──
    impute_weight = st.checkbox(
        "🌙 馬体重が未発表の場合、前走の値で補完する（土曜夜の事前スクリーニング用）",
        value=False,
        key="prerace_impute_weight",
        help=(
            "当日朝の馬体重発表前でも予測できるようにします。"
            "前走の馬体重を代理値として使用し、weight_change は 0 と仮定します。"
            "休み明けの馬では誤差が出やすいので、最終判定は発表後の再予測で行ってください。"
        ),
    )

    # ── 出馬表取得 & 予測 ──
    if st.button("🔮 出馬表を取得して予測", type="primary", key="btn_prerace"):
        _run_pre_race_prediction(
            target_date_str, target_date_input, venue_filter, model_name,
            impute_weight=impute_weight,
        )

    # session_stateに結果があれば表示
    if "prerace_results" in st.session_state and st.session_state["prerace_results"] is not None:
        result_df = st.session_state["prerace_results"]
        result_date = st.session_state.get("prerace_results_date", str(target_date_input))
        result_model = st.session_state.get("prerace_results_model", model_name)
        _display_pre_race_results(result_df, result_date, result_model)


def _repredict_single_race(race_id: str, target_date: str, model_name: str,
                           rebuild_features: bool = False):
    """単一レースを再取得して session_state を更新する

    Args:
        rebuild_features: False（既定）なら**オッズのみ再取得**して edge/EV を
            計算し直す高速パス。過去成績ベースの特徴量は再取得しても変わらない
            ため、オッズ更新用途ではこれで十分（数秒で完了）。
            True なら出馬表ごと取り直し、学習と同じパイプラインで特徴量を
            再構築する（馬体重の発表後など。約2分）。
    """
    current = st.session_state.get("prerace_results")

    # ── 高速パス: オッズだけ更新（特徴量は変わらないので再構築しない）──
    if not rebuild_features and current is not None:
        g = current[current["race_id"] == race_id].copy()
        if len(g) > 0:
            with st.spinner(f"{race_id} のオッズを再取得中..."):
                g = _attach_odds(g, str(race_id))
                g = _add_edge_columns(g)
            remaining = current[current["race_id"] != race_id]
            st.session_state["prerace_results"] = pd.concat([remaining, g], ignore_index=True)
            st.success(f"✅ {race_id} のオッズを更新しました（AI確率は変わりません）")
            return

    # ── 完全再構築パス: 出馬表から取り直して学習と同じ経路で特徴量を作る ──
    from src.features.build_features import build_all_features, race_cards_to_pending
    from src.model.train import get_available_features, prepare_dataset
    from src.scraper.race_card import scrape_race_card

    try:
        model = _load_model(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。")
        return

    with st.spinner(f"{race_id} を再取得して特徴量を再構築中...（約2分）"):
        try:
            card = scrape_race_card(race_id)
        except Exception as e:
            st.error(f"⚠️ {race_id}: 出馬表取得エラー ({e})")
            return
        if card is None:
            st.error(f"⚠️ {race_id} の出馬表を取得できませんでした。")
            return
        if not card["race_info"].get("date"):
            card["race_info"]["date"] = str(target_date)

        pending = race_cards_to_pending([card])
        full_df = build_all_features(pending_races=pending)
        full_df = prepare_dataset(full_df, keep_pending=True)
        features = get_available_features(full_df)

        new_feat = full_df[full_df["is_pending"] == 1].copy()
        if len(new_feat) == 0:
            st.error(f"⚠️ {race_id} の特徴量を構築できませんでした。")
            return
        new_feat["pred_prob"] = model.predict_proba(new_feat[features])[:, 1]
        new_feat = _attach_odds(new_feat, str(race_id))
        new_feat = _add_edge_columns(new_feat)

    if current is None:
        st.session_state["prerace_results"] = new_feat
    else:
        remaining = current[current["race_id"] != race_id]
        st.session_state["prerace_results"] = pd.concat([remaining, new_feat], ignore_index=True)

    st.success(f"✅ {race_id} を再予測しました！")


def _attach_odds(feat_df: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """単勝オッズを取得して odds 列に付与する（失敗時は NaN のまま継続）"""
    from src.scraper.odds import fetch_all_odds

    odds_fetched_at = datetime.now().strftime("%H:%M:%S")
    odds_official_dt = None
    try:
        odds_dict = fetch_all_odds(race_id)
        tansho = odds_dict.get("tansho", {}) if odds_dict else {}
        odds_official_dt = odds_dict.get("official_datetime") if odds_dict else None
    except Exception as e:
        st.warning(f"⚠️ {race_id}: オッズ取得エラー ({e}) — 予測のみ継続")
        tansho = {}

    if tansho:
        feat_df["odds"] = feat_df["post_number"].apply(
            lambda p: tansho.get(str(int(p))) if pd.notna(p) else None
        )
        feat_df["odds_fetched_at"] = odds_fetched_at
        feat_df["odds_official_dt"] = odds_official_dt or ""
        try:
            st.session_state.setdefault("odds_debug", {})[race_id] = {
                "fetched_at": odds_fetched_at,
                "official_dt": odds_official_dt,
                "tansho": dict(tansho),
            }
        except Exception:
            pass
    else:
        feat_df["odds"] = np.nan
        feat_df["odds_fetched_at"] = ""
        feat_df["odds_official_dt"] = ""
    return feat_df


def _add_edge_columns(feat_df: pd.DataFrame) -> pd.DataFrame:
    """オッズから market_prob / edge / expected_value を計算する"""
    if "odds" in feat_df.columns and feat_df["odds"].notna().any():
        from src.betting.market_implied import add_market_top3_column
        feat_df = add_market_top3_column(feat_df, out_col="market_prob")
        # Plackett-Luce が NaN の行は旧式（3/odds）でフォールバック
        fallback = (3.0 / feat_df["odds"]).clip(upper=1.0)
        feat_df["market_prob"] = feat_df["market_prob"].fillna(fallback)
        feat_df["edge"] = feat_df["pred_prob"] - feat_df["market_prob"]
        feat_df["expected_value"] = feat_df["pred_prob"] * (feat_df["odds"] * 0.3).clip(lower=1.1)
    else:
        feat_df["market_prob"] = np.nan
        feat_df["edge"] = np.nan
        feat_df["expected_value"] = np.nan
    return feat_df


def _run_pre_race_prediction(
    date_str: str,
    target_date: date,
    venues: list[str],
    model_name: str,
    impute_weight: bool = False,
):
    """出馬表を取得し、学習と同一のパイプラインで特徴量を構築して予測する"""
    import time

    from config.settings import SCRAPE_INTERVAL_SEC
    from src.features.build_features import build_all_features, race_cards_to_pending
    from src.features.track_bias import clear_track_bias_cache
    from src.model.train import get_available_features, prepare_dataset
    from src.scraper.race_card import get_upcoming_race_ids, scrape_race_card

    # 馬場傾向キャッシュをクリア（DB更新後の鮮度確保）。
    # この後の同一開催日・会場の前日バイアスはバッチ内で再利用され高速化される。
    clear_track_bias_cache()

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

    # 3) 全レースの出馬表を取得（この時点では予測しない）
    progress = st.progress(0)
    cards = []
    for i, race_id in enumerate(race_ids):
        progress.progress((i + 1) / len(race_ids) * 0.6)
        try:
            card = scrape_race_card(race_id)
        except Exception as e:
            st.warning(f"⚠️ {race_id}: 出馬表取得エラー ({e})")
            time.sleep(SCRAPE_INTERVAL_SEC)
            continue
        if card is None:
            time.sleep(SCRAPE_INTERVAL_SEC)
            continue

        info = card["race_info"]
        if not info.get("date"):
            info["date"] = str(target_date)
        venue = info.get("venue", "")
        if venues and venue not in venues:
            time.sleep(SCRAPE_INTERVAL_SEC)
            continue

        status.info(f"📡 {venue} {str(race_id)[-2:]}R の出馬表を取得中...")
        cards.append(card)
        time.sleep(SCRAPE_INTERVAL_SEC)

    if not cards:
        progress.empty()
        status.empty()
        st.warning("予測可能なレースがありませんでした。")
        return

    # 4) 学習と同一のパイプラインで特徴量を構築して一括予測
    #    ※ 予測専用の近似実装だと脚質・レース内相対値が学習時と別物になり
    #      AUC が 0.064 劣化していたため、必ずこの経路を通す
    status.info(f"🧮 {len(cards)} レース分の特徴量を構築中...（約2分）")
    progress.progress(0.7)

    pending = race_cards_to_pending(cards)
    full_df = build_all_features(pending_races=pending)
    full_df = prepare_dataset(full_df, keep_pending=True)
    features = get_available_features(full_df)

    target_df = full_df[full_df["is_pending"] == 1].copy()
    if len(target_df) == 0:
        progress.empty()
        status.empty()
        st.warning("予測対象の行を構築できませんでした。")
        return

    target_df["pred_prob"] = model.predict_proba(target_df[features])[:, 1]
    progress.progress(0.85)

    # 5) レースごとにオッズを取得して edge / EV を計算
    status.info("📡 オッズを取得中...")
    per_race = []
    race_id_list = list(target_df["race_id"].unique())
    for i, rid in enumerate(race_id_list):
        progress.progress(0.85 + (i + 1) / len(race_id_list) * 0.15)
        g = target_df[target_df["race_id"] == rid].copy()
        g = _attach_odds(g, str(rid))
        g = _add_edge_columns(g)
        per_race.append(g)
        time.sleep(SCRAPE_INTERVAL_SEC)

    progress.progress(1.0)
    progress.empty()
    status.empty()

    result_df = pd.concat(per_race, ignore_index=True)
    st.session_state["prerace_results"] = result_df
    st.session_state["prerace_results_date"] = str(target_date)
    st.session_state["prerace_results_model"] = model_name
    st.session_state["prerace_results_imputed"] = impute_weight
    st.session_state["prerace_expand_mode"] = "auto"

    # 補完統計
    if impute_weight and "weight_imputed" in result_df.columns:
        n_imputed = int(result_df["weight_imputed"].fillna(False).astype(bool).sum())
        st.success(
            f"✅ {result_df['race_id'].nunique()} レース・{len(result_df)} 頭の予測が完了しました！"
            f"（うち {n_imputed} 頭は前走馬体重で補完）"
        )
    else:
        st.success(f"✅ {result_df['race_id'].nunique()} レース・{len(result_df)} 頭の予測が完了しました！")


def _display_pre_race_results(df: pd.DataFrame, target_date: str, model_name: str):
    """レース前予測の結果を表示する"""

    # 全レース再予測ボタン + 当日予測の保存ボタン
    col_a, col_b, col_c = st.columns([1, 1.4, 3])
    with col_a:
        if st.button("🗑️ 結果クリア", key="btn_clear_results"):
            st.session_state["prerace_results"] = None
            st.rerun()
    with col_b:
        if st.button("💾 この予測をDBに保存", key="btn_save_predictions",
                     help="当日の予測をそのままDBに保存し、後日「収支レビュー」で実結果と正確に突き合わせられます。"
                          "（再計算ではなく当日の値を保持するため、DB更新後も予測がブレません）"):
            from src.db.predictions import save_predictions
            n = save_predictions(df, model_name=model_name, race_date=str(target_date)[:10])
            st.success(
                f"✅ {df['race_id'].nunique()}レース・{n}頭の予測を保存しました。"
                f"後日「🪞 収支レビュー」で『当日保存した予測』を選んで参照できます。"
            )

    # ── 補完モードの警告バナー ──
    imputed_flag = bool(st.session_state.get("prerace_results_imputed", False))
    if imputed_flag and "weight_imputed" in df.columns:
        n_imputed = int(df["weight_imputed"].fillna(False).astype(bool).sum())
        n_total = len(df)
        st.warning(
            f"🌙 **馬体重補完モードで予測中** — {n_imputed} / {n_total} 頭が前走馬体重で補完されています。\n\n"
            "これはスクリーニング用の参考値です。確定予測は馬体重発表後（当日朝）に再実行してください。"
        )

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
        width='stretch',
        hide_index=True,
    )

    st.divider()

    # ── レースごとの詳細 ──
    st.subheader("📋 レース別予測")

    # 一括展開/畳む + ソート方式選択
    btn_col_expand, btn_col_collapse, sort_col, _ = st.columns([1, 1, 2, 2])
    with btn_col_expand:
        if st.button("📂 全て展開", key="btn_expand_all_prerace"):
            st.session_state["prerace_expand_mode"] = "all"
            st.rerun()
    with btn_col_collapse:
        if st.button("📁 全て畳む", key="btn_collapse_all_prerace"):
            st.session_state["prerace_expand_mode"] = "none"
            st.rerun()
    with sort_col:
        sort_mode = st.radio(
            "並び順",
            ["⏱ 発走時間順", "🏟 競馬場別"],
            horizontal=True,
            key="prerace_sort_mode",
            index=0,  # デフォルト: 発走時間順
            label_visibility="collapsed",
        )

    expand_mode = st.session_state.get("prerace_expand_mode", "auto")

    # ── ソート順を決定 ──
    # 各 race_id について (start_time, race_id) のタプルを作る
    # start_time が無い場合は race_id 末尾2桁 (R番号) で代用
    def _race_sort_key(race_id: str, mode: str) -> tuple:
        sub = df[df["race_id"] == race_id]
        st_raw = ""
        if "start_time" in sub.columns:
            stv = sub["start_time"].dropna()
            stv = stv[stv != ""]
            if len(stv) > 0:
                st_raw = stv.iloc[0]
        venue_v = sub["venue"].iloc[0] if "venue" in sub.columns and len(sub) else ""
        race_num = str(race_id)[-2:]

        if mode == "⏱ 発走時間順":
            # start_time → R番号 → race_id の順でフォールバック
            return (st_raw or f"99:{race_num}", venue_v, race_id)
        # 競馬場別: 会場名 → 発走時刻 → race_id
        return (venue_v, st_raw or f"99:{race_num}", race_id)

    sorted_race_ids = sorted(df["race_id"].unique(), key=lambda rid: _race_sort_key(rid, sort_mode))

    for race_id in sorted_race_ids:
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

        # 確率バッジ（最高予測確率を視覚的に表示）
        prob_pct = int(top_prob * 100)
        if top_prob >= 0.50:
            prob_badge = f"🔥 {prob_pct}%"
        elif top_prob >= 0.40:
            prob_badge = f"⭐ {prob_pct}%"
        else:
            prob_badge = f"　 {prob_pct}%"

        # 3連複BOX推奨判定
        sanren_rec = evaluate_sanrenpuku(race_df)

        # 順位帯の最高ランク (🎯🎯 がいるかチェック) — race_df は pred_prob 降順済み
        has_strong_rank_band = False
        has_rank_band = False
        if "odds" in race_df.columns:
            for i, (_, row) in enumerate(race_df.iterrows(), start=1):
                o = row.get("odds")
                if pd.isna(o):
                    continue
                if 30.0 <= o < 40.0:
                    continue
                if i in (2, 3) and 20.0 <= o <= 50.0:
                    has_strong_rank_band = True
                    break
                if i in (1, 2) and 10.0 <= o < 20.0:
                    has_rank_band = True

        # 発走時刻があればラベル先頭に表示
        st_label = ""
        if "start_time" in race_df.columns:
            stv = race_df["start_time"].dropna()
            stv = stv[stv != ""]
            if len(stv) > 0:
                st_label = f"⏱{stv.iloc[0]} "
        label = f"**{st_label}{venue} {race_num}R** — {surface}{distance}m ({condition})"
        if title:
            label += f" {title}"
        label += f"　　{prob_badge}"
        if sanren_rec.is_recommended:
            label += "　🎯 3連複BOX"
        if has_strong_rank_band:
            label += "　🎯🎯 順位帯"
        elif has_rank_band:
            label += "　🎯 順位帯"

        with st.expander(label, expanded=is_expanded):
            # ── 再取得ボタン（オッズのみ / 完全再予測）──
            btn_col1, btn_col1b, btn_col2 = st.columns([1, 1.2, 3])
            with btn_col1:
                if st.button("💹 オッズ更新", key=f"refresh_odds_{race_id}",
                             help="オッズだけ取り直して期待値・エッジを再計算します（数秒）。"
                                  "AI確率は過去成績ベースなので変わりません。"):
                    _repredict_single_race(race_id, target_date, model_name,
                                           rebuild_features=False)
                    st.rerun()
            with btn_col1b:
                if st.button("🔄 完全に再予測", key=f"repredict_{race_id}",
                             help="出馬表から取り直し、学習と同じパイプラインで特徴量を"
                                  "再構築します（約2分）。馬体重の発表後に使ってください。"):
                    _repredict_single_race(race_id, target_date, model_name,
                                           rebuild_features=True)
                    st.rerun()
            with btn_col2:
                # 馬体重 + オッズ取得時刻を表示
                hw_count = race_df["horse_weight"].notna().sum() if "horse_weight" in race_df.columns else 0
                hw_total = len(race_df)
                if hw_count == 0:
                    weight_msg = "⚠️ 馬体重未公開（発走30分前頃に公開）"
                elif hw_count < hw_total:
                    weight_msg = f"⚠️ 馬体重 {hw_count}/{hw_total} 頭のみ取得済み"
                else:
                    weight_msg = f"✅ 馬体重 全{hw_total}頭取得済み"

                # オッズ取得時刻 + netkeiba 側更新時刻（official_dt）
                fetched_at = ""
                official_dt = ""
                if "odds_fetched_at" in race_df.columns:
                    vals = race_df["odds_fetched_at"].dropna()
                    vals = vals[vals != ""]
                    if len(vals) > 0:
                        fetched_at = vals.iloc[0]
                if "odds_official_dt" in race_df.columns:
                    ovals = race_df["odds_official_dt"].dropna()
                    ovals = ovals[ovals != ""]
                    if len(ovals) > 0:
                        official_dt = ovals.iloc[0]
                if fetched_at:
                    msg = f"📡 取得 {fetched_at}"
                    if official_dt:
                        msg += f" ｜ netkeiba更新時刻 {official_dt}"
                    st.caption(f"{weight_msg} ｜ {msg}")
                else:
                    st.caption(f"{weight_msg} ｜ ⚠️ オッズ未取得")

            cols_src = [
                "post_number", "gate_number", "horse_name", "jockey_name",
                "weight_carried", "horse_weight", "pred_prob",
            ]
            # オッズ系列（取得済みなら）
            has_odds_col = "odds" in race_df.columns and race_df["odds"].notna().any()
            if has_odds_col:
                cols_src += ["odds", "expected_value", "edge"]
            # weight_imputed 列が存在すれば含める
            has_imputed_col = "weight_imputed" in race_df.columns
            if has_imputed_col:
                cols_src.append("weight_imputed")

            display = race_df[cols_src].copy()
            # AI順位 (race_df は pred_prob 降順でソート済み) — 1始まり
            display["ai_rank"] = range(1, len(display) + 1)
            # 相対AI確率（レース内で正規化 = 全頭の合計が100%）
            # AI確率は「独立した3着以内確率」なので合計が約300%近くになる。
            # それをレース内で正規化することで「他馬と比べての相対的な強さ」を示す。
            race_prob_sum = display["pred_prob"].sum()
            if race_prob_sum > 0:
                display["normalized_prob"] = display["pred_prob"] / race_prob_sum
            else:
                display["normalized_prob"] = 0.0

            # 推奨マーク（オッズがあればedge/EVベース、なければAI確率ベース）
            if has_odds_col:
                def _mark(row):
                    edge = row.get("edge")
                    ev = row.get("expected_value")
                    if pd.isna(edge) or pd.isna(ev):
                        return ""
                    if edge >= 0.15 and ev >= 1.0:
                        return "★★★"
                    elif edge >= 0.10 and ev >= 0.8:
                        return "★★"
                    elif edge >= 0.05:
                        return "★"
                    return ""
                display["推奨"] = display.apply(_mark, axis=1)
            else:
                def _mark_prob(prob):
                    if prob >= 0.50:
                        return "★★★"
                    elif prob >= 0.40:
                        return "★★"
                    elif prob >= 0.30:
                        return "★"
                    return ""
                display["推奨"] = display["pred_prob"].apply(_mark_prob)

            display["AI確率"] = display["pred_prob"].apply(lambda x: f"{x:.1%}")
            display["相対%"] = display["normalized_prob"].apply(lambda x: f"{x:.1%}")
            display["馬番"] = display["post_number"].astype(int)
            display["枠番"] = display["gate_number"].astype(int)
            display["斤量"] = display["weight_carried"].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "-"
            )

            # 馬体重カラム（補完時は 🌙 マーク付き）
            def _fmt_weight(row):
                hw = row.get("horse_weight")
                if pd.isna(hw) or not hw:
                    return "-"
                imputed = bool(row.get("weight_imputed", False)) if has_imputed_col else False
                suffix = " 🌙" if imputed else ""
                return f"{int(hw)}{suffix}"

            display["体重"] = display.apply(_fmt_weight, axis=1)
            display["馬名"] = display["horse_name"]
            display["騎手"] = display["jockey_name"]

            if has_odds_col:
                # オッズ帯マーク (v8 診断結果 × 個別馬エッジ):
                #   バックテスト実証で ROI 100% 超えに必要なのは **edge≥0.10**
                #   (edge≥0.05 だと 15-20帯で 103%、25-30帯で 100% と帯依存で不安定)
                #
                #   <5倍   ⛔ 本命過熱  (帯起因の赤字、エッジ無関係に警告)
                #   5-10倍 ⚠️ 要注意    (帯起因の弱い赤字、エッジ無関係に警告)
                #   10-30倍 + edge≥0.10 → 🟢       (15-20帯 ROI 104.1%)
                #          + edge<0.10  → ○        (帯OKだがエッジ不足、ROI <100%)
                #   30-40倍 ⛔ 鬼門帯   (高エッジでも構造的赤字)
                #   40-50倍 + edge≥0.10 → 🟢🟢高ROI (ROI 128.9%)
                #          + edge<0.10  → ○        (帯OKだがエッジ不足)
                #   >50倍   🌪️ 大穴    (分散大)
                def _odds_warn(row):
                    o = row.get("odds")
                    e = row.get("edge")
                    if pd.isna(o):
                        return ""
                    if o < 5.0:
                        return "⛔本命過熱"
                    if o < 10.0:
                        return "⚠️要注意"
                    if 30.0 <= o < 40.0:
                        return "⛔鬼門帯"
                    if o > 50.0:
                        return "🌪️大穴"
                    # 10-30倍 or 40-50倍 (中穴 / 高ROI 帯)
                    edge_ok = pd.notna(e) and e >= 0.10
                    is_high_roi_band = 40.0 <= o <= 50.0
                    if not edge_ok:
                        return "○"  # 帯は良いがエッジ不足 (ROI<100% 想定)
                    return "🟢🟢高ROI" if is_high_roi_band else "🟢"

                # ── AI順位 × オッズ帯 のクロス基準 (v8 バックテスト実績ベース) ──
                # 複勝ROI が 100% 超えのセグメント（複勝券で買った場合）:
                #   AI 1位 × 単勝10-20倍 → 複勝ROI 108%   🎯
                #   AI 2位 × 単勝10-20倍 → 複勝ROI 101%   🎯
                #   AI 2位 × 単勝20-50倍 → 複勝ROI 145%   🎯🎯 (最強・本命枠)
                #   AI 3位 × 単勝20-50倍 → 複勝ROI 130%   🎯🎯
                # 鬼門帯30-40倍は順位無関係に除外
                def _rank_band_mark(row):
                    rank = row.get("ai_rank")
                    o = row.get("odds")
                    if pd.isna(rank) or pd.isna(o):
                        return ""
                    rank = int(rank)
                    # 30-40倍は鬼門帯（複勝ROI 64%等）なので順位関係なく除外
                    if 30.0 <= o < 40.0:
                        return ""
                    # 🎯🎯 高信頼プラス (ROI 130%超)
                    if rank in (2, 3) and 20.0 <= o <= 50.0:
                        return "🎯🎯"
                    # 🎯 プラス (ROI 100-110%)
                    if rank in (1, 2) and 10.0 <= o < 20.0:
                        return "🎯"
                    return ""

                display["AI順位"] = display["ai_rank"].apply(lambda r: f"{int(r)}位")
                display["ｵｯｽﾞ"] = display["odds"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
                display["帯"] = display.apply(_odds_warn, axis=1)
                display["順位帯"] = display.apply(_rank_band_mark, axis=1)
                display["期待値"] = display["expected_value"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                display["エッジ"] = display["edge"].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "-")
                show_cols = ["枠番", "馬番", "馬名", "騎手", "斤量", "体重", "AI順位", "AI確率", "相対%", "ｵｯｽﾞ", "帯", "順位帯", "期待値", "エッジ", "推奨"]
            else:
                show_cols = ["枠番", "馬番", "馬名", "騎手", "斤量", "体重", "AI確率", "相対%", "推奨"]

            st.dataframe(
                display[show_cols],
                width='stretch',
                hide_index=True,
            )

            # 🔍 オッズデバッグ (取得値とDataFrameに格納された値を比較)
            with st.expander("🔍 オッズデバッグ（生取得値 vs 表示値の比較）", expanded=False):
                debug_dict = st.session_state.get("odds_debug", {}).get(race_id)
                if debug_dict:
                    official_dt_dbg = debug_dict.get('official_dt') or '不明'
                    st.caption(
                        f"クライアント取得時刻 (fetched_at): **{debug_dict['fetched_at']}**  |  "
                        f"netkeiba API が報告する公式更新時刻 (official_dt): **{official_dt_dbg}**  \n"
                        f"⚠️ netkeiba 画面と値が違う場合はこの 2つの時刻差を確認してください "
                        f"（official_dt が古ければ netkeiba 側 API のキャッシュ／更新間隔の問題）"
                    )
                    raw_tansho = debug_dict["tansho"]
                    rows_dbg = []
                    for _, r in race_df.sort_values("post_number").iterrows():
                        pn = int(r["post_number"]) if pd.notna(r["post_number"]) else None
                        rows_dbg.append({
                            "馬番": pn,
                            "馬名": r.get("horse_name", "?"),
                            "raw_tansho (API)": raw_tansho.get(str(pn)) if pn is not None else None,
                            "feat_df.odds (表示)": r.get("odds"),
                            "差分": (
                                "✅ 一致"
                                if pn is not None and raw_tansho.get(str(pn)) == r.get("odds")
                                else "❌ 不一致"
                            ),
                        })
                    st.dataframe(pd.DataFrame(rows_dbg), width='stretch', hide_index=True)
                    st.caption(
                        "💡 全行「✅一致」なら fetch は正常 → netkeiba側オッズが古い可能性。"
                        "「❌不一致」が出たらコード側のバグです。"
                    )
                else:
                    st.info("このレースのオッズデバッグデータがありません。再予測ボタンを押してください。")

            if has_odds_col:
                # AI Top3 内に本命過熱帯（<5倍）の馬がいればレース単位で警告
                top3_overheated = race_df.head(3)[
                    race_df.head(3)["odds"].notna() & (race_df.head(3)["odds"] < 5.0)
                ]
                if len(top3_overheated) > 0:
                    posts = ", ".join(f"{int(p)}番({o:.1f}倍)" for p, o in
                                       zip(top3_overheated["post_number"], top3_overheated["odds"]))
                    st.warning(
                        f"⛔ **本命過熱警告**: AI Top3 内に単勝<5倍の馬がいます ({posts})。  \n"
                        f"v6 診断ではこの帯のROIは49〜61%（控除率20%を超える赤字）。"
                        f"購入する場合はオッズ帯フィルタ適用後の買い目推奨ページを利用してください。"
                    )
                # 30-40倍の鬼門帯に AI 上位馬がいる場合の警告
                top5_kimon = race_df.head(5)[
                    race_df.head(5)["odds"].notna()
                    & (race_df.head(5)["odds"] >= 30.0)
                    & (race_df.head(5)["odds"] < 40.0)
                ]
                if len(top5_kimon) > 0:
                    posts = ", ".join(f"{int(p)}番({o:.1f}倍)" for p, o in
                                       zip(top5_kimon["post_number"], top5_kimon["odds"]))
                    st.warning(
                        f"⛔ **鬼門帯警告**: AI Top5 内に 30-40倍の馬がいます ({posts})。  \n"
                        f"v8 診断ではこの帯のROIは43〜69%（10-30倍と40-50倍の両側はプラス、ここだけ赤字）。"
                        f"買うなら 15-20倍 or 40-50倍 にずらすほうが期待値が高くなります。"
                    )
                st.caption(
                    "📊 オッズ帯凡例 (v8 診断 × 個別馬エッジ ≥0.10 でROI100%超え): "
                    "⛔本命過熱(<5倍) / ⚠️要注意(5-10倍) / "
                    "🟢エッジ帯(10-30倍 ∧ edge≥0.10, 15-20帯**ROI 104.1%**) / "
                    "○帯OKだがエッジ不足 (ROI <100%想定) / "
                    "⛔鬼門帯(30-40倍, 構造的赤字) / "
                    "🟢🟢高ROI(40-50倍 ∧ edge≥0.10, **ROI 128.9%**) / 🌪️大穴(>50倍)"
                )
                st.caption(
                    "🎯 順位帯凡例 (AI順位×単勝オッズ帯, **複勝券**で買った場合のROI): "
                    "🎯🎯= AI 2-3位 × 20-50倍 (**複勝ROI 130-145%**) / "
                    "🎯 = AI 1-2位 × 10-20倍 (**複勝ROI 101-108%**) / "
                    "（鬼門帯30-40倍は除外）"
                )
                st.caption(
                    "📐 確率2列の違い: **AI確率**=その馬が3着以内に入る独立確率（レース内合計≈300%）"
                    " / **相対%**=レース内で正規化した相対的な強さ（合計100%）"
                )
            else:
                st.caption("⚠️ オッズが取得できなかったため、期待値・エッジ・3連複BOX判定はスキップされています。")

            if has_imputed_col and display.get("weight_imputed", pd.Series([False])).any():
                st.caption("🌙 = 馬体重未発表のため前走値で補完（参考値）")

            # ── 3連複BOX推奨の表示 ──
            if sanren_rec.is_recommended:
                box_str = "-".join(str(n) for n in sorted(sanren_rec.top3_posts))
                st.success(
                    f"🎯 **{SANRENPUKU_STRATEGY}** 推奨: **{box_str}**（1点・BOX）  \n"
                    f"・頭数 {sanren_rec.head_count}頭 / 3位確率 {sanren_rec.top3_prob:.1%} "
                    f"/ 1番人気 {sanren_rec.fav_post}番 (AI Top3内)  \n"
                    f"・バックテスト実績 ROI {SANRENPUKU_ROI:.1f}% / 的中率 8.2%"
                )
            else:
                with st.expander("🎯 3連複BOX推奨: 対象外", expanded=False):
                    for r in sanren_rec.reasons:
                        st.caption(f"・{r}")

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
            "モデル", ["lightgbm_v8", "lightgbm_v6", "lightgbm_v7", "lightgbm_v5", "lightgbm_v4", "lightgbm_v3", "lightgbm_v2", "lightgbm_v1"], key="db_model",
            help="v8 推奨 (展開シナジー特徴量追加で v6 を全項目上回り)"
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
    # Plackett-Luce ベースの市場暗黙 top3 確率（旧式は本命帯で大幅過大評価）
    from src.betting.market_implied import add_market_top3_column
    target_df = add_market_top3_column(target_df, out_col="market_prob")
    target_df["market_prob"] = target_df["market_prob"].fillna((3.0 / target_df["odds"]).clip(upper=1.0))
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
            width='stretch',
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
        top_prob_db = race_df["pred_prob"].iloc[0]
        prob_pct_db = int(top_prob_db * 100)
        if top_prob_db >= 0.50:
            prob_badge_db = f"🔥 {prob_pct_db}%"
        elif top_prob_db >= 0.40:
            prob_badge_db = f"⭐ {prob_pct_db}%"
        else:
            prob_badge_db = f"　 {prob_pct_db}%"
        value_tag = f" / バリュー{n_value}件" if n_value > 0 else ""

        if expand_mode_db == "all":
            is_expanded = True
        elif expand_mode_db == "none":
            is_expanded = False
        else:
            is_expanded = n_value > 0

        with st.expander(
            f"**{venue} {race_num}R** — {surface}{distance}m ({condition})　　{prob_badge_db}{value_tag}",
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
                width='stretch',
                hide_index=True,
            )
