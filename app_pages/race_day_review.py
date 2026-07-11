"""レース日収支レビュー — 過去開催日の「全レース推奨買い目を購入していたら」をシミュレートする

目的:
    予測ページの推奨買い目を全レース購入した場合の収支を後追いで確認し、
    「どのレースを買えばよかったか」「自分が選んだレースは正解だったか」を
    反省するためのページ。

戦略:
    1) 単勝（v8 サンドイッチ）: edge >= 0.10 ∧ オッズが許容バンド内の馬に 100円
    2) 複勝（v8 サンドイッチ）: 同上 (実払戻 or 概算)
    3) 3連複BOX: 既存の sanrenpuku_filter ロジック (頭数>=13 ∧ 1番人気∈Top3 ∧ Top3確率>=0.20)

入力データ:
    - races / results / payouts テーブル (実際の着順と払戻)
    - 学習済みモデル (デフォルト v8)
"""

import sqlite3

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import DB_PATH
from src.betting.market_implied import add_market_top3_column
from src.betting.sanrenpuku_filter import (
    evaluate_race as evaluate_sanrenpuku,
)
from src.features.build_features import build_all_features
from src.model.train import get_available_features, load_model, prepare_dataset


# v8 サンドイッチ・プリセットと同じ
ODDS_PRESETS = {
    "🍞 v8 サンドイッチ 15-20+40-50 (推奨)": [(15.0, 20.0), (40.0, 50.0)],
    "🚫 30-40倍除外モード 10-30+40-50": [(10.0, 30.0), (40.0, 50.0)],
    "📐 中穴帯 10-50倍 (旧デフォルト)": [(10.0, 50.0)],
    "🎯 高ROI単勝 40-50倍のみ": [(40.0, 50.0)],
    "🛡️ 安定運用 15-20倍のみ": [(15.0, 20.0)],
    "❌ オッズ帯フィルタOFF (全馬対象)": [],
}

BET_UNIT = 100  # 1点あたりの賭金 (円)


# ──────────────────────────────────────────────
# キャッシュ
# ──────────────────────────────────────────────

@st.cache_resource
def _load_model_cached(name: str):
    return load_model(name)


@st.cache_data(ttl=600, show_spinner="特徴量を構築中...")
def _build_features_cached():
    return build_all_features()


# ──────────────────────────────────────────────
# メインページ
# ──────────────────────────────────────────────

def render():
    st.header("🪞 レース日収支レビュー")

    st.markdown(
        "過去のレース開催日を選んで、「**予測ページの推奨買い目を全レース購入していたらどうなったか**」を"
        "後追いシミュレーションします。  \n"
        "→ どのレースで買うべきだったか・自分の選択が正しかったかを反省するためのページです。"
    )

    if not DB_PATH.exists():
        st.error("データベースが存在しません。")
        return

    # ── 設定UI ──
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        # 過去のレース日を取得
        with sqlite3.connect(DB_PATH) as conn:
            dates = pd.read_sql_query(
                "SELECT DISTINCT date FROM races "
                "WHERE date IS NOT NULL "
                "ORDER BY date DESC LIMIT 200",
                conn,
            )["date"].tolist()
        if not dates:
            st.warning("DBにレースデータがありません。")
            return
        target_date = st.selectbox(
            "対象日", dates, index=0, key="rdr_date",
            help="開催日を選択。最新200日分から選べます。",
        )

    with col2:
        model_name = st.selectbox(
            "モデル",
            ["lightgbm_v8", "lightgbm_v6", "lightgbm_v7", "lightgbm_v5"],
            index=0,
            key="rdr_model",
            help="v8 推奨（オッズ帯フィルタの基準値も v8 診断結果準拠）",
        )

    # ── 予測ソース選択 ──
    from src.db.predictions import has_predictions
    saved_exists = has_predictions(str(target_date)[:10], model_name)
    pred_source = st.radio(
        "予測ソース",
        ["💾 当日保存した予測を使う", "🔄 今のモデルで再計算"],
        horizontal=True,
        index=0 if saved_exists else 1,
        key="rdr_pred_source",
        help=(
            "💾 当日保存: 予測ページで『この予測をDBに保存』したスナップショットを使用。"
            "DB更新の影響を受けず、当日に実際に出した予測と実結果を正確に突き合わせられます（推奨）。\n"
            "🔄 再計算: 今のDB・モデルでその場で予測し直す（当日と値がズレる場合あり）。"
        ),
    )
    use_saved = pred_source.startswith("💾")
    if use_saved and not saved_exists:
        st.warning(
            f"⚠️ {target_date} の保存済み予測がありません（モデル {model_name}）。"
            "予測ページで『💾 この予測をDBに保存』を実行するか、『🔄 今のモデルで再計算』を選んでください。"
        )
    elif saved_exists:
        st.caption(f"💾 {target_date} の保存済み予測が利用可能です（モデル {model_name}）。")

    with col3:
        edge_threshold = st.slider(
            "Edge 閾値 (単勝・複勝用)",
            min_value=0.0,
            max_value=0.30,
            value=0.10,
            step=0.05,
            key="rdr_edge",
            help="この値以上の市場乖離がある馬のみ単勝・複勝候補に含める",
        )

    col4, col5 = st.columns([2, 4])
    with col4:
        odds_preset_label = st.selectbox(
            "オッズ帯プリセット (Edge戦略の単勝・複勝)",
            list(ODDS_PRESETS.keys()),
            index=0,
            key="rdr_odds_preset",
        )
    with col5:
        strategies = st.multiselect(
            "シミュレートする戦略",
            ["単勝 (Edge)", "複勝 (Edge)", "3連複BOX", "💎 EV推奨 (単勝・複勝)"],
            default=["単勝 (Edge)", "複勝 (Edge)", "3連複BOX", "💎 EV推奨 (単勝・複勝)"],
            key="rdr_strategies",
            help=(
                "Edge戦略: pred_prob - market_prob >= Edge閾値 ∧ オッズ帯フィルタ通過の馬。\n"
                "EV推奨: bet_recommend ページと同じロジック (pred_prob × オッズ >= EV閾値)。\n"
                "馬連/ワイド/三連単 等のオッズ履歴はDBに無いためシミュレート対象外。"
            ),
        )

    # ── EV推奨用の追加パラメータ ──
    use_ev_mode = "💎 EV推奨 (単勝・複勝)" in strategies
    if use_ev_mode:
        with st.expander("💎 EV推奨の詳細設定", expanded=True):
            st.caption(
                "**bet_recommend ページの「期待値ベース」と同じ判定**: pred_prob × オッズ ≥ EV閾値 を満たす単勝・複勝に "
                "1レース最大N点まで均等買い (greedy_ev 配分方式)。  \n"
                "⚠️ 馬連/ワイド/馬単/三連複(BOX外)/三連単 は確定組合せオッズが DB に無いためシミュレートできません。"
            )
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                ev_threshold = st.slider(
                    "EV 閾値", min_value=1.0, max_value=2.0, value=1.10, step=0.05,
                    key="rdr_ev_threshold",
                    help="bet_recommend と同じデフォルト 1.10 (損益分岐+10%)",
                )
            with col_e2:
                ev_max_per_race = st.slider(
                    "1レースの最大点数", min_value=1, max_value=15, value=5, step=1,
                    key="rdr_ev_max",
                    help="EV順上位N点まで購入。bet_recommend の max_per_type と同じ概念",
                )
    else:
        ev_threshold = 1.10
        ev_max_per_race = 5

    if st.button("🔍 シミュレーション実行", type="primary", key="rdr_run"):
        _run_review(
            target_date, model_name, edge_threshold,
            ODDS_PRESETS[odds_preset_label], odds_preset_label,
            strategies,
            ev_threshold, ev_max_per_race,
            use_saved=use_saved,
        )


# ══════════════════════════════════════════════
# target_df ビルダー（予測ソース別）
# ══════════════════════════════════════════════

def _build_target_df_from_recompute(target_date: str, model_name: str):
    """今のDB・モデルで再計算して target_df を作る（従来方式・build_all_features 使用）"""
    try:
        model = _load_model_cached(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。")
        return None

    df = _build_features_cached()
    df = prepare_dataset(df)
    features = get_available_features(df)

    target_df = df[df["date"] == target_date].copy()
    if len(target_df) == 0:
        st.warning(f"{target_date} の出走データが見つかりません。")
        return None

    target_df["pred_prob"] = model.predict_proba(target_df[features])[:, 1]
    target_df = add_market_top3_column(target_df, out_col="market_prob")
    target_df["market_prob"] = target_df["market_prob"].fillna(
        (3.0 / target_df["odds"]).clip(upper=1.0)
    )
    target_df["edge"] = target_df["pred_prob"] - target_df["market_prob"]

    # 馬名・騎手名
    with sqlite3.connect(DB_PATH) as conn:
        names_df = pd.read_sql_query("SELECT horse_id, name FROM horses", conn)
        jockeys_df = pd.read_sql_query("SELECT jockey_id, name AS jockey_name FROM jockeys", conn)
    target_df = target_df.merge(names_df, on="horse_id", how="left")
    target_df = target_df.merge(jockeys_df, on="jockey_id", how="left")
    return target_df


def _build_target_df_from_saved(target_date: str, model_name: str):
    """当日保存した予測スナップショット + 実結果 から target_df を作る（高速・当日値そのまま）"""
    from src.db.predictions import load_predictions

    saved = load_predictions(str(target_date)[:10], model_name)
    if saved is None or len(saved) == 0:
        st.warning(
            f"⚠️ {target_date} の保存済み予測がありません（モデル {model_name}）。"
            "予測ページで『💾 この予測をDBに保存』を実行してください。"
        )
        return None

    race_ids = saved["race_id"].astype(str).unique().tolist()
    placeholders = ",".join("?" * len(race_ids))
    with sqlite3.connect(DB_PATH) as conn:
        results = pd.read_sql_query(
            f"SELECT race_id, post_number, finish_position, popularity "
            f"FROM results WHERE race_id IN ({placeholders})",
            conn, params=race_ids,
        )
        races = pd.read_sql_query(
            f"SELECT race_id, venue, surface, distance, condition "
            f"FROM races WHERE race_id IN ({placeholders})",
            conn, params=race_ids,
        )

    # 保存済み予測を土台に、実結果(finish_position)とレース情報をマージ
    target_df = saved.rename(columns={"horse_name": "name"}).copy()
    target_df["race_id"] = target_df["race_id"].astype(str)
    results["race_id"] = results["race_id"].astype(str)
    races["race_id"] = races["race_id"].astype(str)
    target_df = target_df.merge(results, on=["race_id", "post_number"], how="left")
    target_df = target_df.merge(races, on="race_id", how="left")

    if target_df["finish_position"].isna().all():
        st.warning(
            f"⚠️ {target_date} の実結果がまだDBにありません。"
            "「🔄 データ更新」でレース結果を取得してから再実行してください。"
        )
        return None
    return target_df


# ══════════════════════════════════════════════
# シミュレーション本体
# ══════════════════════════════════════════════

def _run_review(
    target_date: str,
    model_name: str,
    edge_threshold: float,
    allowed_bands: list,
    preset_label: str,
    strategies: list,
    ev_threshold: float = 1.10,
    ev_max_per_race: int = 5,
    use_saved: bool = False,
):
    # 1-2) 予測ソースに応じて target_df を構築
    if use_saved:
        target_df = _build_target_df_from_saved(target_date, model_name)
        if target_df is None:
            return
        st.info(f"💾 当日保存した予測（モデル {model_name}）を使用しています。")
    else:
        target_df = _build_target_df_from_recompute(target_date, model_name)
        if target_df is None:
            return
        st.info("🔄 今のDB・モデルで再計算した予測を使用しています（当日と値がズレる場合があります）。")

    # 3) 払戻データを一括取得
    race_ids = target_df["race_id"].unique().tolist()
    payouts = _load_payouts(race_ids)

    # 5) レースごとに評価
    race_rows = []
    detail_rows = []  # 全買い目の明細
    for race_id, race_df in target_df.groupby("race_id"):
        race_df = race_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        race_pay = payouts.get(race_id, {})

        venue = race_df["venue"].iloc[0]
        race_num = str(race_id)[-2:]
        head_count = len(race_df)
        surface = race_df["surface"].iloc[0]
        distance = int(race_df["distance"].iloc[0]) if pd.notna(race_df["distance"].iloc[0]) else 0

        race_summary = {
            "race_id": race_id,
            "venue": venue,
            "race_num": race_num,
            "label": f"{venue} {race_num}R ({surface}{distance}m)",
            "head_count": head_count,
            "stake": 0,
            "payout": 0,
            "n_bets": 0,
            "n_hits": 0,
        }

        # ── 単勝 (Edge ベース) ──
        if "単勝 (Edge)" in strategies:
            for _, row in race_df.iterrows():
                if not _passes_filter(row, edge_threshold, allowed_bands):
                    continue
                horse_no = int(row["post_number"])
                tansho_pay = race_pay.get("tansho", {}).get(str(horse_no), 0)
                hit = tansho_pay > 0
                stake = BET_UNIT
                got = tansho_pay if hit else 0  # payout は 100円単位 → そのまま (180=180円)
                race_summary["stake"] += stake
                race_summary["payout"] += got
                race_summary["n_bets"] += 1
                if hit:
                    race_summary["n_hits"] += 1
                detail_rows.append({
                    "race": race_summary["label"],
                    "戦略": "単勝(Edge)",
                    "対象": f"{horse_no}番 {row.get('name', '?')[:8]}",
                    "AI確率": f"{row['pred_prob']:.1%}",
                    "オッズ": f"{row['odds']:.1f}",
                    "判定値": f"Edge {row['edge']:+.3f}",
                    "投資": stake,
                    "配当": got,
                    "損益": got - stake,
                    "結果": "✅ 的中" if hit else "❌ 外れ",
                })

        # ── 複勝 (Edge ベース) ──
        if "複勝 (Edge)" in strategies:
            for _, row in race_df.iterrows():
                if not _passes_filter(row, edge_threshold, allowed_bands):
                    continue
                horse_no = int(row["post_number"])
                fukusho_pay = race_pay.get("fukusho", {}).get(str(horse_no), 0)
                hit = fukusho_pay > 0
                stake = BET_UNIT
                got = fukusho_pay if hit else 0
                race_summary["stake"] += stake
                race_summary["payout"] += got
                race_summary["n_bets"] += 1
                if hit:
                    race_summary["n_hits"] += 1
                detail_rows.append({
                    "race": race_summary["label"],
                    "戦略": "複勝(Edge)",
                    "対象": f"{horse_no}番 {row.get('name', '?')[:8]}",
                    "AI確率": f"{row['pred_prob']:.1%}",
                    "オッズ": f"{row['odds']:.1f}",
                    "判定値": f"Edge {row['edge']:+.3f}",
                    "投資": stake,
                    "配当": got,
                    "損益": got - stake,
                    "結果": "✅ 的中" if hit else "❌ 外れ",
                })

        # ── 💎 EV推奨 (単勝・複勝) ── bet_recommend ページと同じロジック
        if "💎 EV推奨 (単勝・複勝)" in strategies:
            # 各馬について 単勝EV / 複勝EV を計算し、EV>=閾値 のものを EV順に上位N点購入
            ev_candidates = []
            for _, row in race_df.iterrows():
                if pd.isna(row.get("odds")) or row["odds"] <= 0:
                    continue
                horse_no = int(row["post_number"])
                horse_label = f"{horse_no}番 {row.get('name', '?')[:8]}"

                # 単勝 EV = AI確率 × 単勝オッズ
                tansho_ev = float(row["pred_prob"]) * float(row["odds"])
                if tansho_ev >= ev_threshold:
                    tansho_pay = race_pay.get("tansho", {}).get(str(horse_no), 0)
                    ev_candidates.append({
                        "type": "単勝(EV)",
                        "horse_no": horse_no,
                        "label": horse_label,
                        "prob": row["pred_prob"],
                        "odds": row["odds"],
                        "ev": tansho_ev,
                        "payout": tansho_pay,
                    })

                # 複勝 EV = AI確率 × 推定複勝オッズ (単勝×0.3 を下限1.1 で近似)
                fukusho_odds_est = max(float(row["odds"]) * 0.3, 1.1)
                fukusho_ev = float(row["pred_prob"]) * fukusho_odds_est
                if fukusho_ev >= ev_threshold:
                    fukusho_pay = race_pay.get("fukusho", {}).get(str(horse_no), 0)
                    ev_candidates.append({
                        "type": "複勝(EV)",
                        "horse_no": horse_no,
                        "label": horse_label,
                        "prob": row["pred_prob"],
                        "odds": fukusho_odds_est,
                        "ev": fukusho_ev,
                        "payout": fukusho_pay,
                    })

            # EV 降順にソートして上位 N 点まで購入 (1レース上限)
            ev_candidates.sort(key=lambda x: x["ev"], reverse=True)
            for cand in ev_candidates[:ev_max_per_race]:
                hit = cand["payout"] > 0
                stake = BET_UNIT
                got = cand["payout"] if hit else 0
                race_summary["stake"] += stake
                race_summary["payout"] += got
                race_summary["n_bets"] += 1
                if hit:
                    race_summary["n_hits"] += 1
                detail_rows.append({
                    "race": race_summary["label"],
                    "戦略": cand["type"],
                    "対象": cand["label"],
                    "AI確率": f"{cand['prob']:.1%}",
                    "オッズ": f"{cand['odds']:.1f}",
                    "判定値": f"EV {cand['ev']:.2f}",
                    "投資": stake,
                    "配当": got,
                    "損益": got - stake,
                    "結果": "✅ 的中" if hit else "❌ 外れ",
                })

        # ── 3連複BOX (推奨条件を満たすレースのみ) ──
        if "3連複BOX" in strategies:
            sanren_rec = evaluate_sanrenpuku(race_df)
            if sanren_rec.is_recommended:
                box_combo = "-".join(str(n) for n in sorted(sanren_rec.top3_posts))
                # 払戻参照: payouts.sanrenpuku[combo_key] (3頭ソート済キー)
                pay = race_pay.get("sanrenpuku", {}).get(box_combo, 0)
                hit = pay > 0
                stake = BET_UNIT  # 3連複BOX 1点
                got = pay if hit else 0
                race_summary["stake"] += stake
                race_summary["payout"] += got
                race_summary["n_bets"] += 1
                if hit:
                    race_summary["n_hits"] += 1
                detail_rows.append({
                    "race": race_summary["label"],
                    "戦略": "3連複BOX",
                    "対象": f"BOX {box_combo}",
                    "AI確率": f"{sanren_rec.top3_prob:.1%} (3位)",
                    "オッズ": "-",
                    "判定値": f"head≥{sanren_rec.head_count}",
                    "投資": stake,
                    "配当": got,
                    "損益": got - stake,
                    "結果": "✅ 的中" if hit else "❌ 外れ",
                })

        race_summary["profit"] = race_summary["payout"] - race_summary["stake"]
        race_summary["roi"] = (
            race_summary["payout"] / race_summary["stake"] * 100
            if race_summary["stake"] > 0 else 0
        )
        race_rows.append(race_summary)

    # 6) 表示
    _display_results(target_date, race_rows, detail_rows, preset_label, strategies)

    # 7) 予測 vs 実結果 全馬リスト
    _display_prediction_vs_result(target_date, target_df, model_name)


def _display_prediction_vs_result(target_date: str, target_df: pd.DataFrame, model_name: str):
    """予測と実レース結果を馬名・馬番込みで全馬リスト化して表示・CSV出力する"""
    st.divider()
    st.subheader("📋 予測 vs 実結果 全馬リスト")
    st.caption(
        f"対象日 {target_date} の全出走馬について、AI予測 (順位・確率・edge) と"
        f"実際の着順を並べたリストです。CSV でダウンロードできます。"
    )

    rows = []
    for race_id, race_df in target_df.groupby("race_id"):
        race_df = race_df.sort_values("pred_prob", ascending=False).reset_index(drop=True)
        venue = race_df["venue"].iloc[0]
        race_num = str(race_id)[-2:]
        surface = race_df["surface"].iloc[0]
        distance = int(race_df["distance"].iloc[0]) if pd.notna(race_df["distance"].iloc[0]) else 0
        race_label = f"{venue}{race_num}R"
        race_cond = f"{surface}{distance}m"
        race_prob_sum = race_df["pred_prob"].sum()

        for ai_rank, (_, row) in enumerate(race_df.iterrows(), start=1):
            finish = row.get("finish_position")
            finish_int = int(finish) if pd.notna(finish) and finish > 0 else None
            pred_prob = float(row["pred_prob"])
            norm_prob = pred_prob / race_prob_sum if race_prob_sum > 0 else 0.0

            # 予測精度の判定マーク
            is_top3_actual = finish_int is not None and finish_int <= 3
            in_ai_top3 = ai_rank <= 3
            if is_top3_actual and in_ai_top3:
                hit_mark = "◎ 的中"     # AI Top3 かつ 実際も3着以内
            elif is_top3_actual and not in_ai_top3:
                hit_mark = "▲ 見逃し"   # AI Top3 外だが 3着以内に来た
            elif not is_top3_actual and in_ai_top3:
                hit_mark = "✗ 外れ"     # AI Top3 だが 4着以下
            else:
                hit_mark = ""            # AI Top3 外で 4着以下 (予測通りの凡走)

            rows.append({
                "日付": target_date,
                "競馬場": venue,
                "レース": race_label,
                "条件": race_cond,
                "馬番": int(row["post_number"]) if pd.notna(row["post_number"]) else None,
                "馬名": row.get("name", "?"),
                "騎手": row.get("jockey_name", ""),
                "AI順位": ai_rank,
                "AI確率": round(pred_prob * 100, 1),
                "相対%": round(norm_prob * 100, 1),
                "単勝オッズ": round(float(row["odds"]), 1) if pd.notna(row.get("odds")) else None,
                "人気": int(row["popularity"]) if "popularity" in row.index and pd.notna(row.get("popularity")) else None,
                "Edge": round(float(row["edge"]), 3) if pd.notna(row.get("edge")) else None,
                "実着順": finish_int,
                "判定": hit_mark,
            })

    result_df = pd.DataFrame(rows)

    # ── サマリー: AI Top3 の的中状況 ──
    n_races = result_df["レース"].nunique()
    ai_top3 = result_df[result_df["AI順位"] <= 3]
    n_hit = (ai_top3["判定"] == "◎ 的中").sum()
    n_miss = (ai_top3["判定"] == "✗ 外れ").sum()
    n_overlooked = (result_df["判定"] == "▲ 見逃し").sum()

    cols = st.columns(4)
    cols[0].metric("対象レース", f"{n_races}R")
    cols[1].metric("AI Top3 的中", f"{n_hit}頭", help="AI Top3 のうち実際に3着以内に来た数")
    cols[2].metric("AI Top3 外れ", f"{n_miss}頭", help="AI Top3 のうち4着以下だった数")
    cols[3].metric("見逃し", f"{n_overlooked}頭", help="AI Top3 外から3着以内に来た数")

    # ── AI順位 Top5 の成績サマリー (1日単位) ──
    # 「1着-2着-3着-着外 複勝率xx%」の競馬式成績表記でまとめる
    def _rank_summary(df_part: pd.DataFrame) -> list[dict]:
        """AI順位1〜5位の成績 (1着-2着-3着-着外 / 複勝率 / 勝率) を集計"""
        out = []
        for rank in range(1, 6):
            sub = df_part[df_part["AI順位"] == rank]
            if len(sub) == 0:
                continue
            finishes = sub["実着順"]
            n1 = int((finishes == 1).sum())
            n2 = int((finishes == 2).sum())
            n3 = int((finishes == 3).sum())
            n_out = int(len(sub) - n1 - n2 - n3)
            n_total = len(sub)
            out.append({
                "AI順位": f"{rank}位",
                "成績": f"{n1}-{n2}-{n3}-{n_out}",
                "複勝率": f"{(n1 + n2 + n3) / n_total * 100:.1f}%",
                "勝率": f"{n1 / n_total * 100:.1f}%",
                "出走数": n_total,
            })
        return out

    def _rank_one_liner(summary: list[dict]) -> str:
        return "  /  ".join(
            f"AI{r['AI順位']}: {r['成績']} 複勝率{r['複勝率']}" for r in summary
        )

    st.markdown("#### 🏅 AI順位別 成績 (本日)")
    summary_rows = _rank_summary(result_df)
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)
        # 1行テキストでも出す（コピペ用）
        st.caption(f"📋 {target_date}: {_rank_one_liner(summary_rows)}")

    # ── 競馬場別 AI順位別成績 ──
    venues = sorted(result_df["競馬場"].dropna().unique().tolist())
    if len(venues) >= 2:
        st.markdown("#### 🏇 競馬場別 AI順位別成績")
        venue_tabs = st.tabs(venues)
        for tab, venue in zip(venue_tabs, venues):
            with tab:
                vdf = result_df[result_df["競馬場"] == venue]
                n_races_v = vdf["レース"].nunique()
                v_top3 = vdf[vdf["AI順位"] <= 3]
                v_hit = (v_top3["判定"] == "◎ 的中").sum()
                v_miss = (v_top3["判定"] == "✗ 外れ").sum()
                v_overlook = (vdf["判定"] == "▲ 見逃し").sum()
                vc = st.columns(4)
                vc[0].metric("レース数", f"{n_races_v}R")
                vc[1].metric("AI Top3 的中", f"{v_hit}頭")
                vc[2].metric("AI Top3 外れ", f"{v_miss}頭")
                vc[3].metric("見逃し", f"{v_overlook}頭")

                v_summary = _rank_summary(vdf)
                if v_summary:
                    st.dataframe(pd.DataFrame(v_summary), width='stretch', hide_index=True)
                    st.caption(f"📋 {target_date} {venue}: {_rank_one_liner(v_summary)}")
                else:
                    st.info("集計対象データがありません。")

    st.divider()

    # ── レース別タブ形式表示 (予測ページと同じ expander) ──
    st.markdown("#### 📋 レース別 予測 vs 結果")

    # 一括展開/畳む
    btn_e, btn_c, _ = st.columns([1, 1, 4])
    with btn_e:
        if st.button("📂 全て展開", key="rdr_pvr_expand_all"):
            st.session_state["rdr_pvr_expand"] = "all"
            st.rerun()
    with btn_c:
        if st.button("📁 全て畳む", key="rdr_pvr_collapse_all"):
            st.session_state["rdr_pvr_expand"] = "none"
            st.rerun()
    expand_mode = st.session_state.get("rdr_pvr_expand", "auto")

    for race_label in result_df["レース"].unique():
        race_view = result_df[result_df["レース"] == race_label].sort_values("AI順位")
        race_cond = race_view["条件"].iloc[0]

        # レース単位の的中状況をラベルに反映
        n_hit_r = (race_view["判定"] == "◎ 的中").sum()
        n_overlook_r = (race_view["判定"] == "▲ 見逃し").sum()
        if n_hit_r == 3:
            badge = "🎯 3/3 的中"
        elif n_hit_r > 0:
            badge = f"◎ {n_hit_r}/3 的中"
        else:
            badge = "✗ 0/3"
        if n_overlook_r > 0:
            badge += f" ▲{n_overlook_r}"

        # 展開状態: auto = 全的中 or 見逃しありレースを展開
        if expand_mode == "all":
            is_expanded = True
        elif expand_mode == "none":
            is_expanded = False
        else:
            is_expanded = (n_hit_r == 3) or (n_overlook_r > 0)

        with st.expander(f"**{race_label}** — {race_cond}　　{badge}", expanded=is_expanded):
            disp = race_view[[
                "馬番", "馬名", "騎手", "AI順位", "AI確率", "相対%",
                "単勝オッズ", "人気", "Edge", "実着順", "判定",
            ]].copy()
            st.dataframe(disp, width='stretch', hide_index=True)

    # ── CSV ダウンロード ──
    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 CSV ダウンロード (全馬)",
        data=csv_bytes,
        file_name=f"prediction_vs_result_{target_date}_{model_name}.csv",
        mime="text/csv",
        key="rdr_pvr_csv",
    )


# ══════════════════════════════════════════════
# 表示
# ══════════════════════════════════════════════

def _display_results(
    target_date: str,
    race_rows: list,
    detail_rows: list,
    preset_label: str,
    strategies: list,
):
    if not race_rows:
        st.warning("対象日にデータがありません。")
        return

    races_df = pd.DataFrame(race_rows)
    bets_df = pd.DataFrame(detail_rows)

    total_stake = races_df["stake"].sum()
    total_payout = races_df["payout"].sum()
    profit = total_payout - total_stake
    n_bets = races_df["n_bets"].sum()
    n_hits = races_df["n_hits"].sum()
    roi = (total_payout / total_stake * 100) if total_stake > 0 else 0
    hit_rate = (n_hits / n_bets * 100) if n_bets > 0 else 0

    n_races_with_bets = (races_df["stake"] > 0).sum()
    n_profitable_races = (races_df["profit"] > 0).sum()
    n_losing_races = ((races_df["stake"] > 0) & (races_df["profit"] < 0)).sum()
    n_break_even = ((races_df["stake"] > 0) & (races_df["profit"] == 0)).sum()

    # ── サマリー ──
    st.subheader(f"📊 {target_date} 収支サマリー")
    st.caption(f"戦略: {' + '.join(strategies)}  |  オッズ帯: {preset_label}")

    cols = st.columns(5)
    cols[0].metric("対象レース", f"{n_races_with_bets} / {len(races_df)}R", help="買い目が出たレース数 / 総レース数")
    cols[1].metric("買い目", f"{int(n_bets)}点", f"的中{int(n_hits)}点")
    cols[2].metric("投資", f"{int(total_stake):,}円")
    cols[3].metric("配当", f"{int(total_payout):,}円")
    cols[4].metric(
        "損益",
        f"{int(profit):+,}円",
        f"ROI {roi:.1f}%",
        delta_color="normal" if profit >= 0 else "inverse",
    )

    cols2 = st.columns(4)
    cols2[0].metric("的中率", f"{hit_rate:.1f}%")
    cols2[1].metric("プラス収支レース", f"{n_profitable_races}R", f"{n_profitable_races / max(n_races_with_bets,1) * 100:.0f}%")
    cols2[2].metric("マイナス収支レース", f"{n_losing_races}R")
    cols2[3].metric("収支ゼロ", f"{n_break_even}R")

    st.divider()

    # ── レースごとの収支 (ソート可能) ──
    st.subheader("🏁 レース別収支（プラス順）")

    if n_races_with_bets == 0:
        st.info(
            "推奨条件を満たすレースがありませんでした。  \n"
            "Edge閾値を下げる、オッズ帯を広げる、戦略を増やすなどを試してください。"
        )
    else:
        active_races = races_df[races_df["stake"] > 0].copy()
        active_races = active_races.sort_values("profit", ascending=False)
        active_races["収支マーク"] = active_races["profit"].apply(
            lambda p: "🟢" if p > 0 else "🔴" if p < 0 else "⚪"
        )
        active_races["損益"] = active_races["profit"].apply(lambda x: f"{int(x):+,}円")
        active_races["投資"] = active_races["stake"].apply(lambda x: f"{int(x):,}円")
        active_races["配当"] = active_races["payout"].apply(lambda x: f"{int(x):,}円")
        active_races["ROI"] = active_races["roi"].apply(lambda x: f"{x:.0f}%")
        active_races["買い目"] = active_races["n_bets"].astype(int).astype(str) + "点"
        active_races["的中"] = active_races["n_hits"].astype(int).astype(str) + "点"
        active_races["レース"] = active_races["label"]

        st.dataframe(
            active_races[["収支マーク", "レース", "買い目", "的中", "投資", "配当", "損益", "ROI"]],
            width='stretch',
            hide_index=True,
        )

        # 反省ポイント
        if n_profitable_races > 0:
            best = active_races.iloc[0]
            st.success(
                f"🏆 **本日の最高収支レース**: {best['レース']} → "
                f"{best['損益']} (投資 {best['投資']} → 配当 {best['配当']})"
            )
        if n_losing_races > 0:
            worst = active_races.iloc[-1]
            st.error(
                f"💸 **本日の最大損失レース**: {worst['レース']} → "
                f"{worst['損益']} (投資 {worst['投資']} → 配当 {worst['配当']})  \n"
                f"このレースを避けられていれば、全体損益は {int(worst['profit'] * -1):+,}円 改善していました。"
            )

    st.divider()

    # ── 累積損益グラフ ──
    if n_races_with_bets > 0:
        st.subheader("📈 累積損益の推移（レース順）")
        active_in_order = races_df[races_df["stake"] > 0].copy()
        active_in_order["cum_profit"] = active_in_order["profit"].cumsum()

        fig = go.Figure()
        fig.add_scatter(
            x=active_in_order["label"],
            y=active_in_order["cum_profit"],
            mode="lines+markers",
            line=dict(width=3, color="#4ECDC4"),
            marker=dict(size=8, color=[
                "#4ECDC4" if p > 0 else "#FF6B6B" if p < 0 else "#999"
                for p in active_in_order["profit"]
            ]),
            text=active_in_order["profit"].apply(lambda x: f"{int(x):+,}"),
            textposition="top center",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title="レース", yaxis_title="累積損益 (円)",
            height=400, margin=dict(t=20),
        )
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    st.divider()

    # ── 戦略別サマリー ──
    if not bets_df.empty:
        st.subheader("📋 戦略別サマリー")
        strat_summary = bets_df.groupby("戦略").agg(
            買い目=("対象", "count"),
            的中=("配当", lambda s: (s > 0).sum()),
            投資=("投資", "sum"),
            配当=("配当", "sum"),
            損益=("損益", "sum"),
        ).reset_index()
        strat_summary["的中率"] = (strat_summary["的中"] / strat_summary["買い目"] * 100).round(1).astype(str) + "%"
        strat_summary["ROI"] = (strat_summary["配当"] / strat_summary["投資"] * 100).round(1).astype(str) + "%"
        strat_summary["投資"] = strat_summary["投資"].apply(lambda x: f"{int(x):,}円")
        strat_summary["配当"] = strat_summary["配当"].apply(lambda x: f"{int(x):,}円")
        strat_summary["損益"] = strat_summary["損益"].apply(lambda x: f"{int(x):+,}円")
        st.dataframe(
            strat_summary[["戦略", "買い目", "的中", "的中率", "投資", "配当", "損益", "ROI"]],
            width='stretch', hide_index=True,
        )

        # ── 全買い目明細 ──
        with st.expander("📜 全買い目明細を見る", expanded=False):
            display_cols = ["race", "戦略", "対象", "AI確率", "オッズ", "判定値", "投資", "配当", "損益", "結果"]
            disp = bets_df[display_cols].copy()
            disp["投資"] = disp["投資"].apply(lambda x: f"{int(x):,}円")
            disp["配当"] = disp["配当"].apply(lambda x: f"{int(x):,}円")
            disp["損益"] = disp["損益"].apply(lambda x: f"{int(x):+,}円")
            disp = disp.rename(columns={"race": "レース"})
            st.dataframe(disp, width='stretch', hide_index=True)


# ══════════════════════════════════════════════
# ヘルパー
# ══════════════════════════════════════════════

def _passes_filter(row, edge_threshold: float, allowed_bands: list) -> bool:
    """単勝・複勝の対象判定: Edge閾値 ∧ オッズ帯フィルタ"""
    if pd.isna(row.get("edge")) or pd.isna(row.get("odds")):
        return False
    if row["edge"] < edge_threshold:
        return False
    o = float(row["odds"])
    if not allowed_bands:  # 空リスト = フィルタOFF
        return True
    return any(lo <= o <= hi for lo, hi in allowed_bands)


def _load_payouts(race_ids: list) -> dict:
    """payouts テーブルから対象レースの全払戻を辞書化して返す。

    戻り値: {race_id: {bet_type: {key: payout}}}
        bet_type = "tansho", "fukusho", "sanrenpuku" など
        key = 単勝/複勝なら馬番(str), 組合せなら "1-3-7" のソート済キー
    """
    if not race_ids:
        return {}

    placeholders = ",".join("?" * len(race_ids))
    query = f"""
        SELECT race_id, bet_type, horse_number, horse_numbers, payout
        FROM payouts
        WHERE race_id IN ({placeholders})
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, race_ids).fetchall()

    out: dict = {}
    for race_id, bet_type, horse_no, horse_nos, payout in rows:
        out.setdefault(race_id, {}).setdefault(bet_type, {})
        if bet_type in ("tansho", "fukusho") and horse_no is not None:
            key = str(int(horse_no))
        elif horse_nos:
            # ソート済キーに正規化 ("3-1-7" → "1-3-7")
            try:
                parts = sorted(int(x) for x in str(horse_nos).split("-"))
                key = "-".join(str(p) for p in parts)
            except ValueError:
                key = str(horse_nos)
        else:
            continue
        out[race_id][bet_type][key] = int(payout)
    return out
