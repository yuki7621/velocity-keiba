"""買い目推奨ページ — 現在オッズ × AI予測から期待値ベースで買い目を提案"""

import streamlit as st
import pandas as pd

from src.scraper.odds import fetch_all_odds
from src.betting.recommend import compute_all_bets, allocate_budget


def render():
    st.header("💰 買い目推奨")

    st.markdown(
        "「予測」ページで予測済みのレースから対象を選び、"
        "現在オッズを取得して期待値ベースの買い目を提案します。"
    )

    # ── 予測結果の確認 ──
    if "prerace_results" not in st.session_state or st.session_state["prerace_results"] is None:
        st.warning(
            "⚠️ 予測結果がありません。先に「📊 予測」ページで"
            "「🔮 出馬表を取得して予測」を実行してください。"
        )
        return

    df: pd.DataFrame = st.session_state["prerace_results"]

    # ── レース選択 ──
    st.subheader("🎯 対象レース選択")

    race_options = []
    race_meta = {}
    for race_id in sorted(df["race_id"].unique()):
        race_df = df[df["race_id"] == race_id]
        venue = race_df["venue"].iloc[0] if pd.notna(race_df["venue"].iloc[0]) else "?"
        race_num = str(race_id)[-2:]
        title = race_df["race_title"].iloc[0] if "race_title" in race_df.columns else ""
        label = f"{venue} {race_num}R"
        if title:
            label += f" — {title}"
        race_options.append(label)
        race_meta[label] = race_id

    selected_label = st.selectbox("レース", race_options, key="bet_race_select")
    selected_race_id = race_meta[selected_label]
    race_df = df[df["race_id"] == selected_race_id].copy()

    # ── パラメータ ──
    st.subheader("⚙️ 設定")
    col1, col2, col3 = st.columns(3)

    with col1:
        budget = st.number_input(
            "予算 (円)",
            min_value=100,
            max_value=1000000,
            value=1000,
            step=100,
            key="bet_budget",
        )

    with col2:
        ev_threshold = st.slider(
            "EV閾値（最低期待値）",
            min_value=1.0,
            max_value=2.0,
            value=1.10,
            step=0.05,
            key="bet_ev_threshold",
            help="この値以上の期待値を持つ買い目のみ候補にします（1.0 = 損益分岐点）",
        )

    with col3:
        strategy = st.selectbox(
            "配分方式",
            ["greedy_ev", "half_kelly", "kelly", "proportional_ev"],
            format_func=lambda x: {
                "greedy_ev": "EV順に均等買い（100円ずつ）",
                "half_kelly": "Half Kelly（推奨・安定型）",
                "kelly": "Full Kelly（攻撃型）",
                "proportional_ev": "EVに比例配分",
            }[x],
            key="bet_strategy",
        )

    # 券種選択
    bet_types = st.multiselect(
        "対象券種",
        ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"],
        default=["単勝", "複勝", "馬連", "ワイド", "三連複"],
        key="bet_types",
    )

    max_per_type = st.slider(
        "券種ごとの上位候補数",
        min_value=1,
        max_value=20,
        value=5,
        help="各券種でEV上位N件まで購入候補に含めます",
        key="bet_max_per_type",
    )

    # 的中率優先フィルター
    with st.expander("🎯 的中率優先フィルター（詳細設定）"):
        st.caption("AIの予測確率が高い馬だけに絞ることで、的中率を優先できます。回収率は下がる場合があります。")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            use_prob_filter = st.checkbox(
                "予測確率フィルターを使用",
                value=False,
                key="bet_use_prob_filter",
                help="チェックすると、Edgeに加えてAI予測確率でも絞り込みます",
            )
        with col_f2:
            min_pred_prob = st.slider(
                "最低予測確率（top3）",
                min_value=0.10,
                max_value=0.80,
                value=0.35,
                step=0.05,
                key="bet_min_pred_prob",
                help="この確率以上の馬のみ買い目候補に含めます",
                disabled=not use_prob_filter,
            )

    # ── 実行ボタン ──
    if st.button("🔄 オッズ取得 & 買い目計算", type="primary", key="btn_calc_bets"):
        _calculate_and_display(
            selected_race_id, race_df, budget, ev_threshold,
            bet_types, max_per_type, strategy,
            min_pred_prob=min_pred_prob if use_prob_filter else None,
        )

    # ── 既存の計算結果があれば表示 ──
    elif (
        "bet_results" in st.session_state
        and st.session_state.get("bet_race_id") == selected_race_id
    ):
        _display_bet_results(st.session_state["bet_results"], budget)
        if st.session_state.get("bet_prob_filter") is not None:
            st.caption(f"🎯 的中率優先フィルター適用中: 予測確率 {st.session_state['bet_prob_filter']:.0%} 以上")


def _calculate_and_display(
    race_id: str,
    race_df: pd.DataFrame,
    budget: int,
    ev_threshold: float,
    bet_types: list,
    max_per_type: int,
    strategy: str,
    min_pred_prob: float | None = None,
):
    """オッズ取得 → EV計算 → 予算配分 → 表示"""

    # 1) AI top3確率を辞書化（pred_probフィルター適用）
    if min_pred_prob is not None:
        filtered_df = race_df[race_df["pred_prob"] >= min_pred_prob]
        if len(filtered_df) == 0:
            st.warning(f"予測確率 {min_pred_prob:.0%} 以上の馬がいません。閾値を下げてください。")
            filtered_df = race_df  # フォールバック
        else:
            st.info(f"🎯 的中率優先: 予測確率 {min_pred_prob:.0%} 以上の {len(filtered_df)} 頭に絞り込み")
    else:
        filtered_df = race_df

    top3_probs = {
        int(row["post_number"]): float(row["pred_prob"])
        for _, row in filtered_df.iterrows()
        if pd.notna(row.get("post_number")) and pd.notna(row.get("pred_prob"))
    }

    if not top3_probs:
        st.error("AI予測結果が空です。")
        return

    # 2) オッズ取得
    with st.spinner(f"📡 {race_id} のオッズを取得中..."):
        odds = fetch_all_odds(race_id)

    has_any_odds = any(len(odds.get(k, {})) > 0 for k in odds)
    if not has_any_odds:
        st.error(
            "⚠️ オッズが取得できませんでした。\n\n"
            "考えられる原因:\n"
            "- レース発売前 / 発売終了\n"
            "- レースIDが正しくない\n"
            "- netkeibaのAPI変更"
        )
        return

    # 3) 全券種EV計算
    all_bets_df = compute_all_bets(top3_probs, odds, enabled_types=bet_types)

    if len(all_bets_df) == 0:
        st.warning("計算対象の買い目がありませんでした。")
        return

    # 4) 予算配分
    allocated = allocate_budget(
        all_bets_df,
        budget=budget,
        ev_threshold=ev_threshold,
        max_per_type=max_per_type,
        strategy=strategy,
    )

    # session_stateに保存
    st.session_state["bet_results"] = {
        "all_bets": all_bets_df,
        "allocated": allocated,
        "horse_names": _build_horse_name_map(race_df),
        "top3_probs": top3_probs,
    }
    st.session_state["bet_race_id"] = race_id

    _display_bet_results(st.session_state["bet_results"], budget)


def _build_horse_name_map(race_df: pd.DataFrame) -> dict:
    """馬番 → 馬名のマップ"""
    return {
        int(row["post_number"]): row.get("horse_name", "?")
        for _, row in race_df.iterrows()
        if pd.notna(row.get("post_number"))
    }


def _format_combo_with_names(combo: str, horse_names: dict) -> str:
    """組み合わせ文字列を馬番＋馬名で表示"""
    parts = combo.split("-")
    formatted = []
    for p in parts:
        try:
            n = int(p)
            name = horse_names.get(n, "")
            formatted.append(f"{n}{name[:6]}" if name else str(n))
        except ValueError:
            formatted.append(p)
    return " - ".join(formatted)


def _display_bet_results(results: dict, budget: int):
    """計算結果を表示"""
    all_bets_df: pd.DataFrame = results["all_bets"]
    allocated: pd.DataFrame = results["allocated"]
    horse_names: dict = results["horse_names"]

    st.divider()

    # ── 推奨買い目（予算内） ──
    st.subheader("💎 推奨買い目（予算内）")

    if len(allocated) == 0:
        st.warning(
            f"⚠️ EV閾値以上の買い目が予算内に見つかりませんでした。\n\n"
            "EV閾値を下げるか、対象券種を増やしてみてください。"
        )
    else:
        total_stake = allocated["stake"].sum()

        # ── 排他的買い目（同券種に複数点）を検出 ──
        # 単勝・複勝は同一レース内で複数点あっても1点しか的中しない
        # 買い目推奨は同一レース内で実行されるため race_id は不要
        EXCLUSIVE_TYPES = {"単勝", "複勝"}
        excl_mask = allocated["type"].isin(EXCLUSIVE_TYPES)
        excl_df = allocated[excl_mask].copy()
        indep_df = allocated[~excl_mask].copy()

        # 排他グループ = 券種ごと（単勝同士・複勝同士は排他）
        excl_df["_excl_group"] = excl_df["type"]

        # 独立分の期待払戻（馬連・ワイドなど）
        indep_expected = indep_df["expected_return"].sum()

        # 排他分：同グループ内の最大期待払戻（最良シナリオ）・最小（最悪シナリオ）
        excl_group_max = excl_df.groupby("_excl_group")["expected_return"].max().sum()
        excl_group_min = excl_df.groupby("_excl_group")["expected_return"].min().sum()

        # 期待値ベースの合計（数学的に正しい値）
        total_expected = allocated["expected_return"].sum()
        roi = (total_expected / total_stake - 1) * 100 if total_stake > 0 else 0

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("購入点数", f"{len(allocated)}点")
        col_b.metric("購入金額", f"{int(total_stake):,}円", f"予算 {budget:,}円")
        col_c.metric("期待払戻（期待値）", f"{int(total_expected):,}円")
        col_d.metric("期待ROI", f"{roi:+.1f}%")

        # 排他的買い目がある場合はシナリオ別払戻を追加表示
        has_exclusive = len(excl_df) > 0
        excl_has_multiple = (excl_df.groupby("_excl_group").size() > 1).any() if has_exclusive else False

        if has_exclusive and excl_has_multiple:
            with st.expander("💡 シナリオ別払戻の見方", expanded=True):
                st.caption(
                    "単勝・複勝を同レースで複数点購入している場合、"
                    "1点しか的中しないため実際の払戻はシナリオにより異なります。"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "独立分の期待払戻",
                    f"{int(indep_expected):,}円",
                    help="馬連・ワイドなど、互いに独立した買い目の期待払戻合計"
                )
                c2.metric(
                    "排他分（最良シナリオ）",
                    f"+{int(excl_group_max):,}円",
                    help="単勝・複勝で、最もオッズが高い馬が的中した場合"
                )
                c3.metric(
                    "排他分（最悪シナリオ）",
                    f"+{int(excl_group_min):,}円",
                    help="単勝・複勝で、最もオッズが低い馬が的中した場合"
                )

        # 表示用整形
        disp = allocated.copy()
        disp["買い目"] = disp["combo"].apply(lambda c: _format_combo_with_names(c, horse_names))
        disp["AI確率"] = disp["prob"].apply(lambda x: f"{x:.2%}")
        disp["オッズ"] = disp["odds"].apply(lambda x: f"{x:.1f}")
        disp["期待値"] = disp["ev"].apply(lambda x: f"{x:.2f}")
        disp["購入"] = disp["stake"].apply(lambda x: f"{int(x):,}円")
        disp["期待払戻"] = disp["expected_return"].apply(lambda x: f"{int(x):,}円")

        # 排他的買い目に注記を付ける（券種ごとに複数点ある場合）
        excl_type_counts = (
            disp[disp["type"].isin(EXCLUSIVE_TYPES)]
            .groupby("type").size()
        )
        multi_excl_types = set(excl_type_counts[excl_type_counts > 1].index)

        def _mark_exclusive(row):
            if row["type"] in multi_excl_types:
                return row["期待払戻"] + " ※排他"
            return row["期待払戻"]

        disp["期待払戻"] = disp.apply(_mark_exclusive, axis=1)

        st.dataframe(
            disp[["type", "買い目", "AI確率", "オッズ", "期待値", "購入", "期待払戻"]].rename(
                columns={"type": "券種"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        if has_exclusive and excl_has_multiple:
            st.caption("※排他 = 同一レースで複数点購入している単勝・複勝。1点しか的中しません。")

    st.divider()

    # ── 全候補（参考） ──
    st.subheader("📊 全候補一覧（EV順）")

    type_filter = st.multiselect(
        "券種フィルタ",
        sorted(all_bets_df["type"].unique()),
        default=sorted(all_bets_df["type"].unique()),
        key="bet_type_filter",
    )

    filtered = all_bets_df[all_bets_df["type"].isin(type_filter)].head(50).copy()
    filtered["買い目"] = filtered["combo"].apply(lambda c: _format_combo_with_names(c, horse_names))
    filtered["AI確率"] = filtered["prob"].apply(lambda x: f"{x:.2%}")
    filtered["オッズ"] = filtered["odds"].apply(lambda x: f"{x:.1f}")
    filtered["期待値"] = filtered["ev"].apply(lambda x: f"{x:.2f}")

    # EVが1以上は緑、未満は灰色
    def _ev_emoji(ev):
        if ev >= 1.3:
            return "🔥"
        elif ev >= 1.1:
            return "⭐"
        elif ev >= 1.0:
            return "🟢"
        return "⚪"

    filtered["評価"] = filtered["ev"].apply(_ev_emoji)

    st.dataframe(
        filtered[["評価", "type", "買い目", "AI確率", "オッズ", "期待値"]].rename(
            columns={"type": "券種"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "🔥 EV≥1.3 / ⭐ EV≥1.1 / 🟢 EV≥1.0 / ⚪ EV<1.0\n\n"
        "※ 確率は Plackett-Luce モデル（AI3着内確率を強さとする順位モデル）による近似値です。"
    )
