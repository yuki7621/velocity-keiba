"""note記事作成ページ — 予測結果から会場別の配布用記事(Markdown)を生成する"""

from datetime import datetime

import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def render():
    st.header("📝 note記事作成")

    st.markdown(
        "「📊 予測」ページで予測済みのレースから、**会場別にまとめた配布用記事**を"
        "Markdown形式で生成します。note.com にそのまま貼り付け可能です。"
    )

    if "prerace_results" not in st.session_state or st.session_state["prerace_results"] is None:
        st.warning(
            "⚠️ 予測結果がありません。先に「📊 予測」ページで"
            "「🔮 出馬表を取得して予測」を実行してください。"
        )
        return

    df: pd.DataFrame = st.session_state["prerace_results"]
    target_date = st.session_state.get("prerace_results_date", "")
    model_name = st.session_state.get("prerace_results_model", "")

    # ── 設定 ──
    st.subheader("⚙️ 記事設定")

    col1, col2 = st.columns(2)
    with col1:
        venues_all = sorted(df["venue"].dropna().unique().tolist())
        venues = st.multiselect(
            "掲載する会場",
            venues_all,
            default=venues_all,
            key="note_venues",
        )
    with col2:
        include_all_races = st.checkbox(
            "全レース掲載（OFFで重賞・特別戦のみ）",
            value=True,
            key="note_all_races",
        )
        include_secondary = st.checkbox(
            "相手馬（○▲△）も掲載",
            value=True,
            key="note_secondary",
        )

    col3, col4 = st.columns(2)
    with col3:
        intro_text = st.text_area(
            "冒頭あいさつ文（任意）",
            value=f"こんにちは。本日も競馬AI「v6モデル」による予想をお届けします。",
            height=80,
            key="note_intro",
        )
    with col4:
        footer_text = st.text_area(
            "末尾の注意書き（任意）",
            value="※本予想は参考情報です。最終判断はご自身でお願いします。\n※AI確率は複勝（3着以内）に入る確率です。",
            height=80,
            key="note_footer",
        )

    # ── 絞り込み ──
    df_show = df[df["venue"].isin(venues)].copy() if venues else pd.DataFrame()

    if not include_all_races and len(df_show) > 0:
        # 特別戦・重賞のみ：title に G1/G2/G3/S/オープン等を含む想定
        if "race_title" in df_show.columns:
            mask = df_show["race_title"].fillna("").str.contains(
                r"G[123]|GI|GII|GIII|オープン|特別|S\)", regex=True
            )
            keep_race_ids = df_show.loc[mask, "race_id"].unique()
            df_show = df_show[df_show["race_id"].isin(keep_race_ids)]

    if len(df_show) == 0:
        st.info("対象のレースがありません。")
        return

    # ── 生成 ──
    article_md = _generate_article(
        df_show,
        target_date=target_date,
        model_name=model_name,
        intro=intro_text,
        footer=footer_text,
        include_secondary=include_secondary,
    )

    st.subheader("📄 生成された記事（Markdown）")

    st.text_area(
        "コピーしてnoteに貼り付けてください",
        value=article_md,
        height=500,
        key="note_article_md",
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "💾 Markdown (.md) をダウンロード",
            data=article_md.encode("utf-8"),
            file_name=f"note_{target_date}_{_safe_name(model_name)}.md",
            mime="text/markdown",
        )
    with col_dl2:
        st.download_button(
            "💾 プレーンテキスト (.txt) をダウンロード",
            data=article_md.encode("utf-8"),
            file_name=f"note_{target_date}_{_safe_name(model_name)}.txt",
            mime="text/plain",
        )

    # ── プレビュー ──
    st.subheader("👀 プレビュー")
    st.markdown(article_md)


# ──────────────────────────────────────────────
# 記事生成
# ──────────────────────────────────────────────

def _generate_article(
    df: pd.DataFrame,
    target_date: str,
    model_name: str,
    intro: str,
    footer: str,
    include_secondary: bool,
) -> str:
    """会場別の記事Markdownを生成する"""
    lines: list[str] = []

    # ── タイトル ──
    date_str = _format_date_jp(target_date)
    venues = sorted(df["venue"].dropna().unique().tolist())
    venues_label = "・".join(venues) if venues else ""
    lines.append(f"# 【{date_str}】競馬AI予想 {venues_label}")
    lines.append("")

    if intro.strip():
        lines.append(intro.strip())
        lines.append("")

    lines.append(
        "マークの意味：◎本命／○対抗／▲単穴／△連下"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 会場ごと ──
    for venue in venues:
        venue_df = df[df["venue"] == venue]
        lines.append(f"## 🏇 {venue}")
        lines.append("")

        race_ids = sorted(venue_df["race_id"].unique(), key=lambda x: str(x))
        for race_id in race_ids:
            race_df = venue_df[venue_df["race_id"] == race_id].sort_values(
                "pred_prob", ascending=False
            )
            if len(race_df) == 0:
                continue

            race_num = str(race_id)[-2:].lstrip("0") or str(race_id)[-2:]
            title = _safe_str(race_df.get("race_title", pd.Series([""])).iloc[0])
            surface = _safe_str(race_df["surface"].iloc[0])
            distance = race_df["distance"].iloc[0]
            dist_str = f"{int(distance)}m" if pd.notna(distance) else ""

            header = f"### {race_num}R"
            if title:
                header += f" {title}"
            if surface and dist_str:
                header += f"（{surface}{dist_str}）"
            lines.append(header)
            lines.append("")

            # ── 本命馬 ──
            top = race_df.iloc[0]
            lines.append(f"**◎ {top['post_number']:.0f}番 {top['horse_name']}**  (AI確率 {top['pred_prob']:.1%})")

            reasons = _build_reasons(top, race_df)
            if reasons:
                lines.append("")
                lines.append("推奨理由:")
                for r in reasons:
                    lines.append(f"- {r}")
            lines.append("")

            # ── 相手馬 ──
            if include_secondary and len(race_df) >= 2:
                marks = ["○", "▲", "△", "△"]
                for i, mark in enumerate(marks, start=1):
                    if i >= len(race_df):
                        break
                    h = race_df.iloc[i]
                    lines.append(
                        f"{mark} {h['post_number']:.0f}番 {h['horse_name']}  (AI確率 {h['pred_prob']:.1%})"
                    )
                lines.append("")

            lines.append("")  # レース間の余白

        lines.append("---")
        lines.append("")

    # ── フッター ──
    if footer.strip():
        lines.append(footer.strip())
        lines.append("")

    lines.append(f"*モデル: {model_name} / 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 推奨理由の生成
# ──────────────────────────────────────────────

def _build_reasons(horse: pd.Series, race_df: pd.DataFrame) -> list[str]:
    """本命馬の推奨理由を特徴量から自動生成する（最大5件）"""
    reasons: list[str] = []

    def _v(col, default=None):
        v = horse.get(col, default)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return v

    # 1. 過去成績 (近5走複勝率)
    top3_rate = _v("horse_top3_rate_5")
    if top3_rate is not None and top3_rate >= 0.40:
        reasons.append(f"近5走の複勝率 **{top3_rate:.0%}** と安定した成績")

    # 2. 距離適性
    dist_rate = _v("horse_dist_top3_rate")
    dist_n = _v("horse_dist_race_count", 0) or 0
    if dist_rate is not None and dist_rate >= 0.45 and dist_n >= 2:
        reasons.append(f"同距離帯で複勝率 **{dist_rate:.0%}**（{int(dist_n)}戦）と適性◎")

    # 3. 馬場適性（芝/ダ）
    surface_rate = _v("horse_surface_top3_rate")
    surface = _safe_str(horse.get("surface", ""))
    if surface_rate is not None and surface_rate >= 0.45 and surface:
        reasons.append(f"{surface}での複勝率 **{surface_rate:.0%}** と得意条件")

    # 4. コース（競馬場）適性
    venue_rate = _v("horse_venue_top3_rate")
    venue = _safe_str(horse.get("venue", ""))
    if venue_rate is not None and venue_rate >= 0.50 and venue:
        reasons.append(f"{venue}競馬場で複勝率 **{venue_rate:.0%}** の好相性")

    # 5. スピード指数
    speed_idx = _v("horse_last_speed_idx")
    avg_speed = _v("horse_avg_speed_idx_5")
    race_avg_speed = race_df["horse_avg_speed_idx_5"].mean() if "horse_avg_speed_idx_5" in race_df.columns else None
    if avg_speed is not None and race_avg_speed is not None and avg_speed >= race_avg_speed + 3:
        reasons.append(f"近走スピード指数 **{avg_speed:.1f}** はレース平均を {avg_speed - race_avg_speed:+.1f} 上回る")
    elif speed_idx is not None and speed_idx >= 58:
        reasons.append(f"前走スピード指数 **{speed_idx:.1f}** と好時計")

    # 6. 騎手
    j_top3 = _v("jockey_top3_rate")
    j_recent = _v("jockey_recent_top3_20")
    j_name = _safe_str(horse.get("jockey_name", ""))
    if j_recent is not None and j_recent >= 0.35 and j_name:
        reasons.append(f"鞍上 **{j_name}** 騎手は直近20走の複勝率 {j_recent:.0%} と好調")
    elif j_top3 is not None and j_top3 >= 0.30 and j_name:
        reasons.append(f"鞍上 **{j_name}** 騎手の通算複勝率 {j_top3:.0%}")

    # 7. 血統（v6）
    sire = _safe_str(horse.get("sire", ""))
    sire_surface = _v("sire_surface_top3")
    if sire and sire_surface is not None and sire_surface >= 0.30:
        reasons.append(f"父 **{sire}** 産駒は{surface}で複勝率 {sire_surface:.0%} と血統面も好条件")

    # 8. 調子トレンド（form_trend < 0 = 直近が良い）
    form = _v("horse_form_trend")
    if form is not None and form <= -1.0:
        reasons.append(f"直近3走の平均着順が全5走平均より {abs(form):.1f} 着分上昇、上昇気配")

    # 9. レース内での相対的な強さ
    prob = _v("pred_prob", 0)
    race_probs = race_df["pred_prob"]
    if prob is not None and len(race_probs) >= 2:
        second = race_probs.iloc[1] if len(race_probs) >= 2 else 0
        if prob - second >= 0.08:
            reasons.append(f"AI確率 {prob:.0%} は2番手評価より {prob-second:+.0%} 高く抜けた存在")

    # 何も該当しない場合はAI確率だけ
    if not reasons:
        reasons.append(f"AIモデルが総合的に高評価（複勝確率 {prob:.0%}）")

    return reasons[:5]


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────

def _safe_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))


def _format_date_jp(s: str) -> str:
    """'2026-04-27' → '2026年4月27日（日）' に変換"""
    if not s:
        return ""
    try:
        d = datetime.strptime(str(s)[:10], "%Y-%m-%d")
    except ValueError:
        try:
            d = datetime.strptime(str(s)[:8], "%Y%m%d")
        except ValueError:
            return str(s)
    weekday = "月火水木金土日"[d.weekday()]
    return f"{d.year}年{d.month}月{d.day}日（{weekday}）"
