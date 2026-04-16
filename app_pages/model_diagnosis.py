"""モデル診断 & 改善レポート生成ページ

2つのモード:
  1. 週次チェック: 直近N週の的中率・回収率の推移をワンクリックで確認
  2. 本格診断: 期間指定で全分析 → Claude用レポート出力
"""

import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
from pathlib import Path

from config.settings import DB_PATH, PROJECT_ROOT
from src.model.train import (
    load_model, FEATURE_COLUMNS, TARGET_COLUMN,
    prepare_dataset, get_available_features,
)


@st.cache_resource
def _load_model_cached(name: str):
    return load_model(name)


@st.cache_data(ttl=300)
def _build_features_cached():
    from src.features.build_features import build_all_features
    return build_all_features()


def render():
    st.header("🔬 モデル診断 & 改善レポート")

    if not DB_PATH.exists():
        st.warning("データベースが存在しません。")
        return

    tab_weekly, tab_full = st.tabs(["📅 週次チェック（毎週用）", "🔬 本格診断（月1用）"])

    with tab_weekly:
        _render_weekly_check()

    with tab_full:
        _render_full_diagnosis()


# ══════════════════════════════════════════════
# 週次チェック
# ══════════════════════════════════════════════

def _render_weekly_check():
    """直近N週の的中率・回収率の推移をワンクリックで確認"""

    st.markdown(
        "毎週の軽いチェック用。直近の的中率・回収率の推移を確認し、"
        "異変があれば本格診断に進んでください。"
    )

    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "モデル",
            ["lightgbm_v3", "lightgbm_v2", "lightgbm_v1"],
            key="weekly_model",
        )
    with col2:
        n_weeks = st.slider("表示する週数", 4, 52, 12, key="weekly_n_weeks")

    if st.button("📅 週次チェック実行", type="primary", key="btn_weekly"):
        _run_weekly_check(model_name, n_weeks)


def _run_weekly_check(model_name: str, n_weeks: int):
    """週ごとの的中率・回収率を計算して推移グラフを表示"""

    try:
        model = _load_model_cached(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。")
        return

    with st.spinner("特徴量を構築中..."):
        df = _build_features_cached()

    df = prepare_dataset(df)
    features = get_available_features(df)

    X = df[features]
    df["pred_prob"] = model.predict_proba(X)[:, 1]

    # 日付をdatetime化
    df["date_dt"] = pd.to_datetime(df["date"])
    df["week"] = df["date_dt"].dt.to_period("W").apply(lambda r: r.start_time)

    # 直近N週に絞る
    latest = df["date_dt"].max()
    cutoff = latest - timedelta(weeks=n_weeks)
    recent = df[df["date_dt"] >= cutoff].copy()

    if len(recent) == 0:
        st.warning("対象期間にデータがありません。")
        return

    # 複勝概算
    recent["fukusho_odds"] = (recent["odds"] * 0.3).clip(lower=1.1)
    recent["market_prob"] = (3.0 / recent["odds"]).clip(upper=1.0)
    recent["edge"] = recent["pred_prob"] - recent["market_prob"]

    # ── 週ごとの集計 ──
    weekly_stats = []
    for week, wdf in recent.groupby("week"):
        # 全体統計
        n_total = len(wdf)
        n_races = wdf["race_id"].nunique()

        # エッジ >= 0.1 のバリューベットのみ
        bets = wdf[(wdf["edge"] >= 0.10) & (wdf["odds"] >= 3.0)].copy()
        n_bets = len(bets)
        if n_bets == 0:
            weekly_stats.append({
                "week": week, "n_races": n_races, "n_bets": 0,
                "hit_rate": 0, "roi": 0,
                "win_hit_rate": 0, "win_roi": 0,
                "avg_pred": 0,
            })
            continue

        # 複勝（3着以内）
        hits = bets[TARGET_COLUMN].sum()
        payout = np.where(bets[TARGET_COLUMN] == 1, bets["fukusho_odds"], 0).sum()
        roi = payout / n_bets * 100

        # 単勝（1着のみ）
        win_hits = (bets["finish_position"] == 1).sum()
        win_payout = np.where(bets["finish_position"] == 1, bets["odds"], 0).sum()
        win_roi = win_payout / n_bets * 100

        weekly_stats.append({
            "week": week,
            "n_races": n_races,
            "n_bets": n_bets,
            "hits": int(hits),
            "hit_rate": hits / n_bets * 100,
            "roi": roi,
            "win_hits": int(win_hits),
            "win_hit_rate": win_hits / n_bets * 100,
            "win_roi": win_roi,
            "avg_pred": bets["pred_prob"].mean(),
            "avg_odds": bets["odds"].mean(),
            "avg_edge": bets["edge"].mean(),
        })

    ws = pd.DataFrame(weekly_stats)

    # ── サマリー ──
    total_bets = ws["n_bets"].sum()
    if total_bets > 0:
        bets_all = recent[(recent["edge"] >= 0.10) & (recent["odds"] >= 3.0)]
        # 複勝
        total_hits = bets_all[TARGET_COLUMN].sum()
        total_payout = np.where(bets_all[TARGET_COLUMN] == 1, bets_all["fukusho_odds"], 0).sum()
        overall_roi = total_payout / total_bets * 100
        overall_hit_rate = total_hits / total_bets * 100
        # 単勝
        total_win_hits = (bets_all["finish_position"] == 1).sum()
        total_win_payout = np.where(bets_all["finish_position"] == 1, bets_all["odds"], 0).sum()
        overall_win_roi = total_win_payout / total_bets * 100
        overall_win_hit_rate = total_win_hits / total_bets * 100
    else:
        overall_roi = overall_hit_rate = 0
        overall_win_roi = overall_win_hit_rate = 0

    st.markdown("**複勝（3着以内）**")
    cols = st.columns(4)
    cols[0].metric("対象期間", f"直近{n_weeks}週")
    cols[1].metric("総賭け数", f"{int(total_bets)}件")
    cols[2].metric("的中率", f"{overall_hit_rate:.1f}%")
    roi_delta = overall_roi - 100
    cols[3].metric("回収率", f"{overall_roi:.1f}%", f"{roi_delta:+.1f}%")

    st.markdown("**単勝（1着）**")
    cols2 = st.columns(4)
    cols2[0].metric("対象期間", f"直近{n_weeks}週")
    cols2[1].metric("総賭け数", f"{int(total_bets)}件")
    cols2[2].metric("的中率", f"{overall_win_hit_rate:.1f}%")
    win_roi_delta = overall_win_roi - 100
    cols2[3].metric("回収率", f"{overall_win_roi:.1f}%", f"{win_roi_delta:+.1f}%")

    st.divider()

    # ── 推移グラフ ──
    ws_plot = ws[ws["n_bets"] > 0].copy()

    if len(ws_plot) == 0:
        st.warning("賭け対象のあるデータがありません。")
        return

    st.subheader("📈 回収率の推移")
    fig = go.Figure()
    fig.add_scatter(
        x=ws_plot["week"], y=ws_plot["roi"],
        mode="lines+markers", name="複勝 回収率",
        line=dict(color="#FF6B6B", width=2),
        marker=dict(size=8),
    )
    fig.add_scatter(
        x=ws_plot["week"], y=ws_plot["win_roi"],
        mode="lines+markers", name="単勝 回収率",
        line=dict(color="#FF9F43", width=2, dash="dot"),
        marker=dict(size=6),
    )
    fig.add_hline(y=100, line_dash="dash", line_color="gray",
                  annotation_text="損益分岐点 (100%)")
    fig.update_layout(
        yaxis_title="回収率 (%)", xaxis_title="週",
        height=350, margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 的中率の推移")
    fig2 = go.Figure()
    fig2.add_scatter(
        x=ws_plot["week"], y=ws_plot["hit_rate"],
        mode="lines+markers", name="複勝 的中率",
        line=dict(color="#4ECDC4", width=2),
        marker=dict(size=8),
    )
    fig2.add_scatter(
        x=ws_plot["week"], y=ws_plot["win_hit_rate"],
        mode="lines+markers", name="単勝 的中率",
        line=dict(color="#45B7D1", width=2, dash="dot"),
        marker=dict(size=6),
    )
    fig2.update_layout(
        yaxis_title="的中率 (%)", xaxis_title="週",
        height=350, margin=dict(t=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 週別テーブル ──
    st.subheader("📋 週別データ")
    disp = ws.copy()
    disp["週"] = disp["week"].dt.strftime("%m/%d〜")
    disp["レース数"] = disp["n_races"]
    disp["賭け数"] = disp["n_bets"]
    disp["複勝的中率"] = disp["hit_rate"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
    disp["複勝回収率"] = disp["roi"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
    disp["単勝的中率"] = disp["win_hit_rate"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")
    disp["単勝回収率"] = disp["win_roi"].apply(lambda x: f"{x:.1f}%" if x > 0 else "-")

    st.dataframe(
        disp[["週", "レース数", "賭け数", "複勝的中率", "複勝回収率", "単勝的中率", "単勝回収率"]],
        use_container_width=True, hide_index=True,
    )

    # ── 異変アラート ──
    st.subheader("🚨 異変チェック")
    alerts = []

    # 直近4週の回収率が80%未満
    last4 = ws_plot.tail(4)
    if len(last4) >= 4 and last4["roi"].mean() < 80:
        alerts.append("⚠️ 直近4週の平均回収率が80%未満です。本格診断の実行を推奨します。")

    # 直近2週連続で的中率が10%未満
    last2 = ws_plot.tail(2)
    if len(last2) >= 2 and (last2["hit_rate"] < 10).all():
        alerts.append("🔴 直近2週連続で的中率10%未満です。モデルの再学習を検討してください。")

    # 回収率が3週連続で下降
    if len(ws_plot) >= 3:
        last3_roi = ws_plot["roi"].tail(3).tolist()
        if last3_roi[0] > last3_roi[1] > last3_roi[2]:
            alerts.append("📉 回収率が3週連続で下降しています。傾向変化の可能性。")

    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ 現時点で大きな異変は検出されていません。")


# ══════════════════════════════════════════════
# 本格診断
# ══════════════════════════════════════════════

def _render_full_diagnosis():
    """期間指定で本格的な診断を実行 → Claude用レポート出力"""

    st.markdown(
        "月1回の本格分析。弱点の特定 → Claudeへの改善依頼レポートを生成します。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        model_name = st.selectbox(
            "分析するモデル",
            ["lightgbm_v3", "lightgbm_v2", "lightgbm_v1"],
            key="diag_model",
        )
    with col2:
        period_choice = st.selectbox(
            "分析期間",
            ["テストデータ（最新20%）", "直近1ヶ月", "直近3ヶ月", "直近6ヶ月", "カスタム"],
            key="diag_period",
        )
    with col3:
        if period_choice == "カスタム":
            custom_start = st.date_input("開始日", value=date.today() - timedelta(days=90), key="diag_start")
            custom_end = st.date_input("終了日", value=date.today(), key="diag_end")
        else:
            custom_start = None
            custom_end = None

    if st.button("🔬 本格診断を実行", type="primary", key="btn_full_diag"):
        _run_full_diagnosis(model_name, period_choice, custom_start, custom_end)


def _run_full_diagnosis(model_name, period_choice, custom_start, custom_end):
    """モデルの弱点を分析してレポートを生成する"""

    try:
        model = _load_model_cached(model_name)
    except FileNotFoundError:
        st.error(f"モデル {model_name} が見つかりません。")
        return

    with st.spinner("特徴量を構築中..."):
        df = _build_features_cached()

    df = prepare_dataset(df)
    features = get_available_features(df)

    # 期間の決定
    df["date_dt"] = pd.to_datetime(df["date"])

    if period_choice == "テストデータ（最新20%）":
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
    elif period_choice == "直近1ヶ月":
        cutoff = df["date_dt"].max() - timedelta(days=30)
        test_df = df[df["date_dt"] >= cutoff].copy()
    elif period_choice == "直近3ヶ月":
        cutoff = df["date_dt"].max() - timedelta(days=90)
        test_df = df[df["date_dt"] >= cutoff].copy()
    elif period_choice == "直近6ヶ月":
        cutoff = df["date_dt"].max() - timedelta(days=180)
        test_df = df[df["date_dt"] >= cutoff].copy()
    elif period_choice == "カスタム" and custom_start and custom_end:
        test_df = df[
            (df["date_dt"] >= pd.Timestamp(custom_start)) &
            (df["date_dt"] <= pd.Timestamp(custom_end))
        ].copy()
    else:
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()

    if len(test_df) == 0:
        st.warning("対象期間にデータがありません。")
        return

    X_test = test_df[features]
    test_df["pred_prob"] = model.predict_proba(X_test)[:, 1]
    test_df["pred_top3"] = (test_df["pred_prob"] >= 0.5).astype(int)

    period = f"{test_df['date'].min()} 〜 {test_df['date'].max()}"
    st.info(f"分析期間: {period}（{test_df['race_id'].nunique()} レース / {len(test_df)} 頭）")

    # ─── 各種分析を実行 ───
    sections = {}

    sections["overall"] = _analyze_overall(test_df)
    sections["calibration"] = _analyze_calibration(test_df)
    sections["by_surface"] = _analyze_by_category(test_df, "surface", "馬場")
    sections["by_distance"] = _analyze_by_distance(test_df)
    sections["by_condition"] = _analyze_by_category(test_df, "condition", "馬場状態")
    sections["by_venue"] = _analyze_by_category(test_df, "venue", "競馬場")
    sections["by_odds_range"] = _analyze_by_odds(test_df)
    sections["miss_patterns"] = _analyze_miss_patterns(test_df)
    sections["feature_importance"] = _analyze_features(model, features, test_df)
    sections["value_bet"] = _analyze_value_bet(test_df)

    # ─── 画面に分析結果を表示 ───
    _display_analysis(sections)

    # ─── Claudeへの改善依頼レポート生成 ───
    report = _generate_report(model_name, period, sections, features)

    st.divider()
    st.subheader("📋 Claude改善依頼レポート")
    st.markdown("以下をコピーしてClaudeに貼り付けてください：")
    st.code(report, language="markdown")

    st.download_button(
        "📥 レポートをダウンロード",
        data=report,
        file_name="model_improvement_report.md",
        mime="text/markdown",
    )


# ══════════════════════════════════════════════
# 分析関数
# ══════════════════════════════════════════════

def _analyze_overall(df: pd.DataFrame) -> dict:
    """全体の精度指標"""
    n = len(df)
    actual_top3 = df[TARGET_COLUMN].sum()
    pred_positive = (df["pred_prob"] >= 0.5).sum()
    tp = ((df["pred_prob"] >= 0.5) & (df[TARGET_COLUMN] == 1)).sum()
    fp = ((df["pred_prob"] >= 0.5) & (df[TARGET_COLUMN] == 0)).sum()
    fn = ((df["pred_prob"] < 0.5) & (df[TARGET_COLUMN] == 1)).sum()

    precision = tp / pred_positive if pred_positive > 0 else 0
    recall = tp / actual_top3 if actual_top3 > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    from sklearn.metrics import roc_auc_score, log_loss
    auc = roc_auc_score(df[TARGET_COLUMN], df["pred_prob"])
    logloss = log_loss(df[TARGET_COLUMN], df["pred_prob"])

    return {
        "n_samples": n,
        "actual_top3_rate": actual_top3 / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "log_loss": logloss,
    }


def _analyze_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """確率キャリブレーション（予測確率 vs 実際の的中率）"""
    df = df.copy()
    df["prob_bucket"] = pd.cut(
        df["pred_prob"],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    cal = df.groupby("prob_bucket", observed=True).agg(
        count=("pred_prob", "count"),
        avg_pred=(TARGET_COLUMN, lambda x: df.loc[x.index, "pred_prob"].mean()),
        actual_rate=(TARGET_COLUMN, "mean"),
    ).reset_index()
    cal["gap"] = cal["actual_rate"] - cal["avg_pred"]
    return cal


def _analyze_by_category(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    """カテゴリ別の精度"""
    results = []
    for val in df[col].dropna().unique():
        sub = df[df[col] == val]
        if len(sub) < 30:
            continue
        actual_rate = sub[TARGET_COLUMN].mean()
        avg_pred = sub["pred_prob"].mean()
        tp = ((sub["pred_prob"] >= 0.3) & (sub[TARGET_COLUMN] == 1)).sum()
        total_bet = (sub["pred_prob"] >= 0.3).sum()
        hit_rate = tp / total_bet if total_bet > 0 else 0
        bets = sub[sub["pred_prob"] >= 0.3]
        payout = np.where(bets[TARGET_COLUMN] == 1, bets["odds"] * 0.3, 0).sum()
        roi = payout / len(bets) * 100 if len(bets) > 0 else 0

        results.append({
            label: val,
            "サンプル数": len(sub),
            "実際の3着内率": f"{actual_rate:.1%}",
            "予測平均": f"{avg_pred:.3f}",
            "的中率(30%↑)": f"{hit_rate:.1%}",
            "概算回収率": f"{roi:.1f}%",
            "_roi": roi,
            "_gap": avg_pred - actual_rate,
        })
    return pd.DataFrame(results)


def _analyze_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [0, 1200, 1600, 2000, 2400, 9999]
    labels = ["〜1200m", "1201-1600m", "1601-2000m", "2001-2400m", "2401m〜"]
    df["dist_cat"] = pd.cut(df["distance"], bins=bins, labels=labels, include_lowest=True)
    return _analyze_by_category(df, "dist_cat", "距離")


def _analyze_by_odds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [0, 3, 5, 10, 20, 50, 9999]
    labels = ["〜3倍", "3-5倍", "5-10倍", "10-20倍", "20-50倍", "50倍〜"]
    df["odds_range"] = pd.cut(df["odds"], bins=bins, labels=labels, include_lowest=True)
    return _analyze_by_category(df, "odds_range", "オッズ")


def _analyze_miss_patterns(df: pd.DataFrame) -> dict:
    """予測ミスのパターン分析"""
    high_conf_miss = df[(df["pred_prob"] >= 0.5) & (df[TARGET_COLUMN] == 0)].copy()
    low_conf_hit = df[(df["pred_prob"] < 0.2) & (df[TARGET_COLUMN] == 1)].copy()

    result = {
        "high_conf_miss_count": len(high_conf_miss),
        "low_conf_hit_count": len(low_conf_hit),
    }

    if len(high_conf_miss) > 10:
        result["miss_surface"] = high_conf_miss["surface"].value_counts().to_dict()
        result["miss_avg_odds"] = high_conf_miss["odds"].mean()
        result["miss_condition"] = high_conf_miss["condition"].value_counts().to_dict()
        if "horse_race_count" in high_conf_miss.columns:
            result["miss_avg_race_count"] = high_conf_miss["horse_race_count"].mean()
        if "days_since_last" in high_conf_miss.columns:
            result["miss_avg_rest"] = high_conf_miss["days_since_last"].mean()

    if len(low_conf_hit) > 10:
        result["sleeper_surface"] = low_conf_hit["surface"].value_counts().to_dict()
        result["sleeper_avg_odds"] = low_conf_hit["odds"].mean()
        result["sleeper_condition"] = low_conf_hit["condition"].value_counts().to_dict()
        if "horse_race_count" in low_conf_hit.columns:
            result["sleeper_avg_race_count"] = low_conf_hit["horse_race_count"].mean()

    return result


def _analyze_features(model, features, df) -> dict:
    """特徴量の重要度と実際の相関"""
    importance = pd.Series(model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False)

    correlations = {}
    for feat in features:
        if feat in df.columns and df[feat].notna().sum() > 100:
            corr = df[feat].corr(df[TARGET_COLUMN].astype(float))
            correlations[feat] = round(corr, 4) if pd.notna(corr) else 0

    top15 = importance.head(15)
    bottom10 = importance.tail(10)

    suspicious = []
    for feat in top15.index:
        if feat in correlations and abs(correlations[feat]) < 0.02:
            suspicious.append(feat)

    underused = []
    for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        if feat in importance and importance[feat] < importance.median():
            underused.append((feat, corr))

    return {
        "top15": top15.to_dict(),
        "bottom10": bottom10.to_dict(),
        "correlations": correlations,
        "suspicious": suspicious,
        "underused": underused,
    }


def _analyze_value_bet(df: pd.DataFrame) -> dict:
    """バリューベット戦略の実績"""
    df = df.copy()
    df["fukusho_odds"] = (df["odds"] * 0.3).clip(lower=1.1)
    df["ev"] = df["pred_prob"] * df["fukusho_odds"]
    df["market_prob"] = (3.0 / df["odds"]).clip(upper=1.0)
    df["edge"] = df["pred_prob"] - df["market_prob"]

    results = {}
    for edge_th in [0.05, 0.10, 0.15, 0.20]:
        bets = df[df["edge"] >= edge_th]
        if len(bets) == 0:
            continue
        hits = bets[TARGET_COLUMN].sum()
        payout = np.where(bets[TARGET_COLUMN] == 1, bets["fukusho_odds"], 0).sum()
        roi = payout / len(bets) * 100
        results[edge_th] = {
            "n_bets": len(bets),
            "hit_rate": hits / len(bets),
            "roi": roi,
            "avg_odds": bets["odds"].mean(),
            "avg_edge": bets["edge"].mean(),
        }
    return results


# ══════════════════════════════════════════════
# 画面表示
# ══════════════════════════════════════════════

def _display_analysis(sections: dict):
    """分析結果を画面に表示"""

    # ── 全体指標 ──
    st.subheader("📊 全体精度")
    o = sections["overall"]
    cols = st.columns(4)
    cols[0].metric("AUC", f"{o['auc']:.4f}")
    cols[1].metric("Precision", f"{o['precision']:.3f}")
    cols[2].metric("Recall", f"{o['recall']:.3f}")
    cols[3].metric("Log Loss", f"{o['log_loss']:.4f}")

    # ── キャリブレーション ──
    st.subheader("🎯 確率キャリブレーション")
    st.caption("予測確率帯ごとの実際の3着内率。ギャップが大きいほどキャリブレーションが悪い。")
    cal = sections["calibration"]
    if len(cal) > 0:
        # グラフ
        fig = go.Figure()
        fig.add_scatter(
            x=cal["avg_pred"], y=cal["actual_rate"],
            mode="markers+lines", name="実際",
            marker=dict(size=10, color="#FF6B6B"),
        )
        fig.add_scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", name="完全一致",
            line=dict(dash="dash", color="gray"),
        )
        fig.update_layout(
            xaxis_title="予測確率", yaxis_title="実際の3着内率",
            height=300, margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            cal[["prob_bucket", "count", "avg_pred", "actual_rate", "gap"]].rename(
                columns={
                    "prob_bucket": "予測確率帯", "count": "サンプル数",
                    "avg_pred": "予測平均", "actual_rate": "実際の率", "gap": "ギャップ",
                }
            ),
            use_container_width=True, hide_index=True,
        )

    # ── カテゴリ別 ──
    for key, label in [
        ("by_surface", "馬場別"),
        ("by_distance", "距離別"),
        ("by_condition", "馬場状態別"),
        ("by_venue", "競馬場別"),
        ("by_odds_range", "オッズレンジ別"),
    ]:
        data = sections[key]
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            st.subheader(f"📊 {label}")
            display_cols = [c for c in data.columns if not c.startswith("_")]
            st.dataframe(data[display_cols], use_container_width=True, hide_index=True)

    # ── ミスパターン ──
    st.subheader("⚠️ 予測ミスのパターン")
    mp = sections["miss_patterns"]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("高確率ミス（50%↑で4着以下）", f"{mp['high_conf_miss_count']} 頭")
        if "miss_surface" in mp:
            st.caption(f"馬場内訳: {mp['miss_surface']}")
        if "miss_condition" in mp:
            st.caption(f"馬場状態: {mp['miss_condition']}")
    with col2:
        st.metric("見逃し（20%↓で3着以内）", f"{mp['low_conf_hit_count']} 頭")
        if "sleeper_surface" in mp:
            st.caption(f"馬場内訳: {mp['sleeper_surface']}")
        if "sleeper_avg_odds" in mp:
            st.caption(f"平均オッズ: {mp['sleeper_avg_odds']:.1f} 倍")

    # ── 特徴量 ──
    st.subheader("🔧 特徴量分析")
    fi = sections["feature_importance"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**重要度 Top15**")
        for feat, imp in fi["top15"].items():
            corr = fi["correlations"].get(feat, 0)
            flag = " ⚠️" if feat in fi["suspicious"] else ""
            st.caption(f"{feat}: 重要度={imp} 相関={corr:+.4f}{flag}")
    with col2:
        if fi["underused"]:
            st.markdown("**活用不足の可能性がある特徴量**")
            for feat, corr in fi["underused"]:
                st.caption(f"{feat}: 相関={corr:+.4f}")

    # ── バリューベット実績 ──
    st.subheader("💰 バリューベット実績")
    vb = sections["value_bet"]
    if vb:
        vb_rows = []
        for edge_th, stats in vb.items():
            vb_rows.append({
                "エッジ閾値": f"{edge_th:.2f}",
                "賭け数": stats["n_bets"],
                "的中率": f"{stats['hit_rate']:.1%}",
                "回収率": f"{stats['roi']:.1f}%",
                "平均オッズ": f"{stats['avg_odds']:.1f}",
                "平均エッジ": f"{stats['avg_edge']:.3f}",
            })
        st.dataframe(pd.DataFrame(vb_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# レポート生成
# ══════════════════════════════════════════════

def _generate_report(
    model_name: str,
    period: str,
    sections: dict,
    features: list,
) -> str:
    """Claudeに貼り付けるための改善依頼レポートを生成する"""

    o = sections["overall"]
    mp = sections["miss_patterns"]
    fi = sections["feature_importance"]
    vb = sections["value_bet"]

    lines = []
    lines.append("# 競馬予想AI 改善依頼レポート")
    lines.append("")
    lines.append("以下はモデル診断の分析結果です。このデータを基に改善を提案してください。")
    lines.append("")

    lines.append("## 1. 現在のモデル概要")
    lines.append(f"- モデル: {model_name}")
    lines.append(f"- テスト期間: {period}")
    lines.append(f"- サンプル数: {o['n_samples']}")
    lines.append(f"- 使用特徴量数: {len(features)}")
    lines.append(f"- 目的変数: 3着以内（二値分類）")
    lines.append(f"- アルゴリズム: LightGBM")
    lines.append("")

    lines.append("## 2. 全体精度")
    lines.append(f"- AUC: {o['auc']:.4f}")
    lines.append(f"- Precision (50%閾値): {o['precision']:.3f}")
    lines.append(f"- Recall: {o['recall']:.3f}")
    lines.append(f"- F1: {o['f1']:.3f}")
    lines.append(f"- Log Loss: {o['log_loss']:.4f}")
    lines.append(f"- 実際の3着内率: {o['actual_top3_rate']:.3f}")
    lines.append("")

    lines.append("## 3. 確率キャリブレーション")
    lines.append("予測確率帯ごとの実際の3着内率:")
    lines.append("")
    cal = sections["calibration"]
    if len(cal) > 0:
        lines.append("| 予測確率帯 | サンプル数 | 予測平均 | 実際の率 | ギャップ |")
        lines.append("|---|---|---|---|---|")
        for _, row in cal.iterrows():
            lines.append(
                f"| {row['prob_bucket']} | {row['count']} | "
                f"{row['avg_pred']:.3f} | {row['actual_rate']:.3f} | "
                f"{row['gap']:+.3f} |"
            )
    lines.append("")

    lines.append("## 4. カテゴリ別の弱点")
    lines.append("")

    for key, label in [
        ("by_surface", "馬場別"),
        ("by_distance", "距離別"),
        ("by_condition", "馬場状態別"),
        ("by_odds_range", "オッズレンジ別"),
    ]:
        data = sections[key]
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            lines.append(f"### {label}")
            weak = data[data["_roi"] < 80] if "_roi" in data.columns else pd.DataFrame()
            strong = data[data["_roi"] >= 120] if "_roi" in data.columns else pd.DataFrame()

            display_cols = [c for c in data.columns if not c.startswith("_")]
            lines.append("")
            header = "| " + " | ".join(display_cols) + " |"
            sep = "|" + "|".join(["---"] * len(display_cols)) + "|"
            lines.append(header)
            lines.append(sep)
            for _, row in data.iterrows():
                vals = [str(row[c]) for c in display_cols]
                lines.append("| " + " | ".join(vals) + " |")

            if len(weak) > 0:
                weak_cats = ", ".join(str(w) for w in weak.iloc[:, 0].tolist())
                lines.append(f"\n**弱点**: {weak_cats} で回収率80%未満")
            if len(strong) > 0:
                strong_cats = ", ".join(str(s) for s in strong.iloc[:, 0].tolist())
                lines.append(f"**強み**: {strong_cats} で回収率120%以上")
            lines.append("")

    venue_data = sections.get("by_venue")
    if isinstance(venue_data, pd.DataFrame) and len(venue_data) > 0:
        lines.append("### 競馬場別")
        weak_venues = venue_data[venue_data["_roi"] < 80] if "_roi" in venue_data.columns else pd.DataFrame()
        strong_venues = venue_data[venue_data["_roi"] >= 120] if "_roi" in venue_data.columns else pd.DataFrame()
        if len(weak_venues) > 0:
            lines.append(f"**苦手競馬場**: {', '.join(str(v) for v in weak_venues.iloc[:, 0].tolist())}")
        if len(strong_venues) > 0:
            lines.append(f"**得意競馬場**: {', '.join(str(v) for v in strong_venues.iloc[:, 0].tolist())}")
        lines.append("")

    lines.append("## 5. 予測ミスのパターン")
    lines.append(f"- 高確率ミス（50%↑で4着以下）: {mp['high_conf_miss_count']} 頭")
    if "miss_surface" in mp:
        lines.append(f"  - 馬場内訳: {mp['miss_surface']}")
    if "miss_condition" in mp:
        lines.append(f"  - 馬場状態: {mp['miss_condition']}")
    if "miss_avg_odds" in mp:
        lines.append(f"  - 平均オッズ: {mp['miss_avg_odds']:.1f}")
    if "miss_avg_race_count" in mp:
        lines.append(f"  - 平均出走回数: {mp['miss_avg_race_count']:.1f}")
    if "miss_avg_rest" in mp:
        lines.append(f"  - 平均休養日数: {mp['miss_avg_rest']:.0f}日")
    lines.append(f"- 見逃し（20%↓で3着以内）: {mp['low_conf_hit_count']} 頭")
    if "sleeper_surface" in mp:
        lines.append(f"  - 馬場内訳: {mp['sleeper_surface']}")
    if "sleeper_avg_odds" in mp:
        lines.append(f"  - 平均オッズ: {mp['sleeper_avg_odds']:.1f}")
    if "sleeper_avg_race_count" in mp:
        lines.append(f"  - 平均出走回数: {mp['sleeper_avg_race_count']:.1f}")
    lines.append("")

    lines.append("## 6. 特徴量分析")
    lines.append("")
    lines.append("### 重要度 Top15")
    lines.append("| 特徴量 | 重要度 | 対3着内相関 |")
    lines.append("|---|---|---|")
    for feat, imp in fi["top15"].items():
        corr = fi["correlations"].get(feat, 0)
        flag = " ⚠️ノイズ疑い" if feat in fi["suspicious"] else ""
        lines.append(f"| {feat} | {imp} | {corr:+.4f}{flag} |")
    lines.append("")

    if fi["suspicious"]:
        lines.append(f"**ノイズ疑い**: {', '.join(fi['suspicious'])} — 重要度は高いが相関がほぼ0")
        lines.append("")

    if fi["underused"]:
        lines.append("### 活用不足の可能性")
        for feat, corr in fi["underused"]:
            lines.append(f"- {feat}: 相関={corr:+.4f} だが重要度が中央値以下")
        lines.append("")

    lines.append("## 7. バリューベット実績（複勝概算）")
    if vb:
        lines.append("| エッジ閾値 | 賭け数 | 的中率 | 回収率 | 平均オッズ |")
        lines.append("|---|---|---|---|---|")
        for edge_th, stats in vb.items():
            lines.append(
                f"| {edge_th:.2f} | {stats['n_bets']} | "
                f"{stats['hit_rate']:.1%} | {stats['roi']:.1f}% | "
                f"{stats['avg_odds']:.1f} |"
            )
    lines.append("")

    lines.append("## 8. 自動検出された改善ポイント")
    lines.append("")
    suggestions = _auto_suggest(sections)
    for i, sug in enumerate(suggestions, 1):
        lines.append(f"{i}. {sug}")
    lines.append("")

    lines.append("## 依頼")
    lines.append("")
    lines.append("上記の分析結果を踏まえて、以下を実装してください：")
    lines.append("")
    lines.append("1. 最も効果の高い改善を1〜3個に絞って提案")
    lines.append("2. 各改善について、具体的なコード変更内容を説明")
    lines.append("3. 改善後の期待される効果（回収率・的中率の向上見込み）を説明")
    lines.append("4. 改善の実装")
    lines.append("")
    lines.append("※ プロジェクトディレクトリ: D:\\\\開発中\\\\競馬予想AI")
    lines.append("※ 特徴量構築: src/features/build_features.py")
    lines.append("※ モデル学習: src/model/train.py")
    lines.append("※ 予測用特徴量: src/features/predict_features.py")

    return "\n".join(lines)


def _auto_suggest(sections: dict) -> list[str]:
    """分析結果から自動的に改善提案を生成する"""
    suggestions = []
    o = sections["overall"]
    cal = sections["calibration"]
    mp = sections["miss_patterns"]
    fi = sections["feature_importance"]
    vb = sections["value_bet"]

    if len(cal) > 0:
        max_gap = cal["gap"].abs().max()
        if max_gap > 0.1:
            worst_bucket = cal.loc[cal["gap"].abs().idxmax()]
            direction = "過大評価" if worst_bucket["gap"] < 0 else "過小評価"
            suggestions.append(
                f"確率キャリブレーションが悪い: "
                f"予測 {worst_bucket['prob_bucket']} 帯でギャップ {worst_bucket['gap']:+.3f}（{direction}）。"
                f"Plattスケーリングまたはisotonic regressionの適用を検討。"
            )

    if fi["suspicious"]:
        suggestions.append(
            f"ノイズ疑いの特徴量: {', '.join(fi['suspicious'])}。"
            f"重要度は高いが3着内との相関がほぼ0。過学習の原因になっている可能性。"
            f"特徴量の計算ロジックを見直すか、除外を検討。"
        )

    if fi["underused"]:
        feats = [f[0] for f in fi["underused"][:3]]
        suggestions.append(
            f"活用不足の特徴量: {', '.join(feats)}。"
            f"実際の着順との相関は高いがモデルが活かしきれていない。"
            f"交互作用項の追加や非線形変換を検討。"
        )

    for key, label in [("by_surface", "馬場"), ("by_condition", "馬場状態"), ("by_distance", "距離")]:
        data = sections[key]
        if isinstance(data, pd.DataFrame) and "_roi" in data.columns:
            weak = data[data["_roi"] < 70]
            if len(weak) > 0:
                cats = ", ".join(str(w) for w in weak.iloc[:, 0].tolist())
                suggestions.append(
                    f"{label}別の弱点: {cats} で回収率70%未満。"
                    f"この条件に特化した特徴量の追加またはサブモデルの検討。"
                )

    if mp["high_conf_miss_count"] > 50:
        suggestions.append(
            f"高確率ミスが{mp['high_conf_miss_count']}件。"
            f"過信している傾向がある。"
            + (f"馬場状態{mp.get('miss_condition', '')}や" if "miss_condition" in mp else "")
            + f"休養明けの馬への対策を検討。"
        )

    if mp["low_conf_hit_count"] > 200:
        suggestions.append(
            f"見逃し（低確率的中）が{mp['low_conf_hit_count']}件。"
            f"穴馬検出力が弱い。"
            + (f"平均オッズ{mp.get('sleeper_avg_odds', 0):.1f}倍の人気薄に対する" if "sleeper_avg_odds" in mp else "")
            + f"特徴量（血統、トラックバイアスの深掘りなど）の追加を検討。"
        )

    if o["auc"] < 0.70:
        suggestions.append(
            f"AUCが{o['auc']:.4f}と低い。モデルの判別力自体が不足。"
            f"特徴量の大幅な見直しまたはアルゴリズム変更を検討。"
        )

    if not suggestions:
        suggestions.append("明確な弱点は検出されませんでした。微調整フェーズです。")

    return suggestions
