"""当日予測のスナップショット保存・読込（後日レビュー用）

予測ページの当日予測をそのまま DB に保存し、収支レビューで「再計算」ではなく
「当日保存した予測」を使えるようにする。

背景: 予測ページ(predict_features) とレビュー(build_features) は別経路で特徴量を
作るうえ、DB更新後は基準統計が変わるため、後日再計算すると当日と AI確率がズレる。
当日の予測値をそのまま保存しておけば、レビューは「実際にその日出した予測」と
実結果を正確に突き合わせられる。
"""

import sqlite3
from datetime import datetime

import pandas as pd

from config.settings import DB_PATH


def _ensure_table(conn: sqlite3.Connection) -> None:
    """スナップショットテーブルを作成（既存DBの自動マイグレーションも兼ねる）"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_snapshots (
            race_id        TEXT NOT NULL,
            post_number    INTEGER NOT NULL,
            model_name     TEXT NOT NULL,
            race_date      TEXT,             -- 予測対象日 (YYYY-MM-DD)
            horse_id       TEXT,
            horse_name     TEXT,
            jockey_name    TEXT,
            pred_prob      REAL,             -- 当日のAI確率
            market_prob    REAL,
            edge           REAL,
            expected_value REAL,
            odds           REAL,             -- 予測時点の単勝オッズ
            predicted_at   TEXT,             -- 保存時刻 (ISO)
            PRIMARY KEY (race_id, post_number, model_name)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_snap_date ON prediction_snapshots(race_date)"
    )


def _to_float(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def save_predictions(
    df: pd.DataFrame,
    model_name: str,
    race_date: str | None = None,
    db_path=DB_PATH,
) -> int:
    """予測DataFrameをスナップショット保存する。

    同一 (race_id, post_number, model_name) は上書き（最新の当日予測を保持）。

    Returns:
        保存した行数
    """
    if df is None or len(df) == 0:
        return 0

    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("post_number")):
            continue
        # race_date は引数優先、無ければ行の date 列から
        rdate = race_date
        if not rdate:
            d = r.get("date")
            rdate = str(d)[:10] if d is not None and not pd.isna(d) else None
        rows.append((
            str(r["race_id"]),
            int(r["post_number"]),
            str(model_name),
            rdate,
            str(r.get("horse_id") or ""),
            str(r.get("horse_name") or ""),
            str(r.get("jockey_name") or ""),
            _to_float(r.get("pred_prob")),
            _to_float(r.get("market_prob")),
            _to_float(r.get("edge")),
            _to_float(r.get("expected_value")),
            _to_float(r.get("odds")),
            now,
        ))

    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        conn.executemany("""
            INSERT OR REPLACE INTO prediction_snapshots
            (race_id, post_number, model_name, race_date, horse_id, horse_name,
             jockey_name, pred_prob, market_prob, edge, expected_value, odds, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)


def load_predictions(
    target_date: str,
    model_name: str | None = None,
    db_path=DB_PATH,
) -> pd.DataFrame:
    """対象日の保存済み予測を読み込む。

    race_date 列で直接フィルタ（races テーブルとの join 不要 = 結果確定前でも引ける）。
    """
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        query = "SELECT * FROM prediction_snapshots WHERE race_date = ?"
        params: list = [str(target_date)[:10]]
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        return pd.read_sql_query(query, conn, params=params)


def list_saved_dates(db_path=DB_PATH) -> list[str]:
    """保存済み予測がある対象日の一覧（新しい順）を返す"""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT DISTINCT race_date FROM prediction_snapshots "
            "WHERE race_date IS NOT NULL ORDER BY race_date DESC"
        ).fetchall()
    return [r[0] for r in rows]


def has_predictions(target_date: str, model_name: str | None = None, db_path=DB_PATH) -> bool:
    """対象日に保存済み予測があるか"""
    return len(load_predictions(target_date, model_name, db_path)) > 0
