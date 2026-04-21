"""データベーススキーマ定義とテーブル作成"""

import sqlite3
from config.settings import DB_PATH


def create_tables(db_path=DB_PATH):
    """全テーブルを作成する"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # レーステーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS races (
            race_id     TEXT PRIMARY KEY,   -- 例: 202305021211 (年+競馬場+回+日+R)
            date        TEXT NOT NULL,      -- YYYY-MM-DD
            venue       TEXT NOT NULL,      -- 競馬場名
            venue_code  TEXT,               -- 競馬場コード
            race_number INTEGER,            -- 第Nレース
            title       TEXT,               -- レース名
            surface     TEXT,               -- 芝 / ダート / 障害
            distance    INTEGER,            -- 距離(m)
            weather     TEXT,               -- 天候
            condition   TEXT,               -- 馬場状態 (良/稍重/重/不良)
            grade       TEXT,               -- クラス (G1, G2, OP, 3勝クラス 等)
            head_count  INTEGER             -- 出走頭数
        )
    """)

    # 馬テーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS horses (
            horse_id    TEXT PRIMARY KEY,   -- netkeiba の horse_id
            name        TEXT NOT NULL,
            sex         TEXT,               -- 牡/牝/セ
            birthday    TEXT,               -- YYYY-MM-DD
            sire        TEXT,               -- 父
            dam         TEXT,               -- 母
            dam_sire    TEXT                -- 母父
        )
    """)

    # 騎手テーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jockeys (
            jockey_id   TEXT PRIMARY KEY,
            name        TEXT NOT NULL
        )
    """)

    # 調教師テーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trainers (
            trainer_id  TEXT PRIMARY KEY,
            name        TEXT NOT NULL
        )
    """)

    # 出走結果テーブル
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id         TEXT NOT NULL,
            horse_id        TEXT NOT NULL,
            jockey_id       TEXT,
            post_number     INTEGER,        -- 馬番
            gate_number     INTEGER,        -- 枠番
            odds            REAL,           -- 単勝オッズ
            popularity      INTEGER,        -- 人気順
            weight_carried  REAL,           -- 斤量
            horse_weight    INTEGER,        -- 馬体重
            weight_change   INTEGER,        -- 馬体重増減
            finish_position INTEGER,        -- 着順 (0=取消/除外)
            finish_time     TEXT,           -- タイム文字列
            finish_time_sec REAL,           -- タイム(秒)
            last_3f         REAL,           -- 上がり3F
            passing_order   TEXT,           -- 通過順 (例: "3-3-2-1")
            prize           REAL,           -- 賞金(万円)
            trainer_id      TEXT,           -- 調教師ID
            FOREIGN KEY (race_id) REFERENCES races(race_id),
            FOREIGN KEY (horse_id) REFERENCES horses(horse_id),
            FOREIGN KEY (jockey_id) REFERENCES jockeys(jockey_id),
            FOREIGN KEY (trainer_id) REFERENCES trainers(trainer_id),
            UNIQUE(race_id, horse_id)
        )
    """)

    # 払戻テーブル（複勝払戻を馬番ごとに保存）
    cur.execute("""
        CREATE TABLE IF NOT EXISTS payouts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id         TEXT NOT NULL,
            bet_type        TEXT NOT NULL,      -- 'fukusho', 'tansho' 等
            horse_number    INTEGER NOT NULL,   -- 馬番
            payout          INTEGER NOT NULL,   -- 払戻金(円) ※100円あたり
            FOREIGN KEY (race_id) REFERENCES races(race_id),
            UNIQUE(race_id, bet_type, horse_number)
        )
    """)

    # インデックス
    cur.execute("CREATE INDEX IF NOT EXISTS idx_results_race ON results(race_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_results_horse ON results(horse_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_results_jockey ON results(jockey_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_races_date ON races(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_payouts_race ON payouts(race_id)")

    # 既存DBへの後付けカラム追加（初回のみ実行、既存なら無視）
    try:
        cur.execute("ALTER TABLE results ADD COLUMN trainer_id TEXT")
        conn.commit()
    except Exception:
        pass  # already exists

    cur.execute("CREATE INDEX IF NOT EXISTS idx_results_trainer ON results(trainer_id)")

    conn.commit()
    conn.close()
    print(f"データベースを作成しました: {db_path}")


if __name__ == "__main__":
    create_tables()
