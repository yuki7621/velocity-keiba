"""プロジェクト全体の設定"""

from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# データディレクトリ
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# データベース
DB_PATH = DATA_DIR / "keiba.db"

# スクレイピング設定
SCRAPE_INTERVAL_SEC = 2  # リクエスト間隔（秒）
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# netkeiba のベースURL
NETKEIBA_BASE_URL = "https://db.netkeiba.com"

# 対象年の範囲（デフォルト: 過去5年）
SCRAPE_YEAR_START = 2021
SCRAPE_YEAR_END = 2026

# 競馬場コード (netkeiba)
VENUE_CODES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}
