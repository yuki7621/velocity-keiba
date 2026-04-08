"""
データとモデルをエクスポートする（他のPCへの移行用）

使い方:
  python scripts/export_data.py

出力:
  exports/keiba_export.tar.gz
  → このファイルを別PCにコピーしてインポートする
"""

import shutil
import tarfile
from pathlib import Path

EXPORT_DIR = Path("exports")
EXPORT_FILE = EXPORT_DIR / "keiba_export.tar.gz"

FILES_TO_EXPORT = [
    "data/keiba.db",
    "models/lightgbm_v1.pkl",
    "models/lightgbm_v2.pkl",
    "models/lightgbm_v3.pkl",
]


def main():
    EXPORT_DIR.mkdir(exist_ok=True)

    print("エクスポート中...")
    with tarfile.open(EXPORT_FILE, "w:gz") as tar:
        for filepath in FILES_TO_EXPORT:
            p = Path(filepath)
            if p.exists():
                tar.add(filepath)
                size_mb = p.stat().st_size / 1024 / 1024
                print(f"  ✅ {filepath} ({size_mb:.1f} MB)")
            else:
                print(f"  ⏭️ {filepath} (存在しない - スキップ)")

    total_mb = EXPORT_FILE.stat().st_size / 1024 / 1024
    print(f"\n完了: {EXPORT_FILE} ({total_mb:.1f} MB)")
    print("このファイルを別PCにコピーしてください。")


if __name__ == "__main__":
    main()
