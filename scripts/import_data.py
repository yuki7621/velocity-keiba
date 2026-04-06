"""
エクスポートしたデータをインポートする（別PC側で実行）

使い方:
  python scripts/import_data.py exports/keiba_export.tar.gz

  または Docker の場合:
  docker compose run keiba-ai python scripts/import_data.py exports/keiba_export.tar.gz
"""

import sys
import tarfile
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("使い方: python scripts/import_data.py <export_file.tar.gz>")
        sys.exit(1)

    archive_path = Path(sys.argv[1])
    if not archive_path.exists():
        print(f"ファイルが見つかりません: {archive_path}")
        sys.exit(1)

    print(f"インポート中: {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  📦 {member.name}")
        tar.extractall(".")

    print("\nインポート完了！")
    print("アプリを起動: streamlit run app.py")


if __name__ == "__main__":
    main()
