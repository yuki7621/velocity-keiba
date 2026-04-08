@echo off
echo ================================
echo   競馬予想AI セットアップ
echo ================================
echo.

REM Python仮想環境を作成
echo [1/3] Python仮想環境を作成中...
python -m venv .venv
call .venv\Scripts\activate

REM 依存パッケージをインストール
echo [2/3] 依存パッケージをインストール中...
pip install -r requirements.txt streamlit plotly

REM DB初期化
echo [3/3] データベースを初期化中...
python -c "from src.db.schema import create_tables; create_tables()"

echo.
echo ================================
echo   セットアップ完了！
echo ================================
echo.
echo 起動方法:
echo   .venv\Scripts\activate
echo   streamlit run app.py
echo.
echo データがある場合:
echo   python scripts\import_data.py exports\keiba_export.tar.gz
echo.
pause
