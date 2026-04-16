@echo off
chcp 65001 >nul 2>&1
echo ================================
echo   Keiba AI Setup
echo ================================
echo.

REM Python virtual environment
echo [1/3] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate

REM Install dependencies
echo [2/3] Installing dependencies...
pip install -r requirements.txt streamlit plotly

REM Initialize DB
echo [3/3] Initializing database...
python -c "from src.db.schema import create_tables; create_tables()"

echo.
echo ================================
echo   Setup complete!
echo ================================
echo.
echo How to start:
echo   start.bat
echo.
echo Import data (if available):
echo   .venv\Scripts\activate
echo   python scripts\import_data.py exports\keiba_export.tar.gz
echo.
pause
