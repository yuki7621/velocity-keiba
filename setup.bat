@echo off
chcp 65001 >nul 2>&1
echo ================================
echo   Keiba AI Setup
echo ================================
echo.

REM Python virtual environment
echo [1/4] Creating Python virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ first.
    pause
    exit /b 1
)
call .venv\Scripts\activate

REM Install dependencies
echo [2/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

REM Initialize DB (data/ dir is auto-created by create_tables)
echo [3/4] Initializing database...
python -c "from src.db.schema import create_tables; create_tables()"
if errorlevel 1 (
    echo ERROR: DB initialization failed.
    pause
    exit /b 1
)

REM Create desktop shortcut (with custom icon if available)
echo [4/4] Creating desktop shortcut...
powershell -NoProfile -Command ^
  "$ws = New-Object -ComObject WScript.Shell;" ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$sc = $ws.CreateShortcut(\"$desktop\競馬予想AI.lnk\");" ^
  "$sc.TargetPath = '%~dp0start.bat';" ^
  "$sc.WorkingDirectory = '%~dp0';" ^
  "$sc.Description = '競馬予想AI (Streamlit) を起動';" ^
  "if (Test-Path '%~dp0app_icon.ico') { $sc.IconLocation = '%~dp0app_icon.ico,0' };" ^
  "$sc.Save()"

echo.
echo ================================
echo   Setup complete!
echo ================================
echo.
echo How to start:
echo   start.bat  (or desktop shortcut)
echo.
echo Import data + models (REQUIRED for predictions):
echo   1. On old PC : python scripts\export_data.py
echo   2. Copy exports\keiba_export.tar.gz to this PC's exports\ folder
echo   3. On this PC: .venv\Scripts\activate
echo                  python scripts\import_data.py exports\keiba_export.tar.gz
echo.
pause
