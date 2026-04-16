@echo off
chcp 65001 >nul 2>&1
call .venv\Scripts\activate
streamlit run app.py
