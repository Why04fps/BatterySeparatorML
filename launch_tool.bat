@echo off
chcp 65001 >nul
title Battery Separator Screening Tool
cd /d "%~dp0"

echo ============================================
echo   Battery Separator Screening Tool
echo   Starting Streamlit server...
echo ============================================
echo.
start "" http://localhost:8501
"C:\Users\Administrator\miniconda3\envs\pythonproject2\python.exe" -m streamlit run app/streamlit_app.py --server.headless false
