@echo off
chcp 65001 >nul
title 锂电池隔膜筛选工具
cd /d "%~dp0"

echo ============================================
echo   锂电池隔膜筛选工具 - 启动中...
echo   启动后请在浏览器打开: http://localhost:8501
echo ============================================
echo.

"C:\Users\Administrator\miniconda3\envs\pythonproject2\python.exe" -m streamlit run app/streamlit_app.py --server.headless false

pause
