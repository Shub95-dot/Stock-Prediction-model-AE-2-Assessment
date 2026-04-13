@echo off
title SOLiGence Barometer — Dashboard Launcher
echo.
echo  ==========================================
echo   SOLiGence IEAP Barometer Dashboard
echo  ==========================================
echo   [1] FastAPI backend  -^>  http://localhost:8000
echo   [2] Streamlit UI     -^>  http://localhost:8501
echo   Press Ctrl+C to stop
echo  ==========================================
echo.

cd /d "%~dp0"

REM Kill any stale instances on these ports (silent)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 " 2^>nul') do taskkill /PID %%P /F >nul 2>&1
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501 " 2^>nul') do taskkill /PID %%P /F >nul 2>&1

REM Start FastAPI backend in background
start "SOLiGence FastAPI Backend" /B .venv310\Scripts\python.exe dashboard_app.py

REM Wait 3 seconds for FastAPI to initialise
timeout /t 3 /nobreak >nul

REM Start Streamlit frontend (foreground — Ctrl+C stops both via this window)
.venv310\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501

pause
