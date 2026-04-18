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

REM Force UTF-8 encoding so Python doesn't crash on Windows with cp1252 charmap errors
set PYTHONIOENCODING=utf-8

REM Kill any stale instances on these ports (silent)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 " 2^>nul') do taskkill /PID %%P /F >nul 2>&1
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501 " 2^>nul') do taskkill /PID %%P /F >nul 2>&1

REM Start FastAPI backend in background
start "SOLiGence FastAPI Backend" /B py -3.10 dashboard_app.py

REM Wait 3 seconds for FastAPI to initialise
timeout /t 3 /nobreak >nul

REM Start Streamlit frontend (foreground — Ctrl+C stops both via this window)
py -3.10 -m streamlit run streamlit_app.py --server.port 8501

pause
