@echo off
title SOLiGence Barometer Dashboard
echo.
echo  ==========================================
echo   SOLiGence IEAP Barometer Dashboard
echo  ==========================================
echo   Starting server at http://localhost:8000
echo   Press Ctrl+C to stop
echo  ==========================================
echo.

cd /d "%~dp0"
.venv\Scripts\python.exe dashboard_app.py

pause
