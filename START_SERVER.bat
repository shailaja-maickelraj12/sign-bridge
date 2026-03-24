@echo off
echo =============================================
echo   SignBridge - Auto Setup and Run
echo =============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python first:
    echo 1. Go to https://www.python.org/downloads/
    echo 2. Download Python 3.11 or newer
    echo 3. During install, CHECK the box "Add Python to PATH"
    echo 4. Run this file again after installing
    echo.
    pause
    exit /b 1
)

echo [OK] Python found.
echo.

REM Install dependencies
echo Installing required packages...
pip install flask flask-cors numpy
echo.

REM Start the server
echo =============================================
echo   Starting SignBridge Backend Server...
echo   Open frontend\index.html in Chrome
echo =============================================
echo.
cd backend
python app.py
pause
