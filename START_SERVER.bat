@echo off
echo =============================================
echo   SignBridge v2.0 - Auto Setup and Run
echo =============================================
echo.

python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not installed!
    echo Go to https://www.python.org/downloads/
    echo Check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found.
echo.
echo Installing required packages...
pip install flask flask-cors flask-socketio numpy eventlet
echo.
echo =============================================
echo   Starting SignBridge Backend  (v2.0)
echo   REST   ->  http://localhost:5000
echo   WS     ->  ws://localhost:5000
echo   Open frontend/index.html in Chrome
echo =============================================
echo.
cd backend
python app.py
pause