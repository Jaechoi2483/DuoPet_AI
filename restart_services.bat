@echo off
echo Restarting DuoPet AI Services...

echo.
echo [1/3] Stopping existing services...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *FastAPI*" 2>nul
taskkill /F /IM node.exe /FI "WINDOWTITLE eq *npm*" 2>nul
timeout /t 2 /nobreak >nul

echo.
echo [2/3] Starting AI Backend Service...
cd /d D:\final_project\DuoPet_AI
start "DuoPet AI Backend" cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 5 /nobreak >nul

echo.
echo [3/3] Starting Frontend Service...
cd /d D:\final_project\DuoPet_frontend
start "DuoPet Frontend" cmd /k "npm start"

echo.
echo All services have been restarted!
echo.
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause