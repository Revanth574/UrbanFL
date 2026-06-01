@echo off
echo ============================================
echo  UrbanCityOS — Full Stack Launcher
echo ============================================
echo.

:: Start Backend (FastAPI)
echo [1/2] Starting Backend (FastAPI on port 8000)...
start "UrbanCityOS Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait for backend to be ready
echo      Waiting for backend to initialize...
timeout /t 8 /nobreak > nul

:: Start Frontend (Vite dev server)
echo [2/2] Starting Frontend (Vite on port 5173)...
start "UrbanCityOS Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ============================================
echo  UrbanCityOS is starting up!
echo.
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:5173
echo  API Docs: http://localhost:8000/docs
echo ============================================
echo.
echo Press any key to open the dashboard...
pause > nul
start http://localhost:5173
