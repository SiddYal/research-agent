@echo off
rem Double-click to start the Research Agent and open it in your browser.
setlocal
cd /d "%~dp0"
set PORT=8000

if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up Python packages. This takes a few minutes...
    python -m venv .venv || (echo Python 3.12+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
    ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt || (echo Package install failed. & pause & exit /b 1)
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo Add your OpenAI API key to the .env file that just opened, save it, then run start.bat again.
    notepad ".env"
    exit /b 0
)

echo Starting the Research Agent at http://localhost:%PORT%  (close this window to stop it)
echo Your browser opens once it's ready, usually in 10-30 seconds.
start "" /b ".venv\Scripts\python.exe" scripts\open_when_ready.py http://localhost:%PORT%
".venv\Scripts\python.exe" -m uvicorn research_agent.api.main:app --port %PORT%
pause
