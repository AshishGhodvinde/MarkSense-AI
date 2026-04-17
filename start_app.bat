@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found at .venv
    echo Create it first with:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   python -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting MarkSense AI...
".venv\Scripts\python.exe" app.py

if errorlevel 1 (
    echo.
    echo App exited with an error.
    pause
)
