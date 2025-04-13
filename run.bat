@echo off
echo Cornix Trading Signal Analysis Tool Setup

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed.
    echo.
    echo To install Python on Windows:
    echo 1. Visit https://www.python.org/downloads/windows/
    echo 2. Download the latest Python 3.x installer
    echo 3. Run the installer
    echo 4. IMPORTANT: Check "Add Python to PATH" during installation
    echo.
    echo After installation, please run this script again.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Run the script
echo Starting Cornix Trading Signal Analysis Tool...
python cornix-stats.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause 