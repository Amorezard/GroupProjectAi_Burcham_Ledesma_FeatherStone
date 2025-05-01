@echo off
echo Setting up Merrimack College Wayfinding AI environment...

REM Check for Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found. Please install Python 3.8 or later.
    exit /b 1
)

REM Check Python version
for /f "tokens=*" %%a in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VERSION=%%a
echo Using Python version %PY_VERSION%

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Failed to create virtual environment. Please install venv package.
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to install dependencies.
    exit /b 1
)

REM Test OSMnx installation
echo Testing OSMnx installation...
python test_osmnx.py
if %ERRORLEVEL% NEQ 0 (
    echo Warning: OSMnx test failed. Please check the error messages above.
    echo You may need to manually install osmnx and its dependencies.
    echo Try: pip install osmnx==2.0.2 geopandas==1.0.0 networkx==2.8.4
) else (
    echo OSMnx test successful!
)

echo.
echo Setup complete! To run the application:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo 2. Run the application:
echo    python app.py
echo.
echo Open http://localhost:5000 in your browser to use the application.

pause 