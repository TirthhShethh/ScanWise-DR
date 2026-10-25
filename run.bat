@echo off
setlocal
title ScanWise ? Medical Image Classification

echo.
echo  ====================================================
echo   ScanWise -- Retinal AI Research Prototype
echo  ====================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

:: Check streamlit
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] pip install failed. See above for details.
        pause
        exit /b 1
    )
)

:: Run environment check
echo [INFO] Checking environment...
python scripts\check_environment.py
if errorlevel 1 (
    echo [WARN] Some packages may be missing. Attempting to continue...
)

:: Check for trained model
python -c "from pathlib import Path; exit(0 if any(Path('models').glob('*.pth')) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] No trained model found.
    echo [INFO] Running smoke-test training (20 images, 2 epochs)...
    echo [INFO] For full training run: python -m src.train --task dr --epochs 30
    echo.
    python -m src.train --task dr --model efficientnet --smoke-test
    if errorlevel 1 (
        echo [WARN] Training failed. Dashboard will run without a model.
    )
)

echo.
echo [INFO] Starting Streamlit dashboard...
echo [INFO] Open: http://localhost:8501
echo.
python -m streamlit run app\dashboard.py --server.headless false

if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit failed to start. See above for details.
    pause
    exit /b 1
)
endlocal
