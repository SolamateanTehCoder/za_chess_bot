@echo off
REM
REM Quick-Start Setup for Distributed Chess Bot Training
REM Windows PowerShell compatible batch script
REM

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo     Chess Bot Distributed Training - Quick Start Setup
echo ========================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+
    echo         Download from: https://www.python.org/downloads/
    exit /b 1
)

echo [OK] Python found
python --version

REM Check Git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Please install Git
    echo         Download from: https://git-scm.com/download/win
    exit /b 1
)

echo [OK] Git found
git --version

REM Install dependencies
echo.
echo [INFO] Installing Python dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
if errorlevel 1 (
    echo [WARN] PyTorch installation might have issues, continuing...
)

pip install python-chess numpy tqdm -q
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    exit /b 1
)

echo [OK] Dependencies installed

REM Check Stockfish
echo.
echo [INFO] Checking Stockfish...
where stockfish >nul 2>&1
if errorlevel 1 (
    echo [WARN] Stockfish not found
    echo        To install on Windows:
    echo        1. Download from: https://www.stockfishchess.org/download/
    echo        2. Or use: choco install stockfish
    echo.
    echo        You can continue without Stockfish, but local game generation won't work.
    set /p continue="Continue without Stockfish? (y/n): "
    if /i "!continue!" neq "y" exit /b 1
) else (
    echo [OK] Stockfish found
    stockfish --version
)

REM GitHub token
echo.
echo [INFO] GitHub Token Setup
echo.
echo To use distributed training, you need a GitHub personal access token.
echo For instructions on creating a token, see:
echo   https://docs.github.com/en/authentication/keeping-your-data-secure/creating-a-personal-access-token
echo.
set /p github_token="Enter your GitHub token (or press Enter to skip): "

if not "!github_token!"=="" (
    setx GITHUB_TOKEN "!github_token!"
    echo [OK] GitHub token saved to GITHUB_TOKEN environment variable
    echo     (Changes take effect in new terminal windows)
) else (
    echo [SKIP] GitHub token not set. You can set it later with:
    echo       setx GITHUB_TOKEN "your_token_here"
)

REM Verify repository setup
echo.
echo [INFO] Verifying Git repository...
git config --get remote.origin.url >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a valid Git repository
    echo        Run: git init
    echo        Then: git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
    exit /b 1
)

echo [OK] Git repository configured
git config --get remote.origin.url

REM Create directories
echo.
echo [INFO] Creating directories...
if not exist "checkpoints" mkdir checkpoints
if not exist "game_batches" mkdir game_batches
if not exist ".github\workflows" mkdir .github\workflows

echo [OK] Directories created

REM Test imports
echo.
echo [INFO] Testing Python imports...
python -c "import torch; print(f'PyTorch {torch.__version__}')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch import failed
    exit /b 1
)

python -c "import chess; print('python-chess OK')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python-chess import failed
    exit /b 1
)

echo [OK] All Python imports successful

REM Display next steps
echo.
echo ========================================================================
echo                        Setup Complete!
echo ========================================================================
echo.
echo Next Steps:
echo.
echo 1. Push code to GitHub (if not already):
echo    git add .github/workflows/ game_generator.py train_with_batches.py
echo    git commit -m "Add distributed training setup"
echo    git push
echo.
echo 2. Enable GitHub Actions:
echo    - Go to: https://github.com/YOUR_USER/YOUR_REPO/actions
echo    - Click "Bullet Chess Game Generation"
echo    - Click "Run workflow"
echo.
echo 3. Start local training:
echo    python train_with_batches.py
echo.
echo 4. Monitor progress:
echo    - GitHub Actions: https://github.com/YOUR_USER/YOUR_REPO/actions
echo    - Local: Check game_batches/ and merged_games.jsonl
echo.
echo For detailed documentation, see: DISTRIBUTED_TRAINING.md
echo.

pause
