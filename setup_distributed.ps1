#
# Quick-Start Setup for Distributed Chess Bot Training
# PowerShell version for Windows
#

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "     Chess Bot Distributed Training - Quick Start Setup" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+" -ForegroundColor Red
    Write-Host "         Download from: https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check Git
try {
    $gitVersion = git --version 2>&1
    Write-Host "[OK] Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git not found. Please install Git" -ForegroundColor Red
    Write-Host "         Download from: https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host ""
Write-Host "[INFO] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "       (This may take a few minutes...)" -ForegroundColor Yellow

try {
    & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    Write-Host "[OK] PyTorch installed" -ForegroundColor Green
} catch {
    Write-Host "[WARN] PyTorch installation had issues, continuing..." -ForegroundColor Yellow
}

try {
    & pip install python-chess numpy tqdm -q
    Write-Host "[OK] Other dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Check Stockfish
Write-Host ""
Write-Host "[INFO] Checking Stockfish..." -ForegroundColor Yellow

try {
    $stockfishVersion = stockfish --version 2>&1
    Write-Host "[OK] Stockfish found: $stockfishVersion" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Stockfish not found" -ForegroundColor Yellow
    Write-Host "       To install on Windows:" -ForegroundColor Yellow
    Write-Host "       1. Download from: https://www.stockfishchess.org/download/" -ForegroundColor Yellow
    Write-Host "       2. Or use: choco install stockfish" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    
    $continue = Read-Host "Continue without Stockfish? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# GitHub token
Write-Host ""
Write-Host "[INFO] GitHub Token Setup" -ForegroundColor Yellow
Write-Host ""
Write-Host "To use distributed training, you need a GitHub personal access token." -ForegroundColor Cyan
Write-Host "For instructions on creating a token, see:" -ForegroundColor Cyan
Write-Host "  https://docs.github.com/en/authentication/keeping-your-data-secure/creating-a-personal-access-token" -ForegroundColor Cyan
Write-Host ""

$github_token = Read-Host "Enter your GitHub token (or press Enter to skip)"

if ($github_token) {
    [Environment]::SetEnvironmentVariable("GITHUB_TOKEN", $github_token, "User")
    Write-Host "[OK] GitHub token saved to GITHUB_TOKEN environment variable" -ForegroundColor Green
    Write-Host "     (Note: You may need to restart your terminal for changes to take effect)" -ForegroundColor Yellow
} else {
    Write-Host "[SKIP] GitHub token not set. You can set it later with:" -ForegroundColor Yellow
    Write-Host "       [Environment]::SetEnvironmentVariable('GITHUB_TOKEN', 'your_token_here', 'User')" -ForegroundColor Yellow
}

# Verify repository setup
Write-Host ""
Write-Host "[INFO] Verifying Git repository..." -ForegroundColor Yellow

try {
    $remote = git config --get remote.origin.url
    Write-Host "[OK] Git repository configured: $remote" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Not a valid Git repository" -ForegroundColor Red
    Write-Host "        Run: git init" -ForegroundColor Red
    Write-Host "        Then: git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git" -ForegroundColor Red
    exit 1
}

# Create directories
Write-Host ""
Write-Host "[INFO] Creating directories..." -ForegroundColor Yellow

$dirs = @("checkpoints", "game_batches", ".github\workflows")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Host "[OK] Directories created/verified" -ForegroundColor Green

# Test imports
Write-Host ""
Write-Host "[INFO] Testing Python imports..." -ForegroundColor Yellow

try {
    python -c "import torch; print(f'PyTorch OK')" | Out-Null
    Write-Host "[OK] PyTorch import successful" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] PyTorch import failed" -ForegroundColor Red
    exit 1
}

try {
    python -c "import chess; print('python-chess OK')" | Out-Null
    Write-Host "[OK] python-chess import successful" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] python-chess import failed" -ForegroundColor Red
    exit 1
}

try {
    python -c "import numpy; print('numpy OK')" | Out-Null
    Write-Host "[OK] numpy import successful" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] numpy import failed" -ForegroundColor Red
    exit 1
}

# Display next steps
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "                        Setup Complete!" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host ""
Write-Host "1. Push code to GitHub (if not already):" -ForegroundColor Cyan
Write-Host "   git add .github/workflows/ game_generator.py train_with_batches.py" -ForegroundColor White
Write-Host "   git commit -m 'Add distributed training setup'" -ForegroundColor White
Write-Host "   git push" -ForegroundColor White
Write-Host ""
Write-Host "2. Enable GitHub Actions:" -ForegroundColor Cyan
Write-Host "   - Go to: https://github.com/YOUR_USER/YOUR_REPO/actions" -ForegroundColor White
Write-Host "   - Click 'Bullet Chess Game Generation'" -ForegroundColor White
Write-Host "   - Click 'Run workflow'" -ForegroundColor White
Write-Host ""
Write-Host "3. Start local training:" -ForegroundColor Cyan
Write-Host "   python train_with_batches.py" -ForegroundColor White
Write-Host ""
Write-Host "4. Monitor progress:" -ForegroundColor Cyan
Write-Host "   - GitHub Actions: https://github.com/YOUR_USER/YOUR_REPO/actions" -ForegroundColor White
Write-Host "   - Local: Check game_batches/ and merged_games.jsonl" -ForegroundColor White
Write-Host ""
Write-Host "For detailed documentation, see: DISTRIBUTED_TRAINING.md" -ForegroundColor Green
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to close"
