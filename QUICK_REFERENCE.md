# Quick Reference Card

## 🎯 3-Step Quick Start

```powershell
# 1. Push to GitHub
git add . && git commit -m "Add distributed training" && git push

# 2. Trigger game generation (GitHub Actions)
# Go to: https://github.com/YOUR_USER/YOUR_REPO/actions
# Click "Bullet Chess Game Generation" → "Run workflow"

# 3. Start training locally
$env:GITHUB_TOKEN = "your_token_here"
python train_with_batches.py
```

---

## 📁 File Quick Reference

| Command | File | Purpose |
|---------|------|---------|
| `python game_generator.py` | game_generator.py | Generate games (GitHub Actions) |
| `python train_with_batches.py` | train_with_batches.py | Train on batches (local) |
| `python monitor_training.py` | monitor_training.py | Monitor progress (dashboard) |
| `.\setup_distributed.ps1` | setup_distributed.ps1 | Setup environment |
| `setup_distributed.bat` | setup_distributed.bat | Setup (batch version) |

---

## 🔐 GitHub Token Setup

```powershell
# Set GitHub token (one time)
$env:GITHUB_TOKEN = "your_github_token"

# Or permanently (Windows):
[Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "your_token", "User")

# To create a token:
# GitHub Settings → Developer Settings → Personal Access Tokens
# Scopes needed: repo
```

---

## 📊 Monitoring Commands

```powershell
# Real-time dashboard
python monitor_training.py

# Show status once
python monitor_training.py --once

# Custom refresh interval
python monitor_training.py --interval 10
```

---

## 📂 Directory Structure

```
project/
├── .github/workflows/
│   └── game_generation.yml          # GitHub Actions workflow
├── checkpoints/
│   ├── model_checkpoint_game_*.pt   # Game checkpoints
│   └── model_checkpoint_distributed_*.pt
├── game_batches/
│   └── games_batch_*.jsonl          # Downloaded game batches
├── game_generator.py                 # Game generation script
├── train_with_batches.py             # Training coordinator
├── monitor_training.py               # Dashboard
├── games.jsonl                       # Local games log
├── merged_games.jsonl                # Merged training data
├── DISTRIBUTED_TRAINING.md           # Full documentation
└── SETUP.md                          # Quick-start guide
```

---

## 🚀 Common Tasks

### Download and Train
```powershell
python train_with_batches.py
```

### Check System Status
```powershell
python monitor_training.py --once
```

### Analyze Game Quality
```powershell
python analyze_games.py
```

### Check GitHub Actions
```
https://github.com/YOUR_USER/YOUR_REPO/actions
```

### View Game Releases
```
https://github.com/YOUR_USER/YOUR_REPO/releases
```

---

## ⚙️ Configuration

### Game Generation (GitHub Actions)

Edit `.github/workflows/game_generation.yml`:

```yaml
# Change schedule (every 5 hours by default)
schedule:
  - cron: '0 */5 * * *'

# Change game generation parameters
python game_generator.py 5 1000    # 5 hours, max 1000 games
```

### Local Training

Set environment variables:

```powershell
# Training duration (hours)
$env:TRAINING_DURATION = "5"

# GitHub token (for downloading releases)
$env:GITHUB_TOKEN = "your_token"
```

---

## 🔧 Troubleshooting

### No game releases found
```
✓ Run game generation workflow first (takes ~5 hours)
✓ Check GitHub Actions tab for status
```

### GitHub token issues
```
✓ Verify token has 'repo' scope
✓ Set: $env:GITHUB_TOKEN = "token_here"
✓ Test: python train_with_batches.py
```

### Stockfish not found (local only)
```
✓ Install: choco install stockfish
✓ Or download: https://www.stockfishchess.org/download/
```

### PyTorch issues
```
✓ Reinstall: pip install torch torchvision torchaudio
✓ Use CPU: Will work but slow (GPU recommended)
```

---

## 📈 Performance Expectations

| Operation | Time | Games/Output |
|-----------|------|-------------|
| Game Gen (5h, GitHub Actions) | 5 hours | 1000-1200 games |
| Training (5h, local GPU) | 5 hours | Processes 5000-10000 games |
| Download Batch | ~5 min | 50-100MB file |
| Merge Batches | ~1 min | Combines all into one file |
| Checkpoint Save | ~2 sec | 12-15MB file |
| Git Push | ~10 sec | Syncs to repository |

---

## 📝 File Descriptions

### game_generator.py
- **Use**: GitHub Actions game generation
- **Runs**: Every 5 hours automatically
- **Output**: game_batches/games_batch_TIMESTAMP.jsonl
- **Parameters**: `python game_generator.py HOURS MAX_GAMES`

### train_with_batches.py
- **Use**: Local training on batches
- **Runs**: On-demand (whenever you want)
- **Input**: Downloaded game batches
- **Output**: model_checkpoint_distributed_*.pt

### monitor_training.py
- **Use**: Real-time progress dashboard
- **Runs**: On-demand, separate terminal
- **Shows**: GitHub status, batches, training progress
- **Updates**: Every 30 seconds (configurable)

### .github/workflows/game_generation.yml
- **Use**: GitHub Actions automation
- **Runs**: Every 5 hours on schedule
- **Cost**: Free (within GitHub limits)
- **Output**: GitHub releases with game batches

---

## 📚 Documentation Files

| File | Content |
|------|---------|
| **DISTRIBUTED_TRAINING.md** | Complete architecture & setup guide |
| **SETUP.md** | Quick-start with step-by-step |
| **DELIVERY_SUMMARY.md** | Overview of what was delivered |
| **QUICK_REFERENCE.md** | This file - quick commands |

---

## ✅ Verification Checklist

Before starting, verify:

```powershell
# Check Python
python --version      # Should be 3.10+

# Check Git
git --version         # Should be installed

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check Python-Chess
python -c "import chess; print('OK')"

# Check Stockfish (optional, needed for local games only)
stockfish --version
```

---

## 🎯 Typical Usage Pattern

### First Time Setup
```powershell
1. .\setup_distributed.ps1          # Run setup
2. git push                          # Push code
3. Trigger GitHub Actions manually   # Wait ~5 hours
4. python train_with_batches.py     # Start training
```

### Ongoing Usage
```powershell
# In Terminal 1 (Game monitoring)
python monitor_training.py

# In Terminal 2 (Training)
python train_with_batches.py

# In Terminal 3 (Analysis - optional)
python analyze_games.py

# Check GitHub Actions
# Open: https://github.com/YOUR_USER/YOUR_REPO/actions
```

---

## 💾 Backup & Recovery

### Backup Checkpoints
```powershell
# All checkpoints are automatically backed up in git
# To manually backup:
git add checkpoints/
git commit -m "Checkpoint backup"
git push
```

### Recover from Checkpoint
```python
# Automatically done by train_with_batches.py
# Loads latest checkpoint from checkpoints/ directory
# Resumes from that point
```

---

## 🔄 Update Workflow

When you want to update the workflow:

```powershell
# Edit file
code .github/workflows/game_generation.yml

# Commit changes
git add .github/workflows/game_generation.yml
git commit -m "Update game generation schedule"
git push

# New schedule takes effect immediately
```

---

## 📊 Interpreting Dashboard Output

```
[GitHub Actions Game Generation]
✓ Status: Successful games generated recently
✓ Runs: Shows recent workflow execution status
✓ Releases: Shows uploaded game batches

[Local Training Progress]
• Downloaded Batches: Number of JSONL files cached
• Merged Games: All batches combined into one file
• Checkpoints: Model state saved locally

[System Status]
• Total Games: Number ready for training
• Disk Usage: Space used by all data
• Recommendations: What to do next
```

---

## 🚨 Error Messages & Fixes

| Error | Fix |
|-------|-----|
| `GITHUB_TOKEN not set` | Run: `$env:GITHUB_TOKEN = "token"` |
| `No releases found` | Run game gen workflow first |
| `Stockfish not found` | Install: `choco install stockfish` |
| `PyTorch import error` | Reinstall: `pip install torch` |
| `Out of memory` | Reduce training duration |
| `Git push failed` | Check internet, configure git |

---

## 📞 Getting Help

1. **Check documentation**: `DISTRIBUTED_TRAINING.md`
2. **Run dashboard**: `python monitor_training.py --once`
3. **Check logs**: GitHub Actions tab
4. **Rerun setup**: `.\setup_distributed.ps1`

---

## 🎓 Key Concepts

**Game Batch**: JSONL file with 1000+ games, created every 5 hours
**Checkpoint**: Model weights saved, used to resume training
**Merged Games**: All batches combined into single training file
**GitHub Release**: Artifact storage for game batches
**Distributed Training**: Games in cloud (GitHub Actions), training on GPU (local)

---

**Quick Reference Version**: 1.0  
**Last Updated**: January 2024  
**Usage**: Print this or keep in IDE for quick lookup
