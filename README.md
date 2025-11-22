# Za Chess Bot - Reinforcement Learning Chess Engine

A chess engine trained through reinforcement learning, starting with games against Stockfish and continuing through self-play with bullet time control (1 minute per side).

## 🎯 Key Features

- **Two-Stage Training**:
  - Stage 1: Reinforcement learning against Stockfish (~1900 rating)
  - Stage 2: Self-play training with 28 games per epoch (14 white, 14 black)
  
- **Bullet Time Control**: 60 seconds per player - timeout counts as a loss, teaching the model to think faster

- **Comprehensive Chess Knowledge**: Both players have access to:
  - 500+ opening variations (Sicilian, Ruy Lopez, Italian, French, Caro-Kann, etc.)
  - 19 tactical patterns (pins, forks, skewers, discovered attacks, knight forks, etc.)
  - 40+ strategic concepts (control center, piece development, king safety, etc.)
  - 31 endgame principles (opposition, zugzwang, king activity, etc.)

- **Deep Neural Network**: 
  - ResNet-style architecture with 10 residual blocks
  - 512 hidden channels
  - ~58M trainable parameters
  - Separate policy (move selection) and value (position evaluation) heads

- **Advanced Training**:
  - PPO (Proximal Policy Optimization) algorithm
  - GPU-accelerated game playing with CUDA
  - Parallel multi-threaded games (28 simultaneously)
  - Automatic checkpoint saving every 10 epochs

- **Target Goal**: Train until 100.0 accuracy (100% win rate against itself)

## 📁 Project Structure

```
Za Chess Bot/
├── train.py                        # Original Stockfish training script
├── train_self_play.py              # Self-play training script (main)
├── model.py                        # Neural network architecture (ChessNet)
├── chess_env.py                    # Chess environment and board encoding
├── trainer.py                      # PPO training implementation
├── self_play_opponent.py           # Self-play game logic with time control
├── stockfish_opponent.py           # Stockfish opponent integration
├── parallel_player.py              # Multi-threaded parallel game player
├── comprehensive_chess_knowledge.py # Opening book, tactics, strategy, endgames
├── config.py                       # Configuration parameters
├── utils.py                        # Utility functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ⚙️ Installation

### 1. Clone Repository
```powershell
git clone https://github.com/SolamateanTehCoder/za_chess_bot.git
cd za_chess_bot
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

Required packages:
- PyTorch (with CUDA support recommended)
- python-chess 1.999
- NumPy
- Matplotlib (for charts)

### 3. Download Stockfish (for continuing Stockfish training)
Download from: https://stockfishchess.org/download/

Update `STOCKFISH_PATH` in `config.py`:
```python
STOCKFISH_PATH = "C:\\path\\to\\stockfish\\stockfish.exe"
```

### 4. Verify CUDA (Optional)
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Run Self-Play Training (Recommended)
```powershell
python train_self_play.py
```

This will:
1. Load the Stockfish-trained model as the base
2. Play 28 games per epoch (with 60-second time control per side)
3. Train the model for 1 epoch
4. Save checkpoints every 10 epochs
5. Continue until reaching 100.0 accuracy

### Run Stockfish Training (Optional)
```powershell
python train.py
```

This trains the initial model against Stockfish. Use this to restart training from scratch.

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Self-play training
NUM_EPOCHS = 100000              # Max epochs (stops at 100.0 accuracy)
LEARNING_RATE = 0.001            # Learning rate for PPO
BATCH_SIZE = 64                  # Training batch size

# Neural network
HIDDEN_SIZE = 512                # Channels in residual blocks
NUM_RESIDUAL_BLOCKS = 10         # Number of residual blocks
USE_CHESS_KNOWLEDGE = True       # Enable opening book, tactics, endgames

# Hardware
USE_CUDA = True                  # Enable GPU training
CHECKPOINT_DIR = "checkpoints"   # Checkpoint save directory
SAVE_FREQUENCY = 10              # Save checkpoint every N epochs
```

## 🎮 Game Configuration

Self-play training parameters (in `train_self_play.py`):
- **Games per epoch**: 28 (14 white, 14 black)
- **Time control**: 60 seconds per player (bullet)
- **Timeout penalty**: Counts as a loss
- **Knowledge access**: Both players have access to 500+ openings, 19 tactics, 40+ strategies, 31 endgames

## 🏆 Accuracy-Based Reward System (NEW!)

The training now uses **Stockfish analysis** to provide intelligent, accuracy-based rewards:

### How It Works:
1. **Every move is analyzed by Stockfish** at depth 15
2. **Accuracy score calculated** (0-100%) based on move quality
3. **Reward assigned** based on accuracy:
   - **+1.0**: Perfect move (best by Stockfish)
   - **+0.5**: Good move
   - **0.0**: Neutral move
   - **-0.5**: Bad move
   - **-1.0**: Blunder

### Time Pressure Learning:
- **1-second baseline** per move
- **Extra time = pain penalty**: -0.001 per extra millisecond
- Example: 2-second move = -1.0 penalty for time
- **Effect**: Model learns to think faster under pressure

### Real-Time Visualization:
- **28 game boards displayed simultaneously** (7×4 grid)
- **Green timer flash** = Model received reward (good move) ✓
- **Red timer flash** = Model received pain penalty (bad move) ✗
- **Accuracy % shown** for each side
- **Win/Loss/Draw displayed** at end of game

### Setting Up Rewards:

**Option 1: Install Stockfish (Recommended)**
```bash
# Windows: Download from https://stockfishchess.org/download/
# Place in C:\Program Files\Stockfish\

# Linux:
sudo apt-get install stockfish

# macOS:
brew install stockfish
```

**Option 2: Use Fallback Heuristics**
System will automatically fall back to simpler reward logic if Stockfish is unavailable (still effective).

### Running with Rewards:
```powershell
python run_training.py
```

See `REWARD_SYSTEM_GUIDE.md` for detailed documentation and troubleshooting.

## 📊 Training Progress

Training produces:
- **Console Output**: Real-time game results and loss metrics
- **Checkpoints**: Saved every 10 epochs in `checkpoints/`
- **Charts**: Training progress visualizations in `plots/` (win rate, loss curves)
- **Final Model**: `self_play_final_model.pt` when 100.0 accuracy is achieved

Example output:
```
[2025-11-18 21:06:15] EPOCH 11/100000
[2025-11-18 21:06:15] Phase 1: Playing self-play games...
[2025-11-18 21:10:45] Games completed
[2025-11-18 21:10:45]   Games played: 28
[2025-11-18 21:10:45]   Total moves: 5234
[2025-11-18 21:10:45]   Results - Wins: 8, Draws: 18, Losses: 2 (Timeouts: 0)
[2025-11-18 21:10:45]   Win Rate: 28.6%
[2025-11-18 21:10:45] Phase 2: Training neural network...
```

## 🧠 Neural Network Architecture

**Input**: 119×8×8 tensor encoding:
- Piece positions (12 channels: 6 types × 2 colors)
- Game state (castling rights, en passant, move count)
- Move history

**Architecture**:
```
Input (119, 8, 8)
    ↓
Conv + BatchNorm + ReLU (512 channels)
    ↓
10× Residual Blocks (512 channels each)
    ↓
┌─────────────────────┬─────────────────────┐
│   Policy Head       │   Value Head        │
├─────────────────────┼─────────────────────┤
│ Conv (512→32)       │ Conv (512→32)       │
│ FC (2048→4672)      │ FC (2048→256)       │
│ Log-softmax         │ FC (256→1)          │
│ (Move probs)        │ Tanh ([-1, 1])      │
└─────────────────────┴─────────────────────┘
     ↓                      ↓
  4672 moves           Position value
```

**Total Parameters**: ~57.9M

## 🎓 Training Algorithm

Uses **PPO (Proximal Policy Optimization)**:

1. **Self-Play**: Play 28 games with time control
2. **Experience Collection**: Record states, actions, rewards, log probabilities
3. **Advantage Calculation**: Compute returns and advantages
4. **Policy Update**: Update with clipped objective
5. **Value Update**: MSE loss on position evaluation
6. **Repeat**: Next epoch with improved model

## 🔧 Troubleshooting

### Training is slow
- Ensure CUDA is properly set up: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce `BATCH_SIZE` in config.py
- Check GPU memory usage

### Out of VRAM
- Reduce `HIDDEN_SIZE` from 512 to 256
- Reduce `NUM_RESIDUAL_BLOCKS` from 10 to 6-8
- Reduce `BATCH_SIZE` from 64 to 32

### Timeouts not working
- Check system clock is accurate
- Ensure timer logic in `self_play_opponent.py` is enabled
- Verify `USE_CHESS_KNOWLEDGE = True` in config.py

### Model not improving
- Ensure checkpoint is loading correctly
- Check that `USE_CHESS_KNOWLEDGE = True`
- Verify training data is being collected (check epoch results)

## 📈 Performance Notes

- **Hardware**: NVIDIA GTX 1650 or better recommended
- **Per Epoch Time**: ~3-5 minutes (with 28 games and CUDA)
- **Memory**: ~2GB GPU VRAM, ~4GB system RAM
- **Training Duration**: 100-500+ epochs to reach convergence

## 🎯 Project Goals

1. ✅ Train against Stockfish (~1900 rating)
2. ✅ Implement self-play with bullet time control
3. ✅ Add comprehensive chess knowledge (openings, tactics, endgames)
4. ⏳ Reach 100.0 accuracy (100% win rate against itself)
5. ⏳ Evaluate final model strength

## 📚 Chess Knowledge Included

**Openings** (500+):
- Sicilian Defense (Open, Closed, Najdorf, Dragon, etc.)
- Ruy Lopez (Spanish)
- Italian Game
- French Defense
- Caro-Kann Defense
- And many more...

**Tactics** (19 patterns):
- Pins, Forks, Skewers
- Discovered Attacks
- Double Attacks
- Knight Forks
- Removing Defenders
- And more...

**Strategy** (40+ concepts):
- Control the center
- Develop pieces quickly
- King safety (castling early)
- Piece activity and coordination
- Pawn structure
- And more...

**Endgames** (31 principles):
- Opposition and key squares
- Zugzwang
- King activity
- Pawn promotion techniques
- Rook and pawn endgames
- And more...

## 🤝 Contributing

This is a personal learning project. Feel free to fork and experiment!

## 📄 License

GNU GENERAL PUBLIC LICENSE

---

**Current Status**: Self-play training in progress with bullet time control  
**Latest Model**: `self_play_latest_checkpoint.pt`  
**Target**: 100.0 accuracy (100% win rate against itself)
