# Chess Engine with Reinforcement Learning

A chess engine that learns to play chess through reinforcement learning by playing against Stockfish.

## Features

- **Deep Neural Network**: ResNet-style architecture with policy and value heads
- **Reinforcement Learning**: PPO (Proximal Policy Optimization) algorithm
- **Parallel Training**: Plays 15 games simultaneously using multi-threading
- **GPU Acceleration**: CUDA support for faster training
- **Move Lookahead**: Evaluates positions up to 18 moves ahead
- **Checkpoint System**: Save and resume training at any time
- **Comprehensive Charts**: Automatic generation of training progress charts including:
  - Win/Draw/Loss rates over time
  - Policy, Value, and Total loss curves
  - Moving average win rate (smoothed)
- **Automatic Stopping**: Training stops when 100% win rate is achieved

## Project Structure

```
Za Chess Bot/
├── train.py                 # Main training script
├── model.py                 # Neural network architecture
├── chess_env.py             # Chess environment and board encoding
├── stockfish_opponent.py    # Stockfish integration
├── parallel_player.py       # Multi-threaded game player
├── trainer.py               # Training logic (PPO)
├── utils.py                 # Utility functions and chart generation
├── config.py                # Configuration parameters
├── test_engine.py           # Testing and playing against the AI
├── requirements.txt         # Python dependencies
├── checkpoints/             # Saved model checkpoints
├── plots/                   # Training progress charts
└── README.md               # This file
```

## Installation

### 1. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Install Stockfish

Download Stockfish from: https://stockfishchess.org/download/

Extract the executable and update the `STOCKFISH_PATH` in `config.py` to point to the `stockfish.exe` file.

Example:
```python
STOCKFISH_PATH = "C:/path/to/stockfish/stockfish.exe"
```

### 3. Verify CUDA (Optional, for GPU training)

If you have an NVIDIA GPU and want to use CUDA:

```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Configuration

Edit `config.py` to customize training parameters:

- `NUM_PARALLEL_GAMES`: Number of games to play simultaneously (default: 15)
- `NUM_EPOCHS`: Maximum training epochs (default: 100000)
- `TARGET_WIN_RATE`: Target win rate to stop training (default: 100.0%)
- `LOOKAHEAD_MOVES`: Move lookahead depth (default: 18)
- `STOCKFISH_SKILL_LEVEL`: Stockfish difficulty 0-20 (default: 20)
- `STOCKFISH_DEPTH`: Stockfish search depth (default: 18)
- `LEARNING_RATE`: Learning rate for training (default: 0.001)
- `USE_CUDA`: Enable/disable GPU training (default: True)
- `SAVE_FREQUENCY`: Save checkpoint every N epochs (default: 10)

## Usage

### Start Training

```powershell
python train.py
```

The training process:
1. Plays 15 games against Stockfish in parallel
2. Collects experience from all games
3. Trains the neural network for 1 epoch
4. Records win/loss/draw statistics
5. Saves checkpoints and generates charts every 10 epochs
6. Automatically stops when 100% win rate is achieved
7. Repeats until target is reached or max epochs

### Resume Training

If training is interrupted, simply run `train.py` again and choose to load the latest checkpoint when prompted.

### Monitor Progress

Training logs are saved to `training.log` and printed to the console in real-time.

**Checkpoints** are saved every 10 epochs in the `checkpoints/` directory.

**Training charts** are automatically generated in the `plots/` directory every 10 epochs, showing:
- Win/Draw/Loss rates over time
- Win rate progress toward 100% target
- Policy loss, Value loss, and Total loss
- Smoothed win rate (moving average)

When the AI achieves 100% win rate, training automatically stops and saves a final model as `final_model_100percent.pt`.

## How It Works

### 1. Board Encoding
The chess board is encoded as a 119-channel 8x8 tensor containing:
- Piece positions (12 channels: 6 pieces × 2 colors)
- Game state information (castling rights, en passant, move count)
- Move history

### 2. Neural Network
- **Input**: 119×8×8 board representation
- **Body**: ResNet-style with 10 residual blocks
- **Policy Head**: Outputs probability distribution over 4,672 possible moves
- **Value Head**: Outputs position evaluation (-1 to +1)

### 3. Training Algorithm (PPO)
- Plays multiple games to collect experience
- Calculates discounted returns and advantages
- Updates policy using clipped objective function
- Updates value function to better predict game outcomes

### 4. Parallel Game Playing
- 10 games run simultaneously in separate threads
- Each game alternates between AI and Stockfish moves
- Experience from all games is aggregated for training

## Performance Notes

- **GPU Training**: Significantly faster with CUDA-enabled GPU
- **Training Time**: Each epoch takes ~5-15 minutes depending on hardware
- **Memory Usage**: ~2-4 GB GPU memory, ~4-8 GB system RAM
- **Convergence**: Meaningful improvement typically seen after 100+ epochs

## Troubleshooting

### Stockfish Not Found
Make sure `STOCKFISH_PATH` in `config.py` points to the correct executable.

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `NUM_RESIDUAL_BLOCKS` or `HIDDEN_SIZE` in `config.py`
- Set `USE_CUDA = False` to train on CPU

### Slow Training
- Ensure CUDA is properly configured
- Reduce `NUM_PARALLEL_GAMES` if system resources are limited
- Lower `STOCKFISH_DEPTH` for faster opponent moves

## Future Improvements

- Implement full Monte Carlo Tree Search (MCTS)
- Add self-play training (AI vs AI)
- Implement opening book
- Add evaluation metrics and ELO rating
- Support for distributed training across multiple machines

## License

This project is for educational purposes.
