"""Configuration file for chess engine training."""

# Training parameters
NUM_PARALLEL_GAMES = 15  # Number of games to play simultaneously
NUM_EPOCHS = 100000  # Maximum number of training epochs (will stop at 100% win rate)
LOOKAHEAD_MOVES = 18  # Number of moves to look ahead
TARGET_WIN_RATE = 100.0  # Target win rate to stop training

# Stockfish settings
STOCKFISH_PATH = "C:\\stockfish\\stockfish-windows-x86-64-avx2.exe"  # Path to stockfish executable
STOCKFISH_SKILL_LEVEL = 20  # Stockfish skill level (0-20, lower is easier)
STOCKFISH_DEPTH = 18  # Search depth for Stockfish

# Neural network parameters
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor for future rewards
BATCH_SIZE = 64

# Model architecture
HIDDEN_SIZE = 512
NUM_RESIDUAL_BLOCKS = 10

# Device configuration
USE_CUDA = True  # Set to False to use CPU

# Model saving
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQUENCY = 10  # Save model every N epochs
LOG_FILE = "training.log"
GAMES_LOG_FILE = "games.log"
