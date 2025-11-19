"""
REWARD SYSTEM AND REAL-TIME VISUALIZATION GUIDE
================================================

This document explains the new accuracy-based reward system and real-time game visualizer
that have been integrated into the chess engine training.

1. REWARD SYSTEM OVERVIEW
=========================

The training system now uses Stockfish analysis to provide intelligent, accuracy-based rewards
for every move the model makes during self-play. This helps the model learn which moves are
actually good and which are mistakes.

KEY COMPONENTS:

a) Stockfish Reward Analyzer (stockfish_reward_analyzer.py)
   - Analyzes every move using Stockfish engine at 20 depth
   - Calculates move accuracy based on evaluation change
   - Assigns rewards (+) for good moves and penalties (-) for bad moves
   - Applies time penalty for exceeding 1-second baseline per move

b) Move Timing and Time Penalties
   - Each move gets a 1-second baseline
   - Beyond 1 second: each extra millisecond incurs pain penalty
   - Formula: reward -= (excess_ms * 0.001)
   - This teaches the model to think faster under time pressure

c) Accuracy Scores
   - 100% accuracy = best move by Stockfish
   - 85% = good move (loses 20+ centipawns slightly)
   - 60% = okay move (loses 50+ centipawns)
   - 40% = slight mistake (loses 0-50 centipawns)
   - 20% = serious mistake (loses 200+ centipawns)
   - 0% = blunder (loses 200+ centipawns)

2. REWARD RANGES
================

The reward system operates on a [-1.0, +1.0] scale:

POSITIVE REWARDS (GREEN FLASH):
  +1.0   : Excellent move, perfect play
  +0.7   : Good move, beneficial position
  +0.3   : Okay move, maintains position
  
NEUTRAL:
   0.0   : Normal move, no evaluation change

NEGATIVE REWARDS / PAIN:
  -0.3   : Slight mistake, loses small advantage
  -0.7   : Bad move, loses significant material/position
  -1.0   : Blunder, catastrophic move

TIME PENALTIES (RED FLASH):
  -0.001 per ms : Penalty for using time beyond 1-second baseline
  Example: Taking 2000ms (2 seconds) = 1000ms excess × -0.001 = -1.0 penalty


3. REAL-TIME GAME VISUALIZER
=============================

The game_visualizer.py displays all 28 games simultaneously:

FEATURES:

a) Board Display
   - Shows all 28 chess boards in a 7×4 grid
   - Pieces displayed with Unicode chess symbols
   - White background for light squares, gray for dark squares

b) Timer Display
   - Shows remaining time for both sides (MM:SS | MM:SS)
   - White text = neutral (no recent reward)
   - GREEN TEXT = model received reward (positive feedback)
   - RED TEXT = model received pain penalty (negative feedback)
   - Color fades after 0.5 seconds

c) Accuracy Display
   - Shows White: X% | Black: X%
   - Rolling average accuracy for each side
   - Updates after each move

d) Result Display
   - Shows final result when game ends
   - GREEN = Win
   - RED = Loss
   - ORANGE = Draw

e) Status Bar
   - Top of window shows current epoch and overall statistics
   - Format: "Epoch N | Win Rate: X% | Accuracy: W:Y% B:Z% | Games: N"

4. HOW TO USE THE SYSTEM
========================

STARTING TRAINING:

    python train_self_play.py

This will:
1. Initialize the Stockfish reward analyzer
   - Searches for Stockfish executable automatically
   - Falls back to heuristic rewards if Stockfish not found
   
2. Launch the real-time game visualizer
   - 28 game boards displayed simultaneously
   - Updates every 100ms

3. Run self-play games with accuracy rewards
   - Each game up to 60 seconds per side (bullet)
   - Each move analyzed for accuracy
   - Rewards/penalties applied immediately

4. Train the neural network
   - Uses collected experiences with accurate rewards
   - PPO algorithm with accuracy-based learning signals

5. Save checkpoints every 10 epochs

MONITORING GAMES:

Watch the visualizer to see:
- Which moves trigger GREEN flashes (good moves)
- Which moves trigger RED flashes (bad moves)
- How quickly models learn to play better
- Average accuracy improving over epochs
- Win rate increasing toward 100%

5. INSTALLING STOCKFISH
=======================

Stockfish is required for optimal accuracy-based rewards.

WINDOWS:
  Download from: https://stockfishchess.org/download/
  Extract to: C:\Program Files\Stockfish\
  The system will auto-detect it
  
  OR set environment variable:
    $env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"

LINUX:
  sudo apt-get install stockfish
  
MAC:
  brew install stockfish

If Stockfish is not found, the system will use heuristic fallback rewards
(still effective but less accurate).

6. EXAMPLE OUTPUT
=================

[14:32:15] Starting self-play training...
[14:32:16] [SUCCESS] Stockfish found: Stockfish 16 by T. Romstad, M. Costalba, J. Kiiski, G. Linscott
[14:32:16] [INFO] Stockfish reward analyzer initialized and ready
[14:32:16] Launching real-time game visualizer...
[14:32:17] EPOCH 1/100000
[14:32:17] Phase 1: Playing self-play games...
[14:32:25] Games completed in 8.2s
[14:32:25]   Games played: 28
[14:32:25]   Total moves: 3456
[14:32:25]   Results - Wins: 14, Draws: 0, Losses: 14
[14:32:25]   Win Rate: 50.0%
[14:32:25]   Move Accuracy - White: 62.3%, Black: 59.8%
[14:32:25] Phase 2: Training neural network...
[14:32:35] Training completed
[14:32:35]   Policy Loss: 0.234567
[14:32:35]   Value Loss: 0.123456
[14:32:35]   Total Loss: 0.358023

7. METRICS EXPLAINED
====================

Win Rate:
  - Percentage of games won by the model
  - Target: 100% (100.0 accuracy)
  - Indicates overall training progress

Move Accuracy:
  - Average accuracy of all moves made
  - White/Black tracked separately
  - Improves as model learns better moves
  - Useful for detecting convergence

Policy Loss:
  - Measures how well policy predicts good moves
  - Should decrease over training

Value Loss:
  - Measures how well value head predicts game outcome
  - Should decrease over training

8. TROUBLESHOOTING
==================

Visualizer doesn't appear:
  - Check if Tkinter is installed (usually included)
  - Linux users may need: sudo apt-get install python3-tk
  - The training will still proceed, just without visualization

Stockfish analysis is slow:
  - Depth 15 analysis per move takes time
  - For faster training, depth can be lowered (in stockfish_reward_analyzer.py)
  - Set: depth=10 for faster, less accurate analysis

Training is slower than before:
  - Stockfish analysis adds ~100-200ms per move
  - Total time per epoch: ~10-15 seconds (vs ~5-10 before)
  - Accuracy benefits are worth the time cost

Out of memory errors:
  - 28 games × ~100 moves × Stockfish analysis = high memory
  - Reduce num_games in train_self_play.py if needed
  - Or reduce Stockfish depth

9. ADVANCED CONFIGURATION
==========================

Reward Parameters (in StockfishRewardAnalyzer):
  self.max_reward = 1.0              # Reward for perfect move
  self.min_penalty = -1.0            # Penalty for blunder
  self.time_penalty_per_ms = -0.001  # Pain per extra millisecond
  self.time_limit_ms = 1000          # 1 second baseline

Accuracy Thresholds (in analyze_move method):
  eval_change >= 50cp    → 100.0% accuracy (reward +1.0)
  eval_change >= 20cp    → 85.0% accuracy (reward +0.7)
  eval_change >= 0cp     → 60.0% accuracy (reward +0.3)
  eval_change >= -50cp   → 40.0% accuracy (reward -0.3)
  eval_change >= -200cp  → 20.0% accuracy (reward -0.7)
  eval_change < -200cp   → 0.0% accuracy (reward -1.0)

10. INTEGRATION WITH TRAINING
=============================

The reward system integrates seamlessly:

1. Game plays out normally
2. Each move is immediately analyzed by Stockfish
3. Accuracy score and reward calculated
4. Reward is stored in experience tuple:
   {
     'state': board_state,
     'action': move_index,
     'reward': stockfish_reward,  # <-- NEW
     'value': value_prediction,
     'log_prob': policy_log_prob,
     'accuracy': accuracy_score,  # <-- NEW
     'move_time_ms': move_time,   # <-- NEW
     'stockfish_analysis': analysis_dict  # <-- NEW
   }
5. Trainer uses reward signal for PPO updates
6. Model learns from accurate feedback

This replaces the simple environment reward with intelligent, Stockfish-verified signals.

END OF GUIDE
"""