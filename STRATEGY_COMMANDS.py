#!/usr/bin/env python3
"""
Strategy Training Command Reference
Quick commands to train the bot with all strategies.
"""

COMMANDS = """
╔════════════════════════════════════════════════════════════════════════════╗
║                   ALL-STRATEGIES TRAINING COMMANDS                         ║
║                                                                            ║
║                    Play all 8 strategies against each other               ║
║                   Train neural network from both sides                    ║
╚════════════════════════════════════════════════════════════════════════════╝

QUICK START
═══════════════════════════════════════════════════════════════════════════

1. Generate 5 quick games (1-2 minutes):
   python train_all_strategies.py --mode diverse --games 5

2. Generate 20 diverse games (5-8 minutes):
   python train_all_strategies.py --mode diverse --games 20

3. Test all 64 strategy combinations (20-30 minutes):
   python train_all_strategies.py --mode complete

4. Generate 100+ games for real training (30-45 minutes):
   python train_all_strategies.py --mode diverse --games 100


FULL TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════════

Generate games with all strategies:
   python train_all_strategies.py --mode diverse --games 200

Train neural network on generated games:
   python wccc_main.py --mode train --games 200 --epochs 10

Evaluate on tournament:
   python wccc_main.py --mode tournament --tournament-games 30


STRATEGY-SPECIFIC COMMANDS
═══════════════════════════════════════════════════════════════════════════

Play interactive game with aggressive strategy:
   python wccc_main.py --mode interactive --strategy aggressive

Play interactive game with defensive strategy:
   python wccc_main.py --mode interactive --strategy defensive

Play interactive game with balanced strategy:
   python wccc_main.py --mode interactive --strategy balanced


TESTING
═══════════════════════════════════════════════════════════════════════════

Quick test all strategies work:
   python test_strategies_minimal.py

Comprehensive strategy test:
   python test_strategies.py

Quick tournament between strategies:
   python quick_strategy_test.py


ANALYSIS
═══════════════════════════════════════════════════════════════════════════

Check generated games:
   ls -la *.jsonl
   
View strategy performance:
   python -c "from strategy import StrategyAnalyzer; print(StrategyAnalyzer())"

Analyze game quality:
   python analyze_games.py all_strategy_games.jsonl


8 AVAILABLE STRATEGIES
═══════════════════════════════════════════════════════════════════════════

1. aggressive    - Attack-focused, seeks checks and captures
2. defensive     - Safety-focused, avoids risks
3. positional    - Long-term advantage, control and structure
4. tactical      - Immediate tactics, combinations
5. endgame       - Piece promotion, king activity
6. opening       - Principled development, center control
7. balanced      - All factors equally important
8. machine_learning - Pure neural network (minimal heuristics)


EXPECTED RESULTS
═══════════════════════════════════════════════════════════════════════════

After 20 games:
  - 1,500+ total moves collected
  - 5-8 seconds total computation
  - Strategy performance analysis
  - Training data ready for neural network

After 100 games:
  - 7,500+ moves collected
  - 30-40 seconds total computation
  - Good diversity for training

After 500 games:
  - 37,500+ moves collected
  - 200+ seconds computation
  - Excellent training foundation


CUSTOM STRATEGY TRAINING
═══════════════════════════════════════════════════════════════════════════

Create custom strategy:
   python -c "
from hybrid_player import HybridChessPlayer
from strategy import ChessStrategy

config = {
    'capture_weight': 3.0,
    'check_weight': 4.0,
    'safety_weight': 6.0,
    'activity_weight': 4.0,
    'position_weight': 5.0,
    'endgame_weight': 3.0
}
strategy = ChessStrategy('custom', config)
player = HybridChessPlayer()
player.strategy = strategy
"


GPU OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════

Check GPU status:
   python wccc_setup.py verify

Enable GPU:
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

Check GPU memory:
   nvidia-smi


TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════

If games play slowly:
   - Use strategy "machine_learning" to skip heuristics
   - Disable Stockfish: --no-stockfish
   - Reduce analysis depth: --depth 10

If no .jsonl files created:
   - Check permissions in current directory
   - Check disk space
   - Run with --no-save to skip saving

If neural network doesn't improve:
   - Increase number of games (100+ needed)
   - Increase number of epochs (5-10 minimum)
   - Check loss values during training


EXAMPLE WORKFLOW
═══════════════════════════════════════════════════════════════════════════

# Phase 1: Quick test
python train_all_strategies.py --mode diverse --games 5

# Phase 2: Generate training data
python train_all_strategies.py --mode diverse --games 100

# Phase 3: Train on generated games
python wccc_main.py --mode train --games 100 --epochs 5

# Phase 4: Evaluate performance
python wccc_main.py --mode tournament --tournament-games 20

# Phase 5: Repeat with more data if needed
# (Phases 2-4 can be repeated multiple times)


FILE OUTPUTS
═══════════════════════════════════════════════════════════════════════════

Generated by train_all_strategies.py:
  - all_strategy_games.jsonl    (all games in JSONL format)
  - Training output             (console printed)

Generated by training pipeline:
  - checkpoints/latest.pt       (trained model)
  - training_log.json           (training metrics)

Loaded/Used:
  - games.jsonl                 (existing games)
  - checkpoints/game_*.pt       (model checkpoints)
  - openings.json               (opening book)


═══════════════════════════════════════════════════════════════════════════
Need help? Check:
  - ALL_STRATEGIES_GUIDE.md     (comprehensive guide)
  - WCCC_README.md              (WCCC documentation)
  - QUICKSTART.md               (quick start guide)
═══════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(COMMANDS)
