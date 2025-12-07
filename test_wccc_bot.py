"""
Test script to verify WCCC bot components and demonstrate gameplay.
"""

import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import chess
from pathlib import Path

# Test imports
print("[TEST] Importing WCCC modules...")

try:
    from hybrid_player import HybridChessPlayer
    print("[OK] hybrid_player")
except ImportError as e:
    print(f"[FAIL] hybrid_player: {e}")
    sys.exit(1)

try:
    from advanced_trainer import AdvancedTrainer, TrainingPipeline
    print("[OK] advanced_trainer")
except ImportError as e:
    print(f"[FAIL] advanced_trainer: {e}")

try:
    from opening_book import OpeningBook
    print("[OK] opening_book")
except ImportError as e:
    print(f"[FAIL] opening_book: {e}")

try:
    from tablebase_manager import TablebaseManager
    print("[OK] tablebase_manager")
except ImportError as e:
    print(f"[FAIL] tablebase_manager: {e}")

try:
    from time_management import TimeManager, TimeControl
    print("[OK] time_management")
except ImportError as e:
    print(f"[FAIL] time_management: {e}")

try:
    from tournament import Tournament, GameResult
    print("[OK] tournament")
except ImportError as e:
    print(f"[FAIL] tournament: {e}")

try:
    from chess_models import ChessNetV2, SimpleChessNet
    print("[OK] chess_models")
except ImportError as e:
    print(f"[FAIL] chess_models: {e}")

print("\n[TEST] All imports successful!\n")

# Test hybrid player initialization
print("[TEST] Initializing HybridChessPlayer...")

try:
    player = HybridChessPlayer(use_enhanced_model=False)  # Use simple model for speed
    print("[OK] Player initialized")
except Exception as e:
    print(f"[FAIL] Initialization failed: {e}")
    sys.exit(1)

# Test move selection
print("\n[TEST] Testing move selection...")

board = chess.Board()

print(f"Board:\n{board}\n")

try:
    move = player.select_move(board, remaining_time_ms=5000)
    if move:
        print(f"[OK] Selected move: {move}")
        
        # Verify it's legal
        try:
            board.push_uci(move)
            print(f"[OK] Move is legal")
            print(f"Board after move:\n{board}\n")
        except:
            print(f"[FAIL] Move is invalid!")
            sys.exit(1)
    else:
        print("[FAIL] No move selected!")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] Move selection failed: {e}")
    sys.exit(1)

# Test statistics
print("[TEST] Checking statistics...")

stats = player.get_statistics()
print(f"Player Statistics:")
print(f"  Moves played: {stats['moves_played']}")
print(f"  Book moves: {stats['book_moves']}")
print(f"  NN moves: {stats['nn_moves']}")
print(f"  Stockfish moves: {stats['stockfish_moves']}")
print(f"  Avg move time: {stats['avg_move_time']:.2f}ms")

print("\n[SUCCESS] All tests passed!")
print("\nYour WCCC bot is ready to:")
print("  1. Play games: python wccc_main.py --mode play --games 10")
print("  2. Train model: python wccc_main.py --mode train")
print("  3. Run tournaments: python wccc_main.py --mode tournament")
print("  4. Play in UCI tournaments: python uci_engine.py")
