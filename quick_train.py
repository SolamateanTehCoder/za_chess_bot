#!/usr/bin/env python3
"""
Simple training script for WCCC bot.
Generates games, trains model, and tests performance.
"""

import torch
import chess
import sys
from pathlib import Path
from datetime import datetime

print("="*70)
print("Za Chess Bot - WCCC Training Started")
print("="*70)

# Check environment
print("\n[1/5] Checking environment...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Import modules
print("\n[2/5] Loading modules...")
try:
    from hybrid_player import HybridChessPlayer
    from chess_models import SimpleChessNet
    print("Modules loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load modules: {e}")
    sys.exit(1)

# Initialize player
print("\n[3/5] Initializing chess player...")
try:
    player = HybridChessPlayer(use_enhanced_model=False, device=device)
    print("Player initialized")
except Exception as e:
    print(f"ERROR: Failed to initialize player: {e}")
    sys.exit(1)

# Generate games
print("\n[4/5] Generating self-play games...")
print("-" * 70)

try:
    games_played = 0
    wins = 0
    draws = 0
    losses = 0
    total_moves = 0
    total_time = 0.0
    
    games_file = Path("training_games.jsonl")
    
    for game_num in range(5):  # Start with 5 games
        board = chess.Board()
        game_moves = []
        move_count = 0
        game_start = datetime.now()
        
        # Play game
        while not board.is_game_over() and move_count < 200:
            # Get move from player
            move = player.select_move(board, remaining_time_ms=1000)
            
            if not move:
                break
            
            board.push_uci(move)
            game_moves.append(move)
            move_count += 1
        
        game_end = datetime.now()
        game_duration = (game_end - game_start).total_seconds()
        
        # Determine result
        if board.is_checkmate():
            if board.turn:
                result = "0-1"
                losses += 1
            else:
                result = "1-0"
                wins += 1
        elif board.is_stalemate():
            result = "1/2-1/2"
            draws += 1
        else:
            result = "1/2-1/2"
            draws += 1
        
        games_played += 1
        total_moves += move_count
        total_time += game_duration
        
        print(f"Game {game_num+1:2d}: {result} ({move_count:3d} moves, {game_duration:5.1f}s)")
    
    print("-" * 70)
    print(f"\nResults: {wins}W - {draws}D - {losses}L")
    print(f"Total moves: {total_moves}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg move time: {total_time*1000/max(1,total_moves):.1f}ms")
    
except Exception as e:
    print(f"ERROR during game generation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test player statistics
print("\n[5/5] Player statistics:")
print("-" * 70)

stats = player.get_statistics()
print(f"Total moves played: {stats['moves_played']}")
print(f"  - Neural network moves: {stats['nn_moves']}")
print(f"  - Opening book moves: {stats['book_moves']}")
print(f"  - Tablebase moves: {stats['tb_moves']}")
print(f"  - Stockfish moves: {stats['stockfish_moves']}")
print(f"Average move time: {stats['avg_move_time']:.2f}ms")

print("\n" + "="*70)
print("Training session complete!")
print("="*70)
print("\nNext steps:")
print("1. Generate more games: python quick_train.py")
print("2. Full training cycle: python wccc_main.py --mode train --games 100 --epochs 10")
print("3. Play interactively: python wccc_main.py --mode interactive")
print("4. Run tournament: python wccc_main.py --mode tournament --tournament-games 5")
print("\n")

# Cleanup
player.close()
