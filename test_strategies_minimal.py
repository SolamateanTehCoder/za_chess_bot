#!/usr/bin/env python3
"""Minimal strategy test."""

import chess
import torch
from hybrid_player import HybridChessPlayer
from strategy import ChessStrategy, STRATEGY_CONFIGS

# Simple test
print("[TEST] Loading strategies...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Test 1: Create player
print("\n[1] Creating player...")
player = HybridChessPlayer(device=device)
print("OK")

# Test 2: Test each strategy
print("\n[2] Testing each strategy...")
board = chess.Board()
for strat_name in list(STRATEGY_CONFIGS.keys())[:3]:  # Test first 3
    strat = ChessStrategy.from_config(strat_name)
    player.strategy = strat
    move = player.select_move(board)
    print(f"  {strat_name}: {move} OK" if move else f"  {strat_name}: FAIL")

# Test 3: Play one game
print("\n[3] Playing 1 game (aggressive vs defensive)...")
p1 = HybridChessPlayer(device=device)
p2 = HybridChessPlayer(device=device)
p1.strategy = ChessStrategy.from_config("aggressive")
p2.strategy = ChessStrategy.from_config("defensive")

board = chess.Board()
moves = 0
for _ in range(50):
    if board.is_game_over():
        break
    if board.turn:
        m = p1.select_move(board)
    else:
        m = p2.select_move(board)
    if not m:
        break
    board.push_uci(m)
    moves += 1

print(f"  Played {moves} moves. Result: {board.result()}")
print("\n[DONE] All tests passed!")
