#!/usr/bin/env python3
"""Quick test of strategy system - generate games with different strategies."""

import sys
import torch
from pathlib import Path
from hybrid_player import HybridChessPlayer
from strategy import ChessStrategy, STRATEGY_CONFIGS
import chess
import time

import chess

def test_single_strategy_game():
    """Test a single strategy playing against another."""
    print("\n" + "="*70)
    print("Testing Single Strategy Games")
    print("="*70)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create two players with different strategies
        player1 = HybridChessPlayer(device=device)
        player2 = HybridChessPlayer(device=device)
        
        # Assign strategies
        strategy_aggressive = ChessStrategy.from_config("aggressive")
        strategy_defensive = ChessStrategy.from_config("defensive")
        
        player1.strategy = strategy_aggressive
        player2.strategy = strategy_defensive
        
        print(f"\n▶ Player 1 Strategy: AGGRESSIVE")
        print(f"▶ Player 2 Strategy: DEFENSIVE")
        print("-" * 70)
        
        # Play game
        board = chess.Board()
        move_count = 0
        start_time = time.time()
        
        while not board.is_game_over() and move_count < 150:
            if board.turn:
                move = player1.select_move(board)
                strategy_used = "Aggressive"
            else:
                move = player2.select_move(board)
                strategy_used = "Defensive"
            
            if move:
                board.push(chess.Move.from_uci(move))
                move_count += 1
                if move_count % 10 == 0:
                    print(f"  Move {move_count}: {move} ({strategy_used})")
            else:
                print(f"  ERROR: No move found!")
                break
        
        elapsed = time.time() - start_time
        
        print("-" * 70)
        print(f"Game ended: {board.result()}")
        print(f"Total moves: {move_count}")
        print(f"Duration: {elapsed:.2f}s")
        print(f"Avg move time: {elapsed/move_count*1000:.1f}ms")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in single strategy game: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_strategies():
    """Test that all strategies can select moves."""
    print("\n" + "="*70)
    print("Testing All Strategy Move Selection")
    print("="*70)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        player = HybridChessPlayer(device=device)
        board = chess.Board()
        
        results = {}
        for strategy_name in STRATEGY_CONFIGS.keys():
            strategy = ChessStrategy.from_config(strategy_name)
            player.strategy = strategy
            
            move = player.select_move(board)
            if move:
                results[strategy_name] = "[OK]"
                print(f"  {strategy_name:20} -> {move:6} [OK]")
            else:
                results[strategy_name] = "[FAIL]"
                print(f"  {strategy_name:20} -> NO MOVE [FAIL]")
        
        success = all(v == "[OK]" for v in results.values())
        print("-" * 70)
        print(f"Summary: {sum(1 for v in results.values() if v == '[OK]')}/{len(results)} strategies working")
        return success
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_strategy_tournament():
    """Test 5 games with different random strategy combinations."""
    print("\n" + "="*70)
    print("Testing Mixed Strategy Tournament (5 Games)")
    print("="*70)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        strategies = list(STRATEGY_CONFIGS.keys())
        
        results = []
        for game_num in range(5):
            import random
            white_strat = random.choice(strategies)
            black_strat = random.choice(strategies)
            
            print(f"\nGame {game_num+1}: {white_strat.upper()} (White) vs {black_strat.upper()} (Black)")
            print("-" * 70)
            
            player_white = HybridChessPlayer(device=device)
            player_black = HybridChessPlayer(device=device)
            
            player_white.strategy = ChessStrategy.from_config(white_strat)
            player_black.strategy = ChessStrategy.from_config(black_strat)
            
            board = chess.Board()
            move_count = 0
            start_time = time.time()
            
            while not board.is_game_over() and move_count < 150:
                if board.turn:
                    move = player_white.select_move(board)
                else:
                    move = player_black.select_move(board)
                
                if move:
                    board.push(chess.Move.from_uci(move))
                    move_count += 1
                else:
                    break
            
            elapsed = time.time() - start_time
            result = board.result()
            
            print(f"Result: {result} | Moves: {move_count} | Time: {elapsed:.2f}s")
            results.append({
                'white': white_strat,
                'black': black_strat,
                'result': result,
                'moves': move_count,
                'time': elapsed
            })
        
        print("\n" + "="*70)
        print("TOURNAMENT SUMMARY")
        print("="*70)
        total_moves = sum(r['moves'] for r in results)
        total_time = sum(r['time'] for r in results)
        avg_moves = total_moves / len(results)
        avg_time = total_time / len(results)
        
        print(f"Games: {len(results)}")
        print(f"Total moves: {total_moves}")
        print(f"Avg moves per game: {avg_moves:.1f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per game: {avg_time:.2f}s")
        print(f"Avg move time: {total_time/total_moves*1000:.1f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("[TEST] CHESS STRATEGY SYSTEM TEST")
    print("="*70)
    
    test1 = test_all_strategies()
    test2 = test_single_strategy_game()
    test3 = test_mixed_strategy_tournament()
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"All strategies working:  {('[PASS]' if test1 else '[FAIL]')}")
    print(f"Single strategy game:    {('[PASS]' if test2 else '[FAIL]')}")
    print(f"Mixed strategy games:    {('[PASS]' if test3 else '[FAIL]')}")
    
    all_pass = test1 and test2 and test3
    print("\n" + (("[PASS] ALL TESTS PASSED!" if all_pass else "[FAIL] SOME TESTS FAILED")))
    print("="*70 + "\n")
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
