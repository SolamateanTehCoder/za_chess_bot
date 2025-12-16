#!/usr/bin/env python3
"""
All-Strategies Training System
Play all chess strategies against each other and train from both sides.
"""

import chess
import torch
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from hybrid_player import HybridChessPlayer
from strategy import ChessStrategy, STRATEGY_CONFIGS, StrategyAnalyzer


class AllStrategiesTrainer:
    """Train bot using all 8 chess strategies."""
    
    def __init__(self, device: str = None):
        """Initialize trainer with all strategies."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.strategies = list(STRATEGY_CONFIGS.keys())
        self.analyzer = StrategyAnalyzer()
        self.all_games = []
        
        print(f"\n[INIT] Strategies loaded: {len(self.strategies)}")
        for strat in self.strategies:
            print(f"  - {strat.ljust(18)} : {STRATEGY_CONFIGS[strat]['description']}")
    
    def play_all_combinations(self, games_per_matchup: int = 2) -> int:
        """
        Play every strategy against every other strategy.
        Each matchup plays games_per_matchup games (once as white, once as black).
        """
        print("\n" + "="*80)
        print("PLAYING ALL STRATEGY COMBINATIONS")
        print("="*80)
        
        total_games = 0
        total_moves = 0
        total_time = 0.0
        
        # Generate all unique pairs
        pairs = []
        for i, s1 in enumerate(self.strategies):
            for s2 in self.strategies:
                pairs.append((s1, s2))
        
        print(f"\nTotal matchups: {len(pairs)}")
        print(f"Games per matchup: {games_per_matchup}")
        print(f"Total games to play: {len(pairs) * games_per_matchup}")
        print("\n" + "-"*80)
        
        for matchup_idx, (white_strat, black_strat) in enumerate(pairs):
            matchup_num = matchup_idx + 1
            print(f"\n[{matchup_num:2}/{len(pairs)}] {white_strat.upper().ljust(18)} vs {black_strat.upper().ljust(18)}")
            
            for game_in_matchup in range(games_per_matchup):
                game_data = self._play_single_game(white_strat, black_strat)
                
                if game_data:
                    self.all_games.append(game_data)
                    self.analyzer.record_game(white_strat, black_strat, 
                                            game_data['result'], game_data['moves'])
                    
                    total_games += 1
                    total_moves += game_data['moves']
                    total_time += game_data['time']
                    
                    print(f"  Game {game_in_matchup+1}: {game_data['result'].ljust(7)} "
                          f"({game_data['moves']:3} moves, {game_data['time']:.2f}s)")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total games: {total_games}")
        print(f"Total moves: {total_moves}")
        print(f"Avg moves/game: {total_moves/total_games:.1f}" if total_games > 0 else "N/A")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time/game: {total_time/total_games:.2f}s" if total_games > 0 else "N/A")
        print(f"Avg move time: {total_time/total_moves*1000:.1f}ms" if total_moves > 0 else "N/A")
        
        return total_games
    
    def play_diverse_games(self, num_games: int = 20) -> int:
        """Play random diverse games for faster training."""
        print("\n" + "="*80)
        print(f"PLAYING {num_games} DIVERSE STRATEGY GAMES")
        print("="*80)
        
        total_moves = 0
        total_time = 0.0
        
        for game_num in range(num_games):
            white_strat = random.choice(self.strategies)
            black_strat = random.choice(self.strategies)
            
            print(f"\n[{game_num+1:2}/{num_games}] {white_strat.upper().ljust(18)} vs {black_strat.upper().ljust(18)}", end=" | ")
            
            game_data = self._play_single_game(white_strat, black_strat)
            
            if game_data:
                self.all_games.append(game_data)
                self.analyzer.record_game(white_strat, black_strat, 
                                        game_data['result'], game_data['moves'])
                
                total_moves += game_data['moves']
                total_time += game_data['time']
                
                print(f"{game_data['result'].ljust(7)} | {game_data['moves']:3} moves | {game_data['time']:.2f}s")
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Games played: {num_games}")
        print(f"Total moves: {total_moves}")
        print(f"Avg moves/game: {total_moves/num_games:.1f}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg move time: {total_time/total_moves*1000:.1f}ms" if total_moves > 0 else "N/A")
        
        return num_games
    
    def _play_single_game(self, white_strat: str, black_strat: str) -> Optional[Dict]:
        """Play a single game between two strategies."""
        try:
            # Create players
            player_white = HybridChessPlayer(device=self.device)
            player_black = HybridChessPlayer(device=self.device)
            
            # Set strategies
            player_white.strategy = ChessStrategy.from_config(white_strat)
            player_black.strategy = ChessStrategy.from_config(black_strat)
            
            board = chess.Board()
            moves_played = 0
            game_start = time.time()
            positions = []
            
            # Play game up to 150 moves
            while not board.is_game_over() and moves_played < 150:
                if board.turn:
                    move = player_white.select_move(board)
                else:
                    move = player_black.select_move(board)
                
                if not move:
                    break
                
                # Save position data
                positions.append({
                    'fen': board.fen(),
                    'move': move,
                    'side': 'white' if board.turn else 'black'
                })
                
                board.push_uci(move)
                moves_played += 1
            
            game_time = time.time() - game_start
            
            return {
                'white_strategy': white_strat,
                'black_strategy': black_strat,
                'result': board.result(),
                'moves': moves_played,
                'time': game_time,
                'positions': positions,
                'final_fen': board.fen()
            }
            
        except Exception as e:
            print(f"[ERROR] Game failed: {e}")
            return None
    
    def save_games(self, filename: str = "all_strategy_games.jsonl") -> bool:
        """Save all games to file."""
        try:
            with open(filename, 'w') as f:
                for game in self.all_games:
                    f.write(json.dumps(game) + '\n')
            
            print(f"\n[SAVE] Saved {len(self.all_games)} games to {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save games: {e}")
            return False
    
    def print_strategy_stats(self):
        """Print detailed strategy performance statistics."""
        print("\n" + "="*80)
        print("STRATEGY PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Header
        print(f"\n{'Strategy'.ljust(20)} | {'Games':>5} | {'Wins':>4} | {'Draws':>4} | {'Losses':>4} | {'Win Rate':>8} | {'Score':>7}")
        print("-" * 80)
        
        # Calculate scores and sort
        strategy_scores = []
        for strategy in self.strategies:
            stats = self.analyzer.strategy_stats[strategy]
            games = stats['games']
            
            if games > 0:
                wins = stats['wins']
                draws = stats['draws']
                losses = stats['losses']
                win_rate = (wins / games) * 100
                score = (wins + draws/2) / games
                
                strategy_scores.append((strategy, wins, draws, losses, win_rate, score, games))
        
        # Sort by win rate descending
        strategy_scores.sort(key=lambda x: x[4], reverse=True)
        
        for strategy, wins, draws, losses, win_rate, score, games in strategy_scores:
            print(f"{strategy.ljust(20)} | {games:5} | {wins:4} | {draws:4} | {losses:4} | {win_rate:7.1f}% | {score:7.3f}")
        
        # Best strategies
        if strategy_scores:
            best = strategy_scores[0]
            print("\n" + "="*80)
            print(f"[BEST] {best[0].upper()} with {best[4]:.1f}% win rate and {best[5]:.3f} score")
            print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train bot with all chess strategies")
    parser.add_argument("--mode", choices=["diverse", "complete"], default="diverse",
                       help="diverse: random games, complete: all combinations")
    parser.add_argument("--games", type=int, default=20,
                       help="Number of games to play (for diverse mode)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save games to file")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AllStrategiesTrainer()
    
    # Run training
    if args.mode == "complete":
        print("\n[MODE] Complete - All strategy combinations")
        trainer.play_all_combinations(games_per_matchup=1)
    else:
        print(f"\n[MODE] Diverse - {args.games} random games")
        trainer.play_diverse_games(num_games=args.games)
    
    # Save games
    if not args.no_save:
        trainer.save_games()
    
    # Print statistics
    trainer.print_strategy_stats()
    
    print("\n[DONE] Strategy training complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
