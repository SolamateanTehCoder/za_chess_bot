"""
Strategy-Based Self-Play Training System.
Trains bot by playing different strategies against each other.
Learns from both sides of the board using diverse playstyles.
"""

import chess
import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time as time_module
from collections import defaultdict

from hybrid_player import HybridChessPlayer
from chess_strategies import StrategyPlayer, ChessStrategy, StrategyEvaluator
from advanced_trainer import AdvancedTrainer
from chess_models import ChessNetV2


class StrategyTrainingGame:
    """Represents a single game between two strategies."""
    
    def __init__(self, white_strategy: ChessStrategy, black_strategy: ChessStrategy,
                 board: chess.Board = None):
        self.white_strategy = white_strategy
        self.black_strategy = black_strategy
        self.board = board if board else chess.Board()
        self.moves = []
        self.result = None
        self.move_times = []
        self.start_time = None
        self.end_time = None
    
    def to_dict(self) -> Dict:
        """Convert game to dictionary for JSON storage."""
        return {
            "white_strategy": self.white_strategy.value,
            "black_strategy": self.black_strategy.value,
            "moves": [m.uci() for m in self.moves],
            "result": self.result,
            "move_count": len(self.moves),
            "duration_seconds": (self.end_time - self.start_time) if self.start_time else 0,
            "timestamp": datetime.now().isoformat()
        }


class StrategyMatchup:
    """Track results between two specific strategies."""
    
    def __init__(self, strategy1: ChessStrategy, strategy2: ChessStrategy):
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        self.games = []
        self.wins_s1 = 0
        self.wins_s2 = 0
        self.draws = 0
    
    def add_game_result(self, game: StrategyTrainingGame):
        """Add game result to matchup."""
        self.games.append(game)
        
        if "1-0" in game.result:
            self.wins_s1 += 1
        elif "0-1" in game.result:
            self.wins_s2 += 1
        else:
            self.draws += 1
    
    def get_stats(self) -> Dict:
        """Get statistics for this matchup."""
        total = len(self.games)
        if total == 0:
            return {}
        
        return {
            "strategy1": self.strategy1.value,
            "strategy2": self.strategy2.value,
            "total_games": total,
            "strategy1_wins": self.wins_s1,
            "strategy2_wins": self.wins_s2,
            "draws": self.draws,
            "strategy1_win_rate": self.wins_s1 / total,
            "strategy2_win_rate": self.wins_s2 / total,
            "draw_rate": self.draws / total
        }


class StrategyTrainer:
    """
    Trains chess bot by playing all strategies against each other.
    Learns from both sides of every game.
    """
    
    def __init__(self, player: HybridChessPlayer, device: str = "cuda"):
        """
        Initialize strategy trainer.
        
        Args:
            player: HybridChessPlayer instance
            device: "cuda" or "cpu"
        """
        self.player = player
        self.device = device
        self.trainer = AdvancedTrainer(player.model, device)
        self.strategy_players = {}
        self.matchups = defaultdict(lambda: defaultdict(StrategyMatchup))
        self.all_games = []
        self.games_data_file = "strategy_games.jsonl"
        
        # Initialize strategy players
        self._init_strategy_players()
    
    def _init_strategy_players(self):
        """Initialize StrategyPlayer for each strategy."""
        for strategy in ChessStrategy:
            self.strategy_players[strategy] = StrategyPlayer(
                self.player, strategy
            )
    
    def play_game_between_strategies(self, white_strategy: ChessStrategy,
                                     black_strategy: ChessStrategy,
                                     max_moves: int = 300) -> StrategyTrainingGame:
        """
        Play a game between two strategies.
        White uses white_strategy, Black uses black_strategy.
        """
        game = StrategyTrainingGame(white_strategy, black_strategy)
        game.start_time = time_module.time()
        
        white_player = self.strategy_players[white_strategy]
        black_player = self.strategy_players[black_strategy]
        
        move_count = 0
        
        while not game.board.is_game_over() and move_count < max_moves:
            # White's turn
            if game.board.turn == chess.WHITE:
                move = white_player.select_move_with_strategy(game.board, temperature=0.3)
            else:
                move = black_player.select_move_with_strategy(game.board, temperature=0.3)
            
            if move is None:
                break
            
            game.board.push(move)
            game.moves.append(move)
            move_count += 1
        
        game.end_time = time_module.time()
        game.result = game.board.result()
        
        return game
    
    def play_round_robin_tournament(self, num_games_per_matchup: int = 2) -> Dict:
        """
        Play round-robin tournament between all strategies.
        Each pair plays num_games_per_matchup games (both colors).
        """
        strategies = list(ChessStrategy)
        print(f"\n[TOURNAMENT] Starting round-robin with {len(strategies)} strategies")
        print(f"[TOURNAMENT] Each matchup: {num_games_per_matchup} games\n")
        
        tournament_games = []
        
        # Play all matchups
        for i, white_strat in enumerate(strategies):
            for black_strat in strategies:
                if white_strat == black_strat:
                    continue
                
                key = (white_strat.value, black_strat.value)
                matchup_key = tuple(sorted([white_strat.value, black_strat.value]))
                
                for game_num in range(num_games_per_matchup):
                    print(f"[GAME] {white_strat.value} (W) vs {black_strat.value} (B)... ", end="", flush=True)
                    
                    game = self.play_game_between_strategies(
                        white_strat, black_strat, max_moves=300
                    )
                    
                    tournament_games.append(game)
                    self.all_games.append(game)
                    
                    # Track matchup
                    if matchup_key not in self.matchups:
                        self.matchups[matchup_key] = StrategyMatchup(
                            white_strat, black_strat
                        )
                    
                    self.matchups[matchup_key].add_game_result(game)
                    
                    duration = game.end_time - game.start_time
                    print(f"{game.result} ({len(game.moves)} moves, {duration:.2f}s)")
        
        return self._get_tournament_stats()
    
    def _get_tournament_stats(self) -> Dict:
        """Get tournament statistics."""
        stats = {
            "total_games": len(self.all_games),
            "matchups": {},
            "strategy_performance": defaultdict(lambda: {
                "games_as_white": 0,
                "games_as_black": 0,
                "wins_as_white": 0,
                "wins_as_black": 0,
                "draws": 0,
                "total_wins": 0
            })
        }
        
        # Matchup stats
        for (s1, s2), matchup in self.matchups.items():
            stats["matchups"][f"{s1} vs {s2}"] = matchup.get_stats()
        
        # Strategy performance
        for game in self.all_games:
            white_perf = stats["strategy_performance"][game.white_strategy.value]
            black_perf = stats["strategy_performance"][game.black_strategy.value]
            
            white_perf["games_as_white"] += 1
            black_perf["games_as_black"] += 1
            
            if "1-0" in game.result:
                white_perf["wins_as_white"] += 1
                white_perf["total_wins"] += 1
            elif "0-1" in game.result:
                black_perf["wins_as_black"] += 1
                black_perf["total_wins"] += 1
            else:
                white_perf["draws"] += 1
                black_perf["draws"] += 1
        
        return stats
    
    def train_from_strategy_games(self, epochs: int = 10, batch_size: int = 32):
        """
        Train model from all strategy games.
        Uses experience from both white and black perspectives.
        """
        print(f"\n[TRAINING] Training on {len(self.all_games)} games, {epochs} epochs")
        
        # Convert games to training data
        positions = []
        move_labels = []
        game_results = []
        
        for game in self.all_games:
            # Replay game and collect positions
            board = chess.Board()
            
            for i, move in enumerate(game.moves):
                # Encode board state
                board_tensor = self.player.encode_board(board)
                positions.append(board_tensor)
                
                # Move label (one-hot over possible moves)
                legal_moves = list(board.legal_moves)
                move_idx = legal_moves.index(move) if move in legal_moves else 0
                move_labels.append(move_idx)
                
                # Game result (train from both perspectives)
                if game.result == "1/2-1/2":
                    result_value = 0.5
                elif board.turn == chess.WHITE:
                    result_value = 1.0 if "1-0" in game.result else 0.0
                else:
                    result_value = 1.0 if "0-1" in game.result else 0.0
                
                game_results.append(result_value)
                board.push(move)
        
        print(f"[TRAINING] Collected {len(positions)} positions from games")
        
        if len(positions) == 0:
            print("[WARN] No training data available")
            return
        
        # Train model
        positions_tensor = torch.stack(positions)
        move_labels_tensor = torch.tensor(move_labels, device=self.device)
        results_tensor = torch.tensor(game_results, dtype=torch.float32, device=self.device)
        
        self.trainer.train_epoch(
            positions_tensor,
            move_labels_tensor,
            results_tensor,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("[TRAINING] Training complete")
    
    def save_games(self, filename: str = None):
        """Save all games to JSONL file."""
        if filename is None:
            filename = self.games_data_file
        
        with open(filename, 'a') as f:
            for game in self.all_games:
                f.write(json.dumps(game.to_dict()) + '\n')
        
        print(f"[SAVE] Saved {len(self.all_games)} games to {filename}")
    
    def print_tournament_summary(self, stats: Dict):
        """Print tournament summary."""
        print("\n" + "="*80)
        print("STRATEGY TOURNAMENT SUMMARY")
        print("="*80)
        
        print(f"\nTotal Games: {stats['total_games']}")
        
        # Strategy rankings
        print("\nSTRATEGY PERFORMANCE RANKING:")
        print("-" * 80)
        
        perf_list = []
        for strat, perf in stats["strategy_performance"].items():
            if perf["games_as_white"] + perf["games_as_black"] > 0:
                total_games = perf["games_as_white"] + perf["games_as_black"]
                win_rate = perf["total_wins"] / total_games if total_games > 0 else 0
                perf_list.append((strat, perf["total_wins"], total_games, win_rate))
        
        perf_list.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (strat, wins, total, win_rate) in enumerate(perf_list, 1):
            print(f"{rank:2d}. {strat:15s} | Wins: {wins:2d} | Games: {total:2d} | Rate: {win_rate:.1%}")
        
        # Matchup details
        print("\nMATCHUP RESULTS:")
        print("-" * 80)
        
        for matchup_key, matchup_stats in stats["matchups"].items():
            print(f"{matchup_key}: {matchup_stats['strategy1_wins']}-{matchup_stats['strategy2_wins']}-{matchup_stats['draws']}")
        
        print("="*80 + "\n")


def main():
    """Main function to run strategy training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy-based training")
    parser.add_argument("--games-per-matchup", type=int, default=2,
                       help="Games per matchup (default: 2)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    
    args = parser.parse_args()
    
    # Initialize player
    print("[INIT] Loading chess bot...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    player = HybridChessPlayer(device=device)
    
    # Initialize strategy trainer
    print("[INIT] Initializing strategy trainer...")
    trainer = StrategyTrainer(player, device=device)
    
    # Run tournament
    print("[START] Running strategy tournament...\n")
    stats = trainer.play_round_robin_tournament(
        num_games_per_matchup=args.games_per_matchup
    )
    
    # Train from games
    trainer.train_from_strategy_games(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save and summarize
    trainer.save_games()
    trainer.print_tournament_summary(stats)
    
    # Save checkpoint
    checkpoint_path = f"model_strategy_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(player.model.state_dict(), checkpoint_path)
    print(f"[SAVE] Model checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
