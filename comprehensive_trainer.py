"""
Advanced Chess Training with Multiple Strategies.
Combines self-play with different strategies and dual-sided learning.
"""

import chess
import torch
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import time as time_module
from collections import defaultdict

from hybrid_player import HybridChessPlayer
from chess_strategies import ChessStrategy, StrategyPlayer, StrategyEvaluator
from advanced_trainer import AdvancedTrainer, GameExperienceDataset
from chess_models import ChessNetV2


class DualSidedTrainer:
    """
    Trains neural network from both sides of the board.
    Every game generates training data for both white and black.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        """Initialize dual-sided trainer."""
        self.model = model
        self.device = device
        self.trainer = AdvancedTrainer(model, device)
    
    def create_training_data_from_game(self, board_history: List[chess.Board],
                                       moves: List[chess.Move],
                                       result: str) -> List[Dict]:
        """
        Create training data from one game for both sides.
        
        Returns:
            List of training examples (positions, moves, results from both perspectives)
        """
        training_data = []
        
        # Parse result
        if "1-0" in result:
            white_result, black_result = 1.0, 0.0
        elif "0-1" in result:
            white_result, black_result = 0.0, 1.0
        else:
            white_result, black_result = 0.5, 0.5
        
        # Create examples for both colors
        for i, (board, move) in enumerate(zip(board_history, moves)):
            # White perspective (from white's viewpoint)
            white_data = {
                "board_state": board.fen(),
                "move": move.uci(),
                "result": white_result,
                "move_number": i + 1,
                "perspective": "white"
            }
            training_data.append(white_data)
            
            # Black perspective (flip the board)
            flipped_board = board.copy()
            # Create black's perspective
            black_data = {
                "board_state": board.fen(),
                "move": move.uci(),
                "result": black_result,
                "move_number": i + 1,
                "perspective": "black"
            }
            training_data.append(black_data)
        
        return training_data


class MultiStrategyGame:
    """Represents a game with explicit strategy information."""
    
    def __init__(self, white_strategy: ChessStrategy, 
                 black_strategy: ChessStrategy):
        self.white_strategy = white_strategy
        self.black_strategy = black_strategy
        self.board = chess.Board()
        self.moves = []
        self.boards = []
        self.result = None
        self.duration = 0
        self.evaluation_history = []
    
    def record_move(self, move: chess.Move, evaluation: float = None):
        """Record a move in the game."""
        self.boards.append(self.board.copy())
        self.moves.append(move)
        if evaluation is not None:
            self.evaluation_history.append(evaluation)
        self.board.push(move)
    
    def finalize(self, result: str, duration: float):
        """Finalize game."""
        self.result = result
        self.duration = duration
    
    def to_training_format(self, trainer: DualSidedTrainer) -> List[Dict]:
        """Convert game to training format."""
        return trainer.create_training_data_from_game(
            self.boards, self.moves, self.result
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "white_strategy": self.white_strategy.value,
            "black_strategy": self.black_strategy.value,
            "moves": [m.uci() for m in self.moves],
            "result": self.result,
            "duration": self.duration,
            "move_count": len(self.moves),
            "timestamp": datetime.now().isoformat(),
            "evaluations": self.evaluation_history
        }


class ComprehensiveStrategyTrainer:
    """
    Master trainer that:
    1. Plays all strategies against each other
    2. Trains from both sides of every game
    3. Accumulates experience for curriculum learning
    4. Regularly saves checkpoints
    """
    
    def __init__(self, model_path: str = None, device: str = "cuda"):
        """Initialize comprehensive trainer."""
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load or create model
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = ChessNetV2().to(self.device)
            self.model.load_state_dict(checkpoint)
        else:
            self.model = ChessNetV2().to(self.device)
        
        self.model.eval()
        
        # Initialize components
        self.player = HybridChessPlayer(model=self.model, device=self.device)
        self.dual_trainer = DualSidedTrainer(self.model, self.device)
        self.strategy_players = {
            strat: StrategyPlayer(self.player, strat)
            for strat in ChessStrategy
        }
        
        # Tracking
        self.all_games = []
        self.strategy_stats = defaultdict(lambda: {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "avg_moves": 0
        })
        self.training_data = []
        self.checkpoint_dir = Path("checkpoints_strategy")
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def play_strategy_game(self, white_strategy: ChessStrategy,
                          black_strategy: ChessStrategy,
                          max_moves: int = 300) -> MultiStrategyGame:
        """
        Play a game between two strategies.
        Records full game history for training.
        """
        game = MultiStrategyGame(white_strategy, black_strategy)
        white_player = self.strategy_players[white_strategy]
        black_player = self.strategy_players[black_strategy]
        
        start_time = time_module.time()
        move_count = 0
        
        while not game.board.is_game_over() and move_count < max_moves:
            if game.board.turn == chess.WHITE:
                move = white_player.select_move_with_strategy(game.board, temperature=0.2)
            else:
                move = black_player.select_move_with_strategy(game.board, temperature=0.2)
            
            if move is None:
                break
            
            game.record_move(move)
            move_count += 1
        
        duration = time_module.time() - start_time
        game.finalize(game.board.result(), duration)
        
        return game
    
    def train_cycle(self, num_games: int = 100, 
                   epochs: int = 10,
                   save_checkpoint: bool = True) -> Dict:
        """
        Run a complete training cycle:
        1. Play games with all strategy combinations
        2. Generate training data from both perspectives
        3. Train model
        4. Save checkpoint
        """
        print(f"\n{'='*80}")
        print(f"STRATEGY TRAINING CYCLE - {num_games} Games")
        print(f"{'='*80}\n")
        
        strategies = list(ChessStrategy)
        games_per_strategy_pair = max(1, num_games // (len(strategies) * len(strategies)))
        
        cycle_games = []
        
        # Play games
        print(f"[GAMES] Playing {num_games} games with all strategies...\n")
        
        games_played = 0
        for white_strat in strategies:
            for black_strat in strategies:
                if white_strat == black_strat:
                    continue  # Skip same strategy matchups
                
                for _ in range(games_per_strategy_pair):
                    if games_played >= num_games:
                        break
                    
                    print(f"[GAME {games_played+1:3d}] {white_strat.value:15s} vs "
                          f"{black_strat.value:15s}... ", end="", flush=True)
                    
                    game = self.play_strategy_game(
                        white_strat, black_strat, max_moves=300
                    )
                    
                    cycle_games.append(game)
                    self.all_games.append(game)
                    games_played += 1
                    
                    # Update stats
                    self._update_strategy_stats(game)
                    
                    print(f"{game.result} ({len(game.moves)} moves, {game.duration:.2f}s)")
                
                if games_played >= num_games:
                    break
            
            if games_played >= num_games:
                break
        
        # Generate training data from both sides
        print(f"\n[TRAINING] Generating dual-sided training data from {len(cycle_games)} games...")
        
        cycle_training_data = []
        for game in cycle_games:
            data = game.to_training_format(self.dual_trainer)
            cycle_training_data.extend(data)
        
        print(f"[TRAINING] Generated {len(cycle_training_data)} training examples")
        self.training_data.extend(cycle_training_data)
        
        # Train model
        print(f"\n[TRAINING] Training model for {epochs} epochs...\n")
        
        self.train_on_data(cycle_training_data, epochs=epochs)
        
        # Save checkpoint
        if save_checkpoint:
            checkpoint_path = self._save_checkpoint()
            print(f"\n[CHECKPOINT] Saved: {checkpoint_path}")
        
        # Print stats
        stats = self._get_cycle_stats(cycle_games)
        self._print_stats(stats)
        
        return stats
    
    def train_on_data(self, training_data: List[Dict], epochs: int = 10):
        """Train model on accumulated data."""
        if not training_data:
            print("[WARN] No training data")
            return
        
        # Convert to tensors
        positions = []
        moves = []
        results = []
        
        for example in training_data:
            try:
                board = chess.Board(example["board_state"])
                move = chess.Move.from_uci(example["move"])
                
                # Encode position
                pos_tensor = self.player.encode_board(board)
                positions.append(pos_tensor)
                
                # Encode move
                legal_moves = list(board.legal_moves)
                move_idx = legal_moves.index(move) if move in legal_moves else 0
                moves.append(move_idx)
                
                # Result value
                results.append(example["result"])
            except Exception as e:
                continue
        
        if not positions:
            print("[WARN] Could not convert training data")
            return
        
        positions_tensor = torch.stack(positions).to(self.device)
        moves_tensor = torch.tensor(moves, dtype=torch.long, device=self.device)
        results_tensor = torch.tensor(results, dtype=torch.float32, device=self.device)
        
        # Train with AdvancedTrainer
        # Create dataset mimicking GameExperienceDataset structure with simpler in-memory data
        class InMemoryDataset(torch.utils.data.Dataset):
            def __init__(self, positions, moves, results):
                self.positions = positions
                self.moves = moves
                self.results = results
            
            def __len__(self):
                return len(self.positions)
            
            def __getitem__(self, idx):
                return {
                    "position": self.positions[idx],
                    "move": self.moves[idx].unsqueeze(-1),
                    "reward": self.results[idx].unsqueeze(-1),
                    "outcome": self.results[idx].unsqueeze(-1)
                }
        
        dataset = InMemoryDataset(positions_tensor, moves_tensor, results_tensor)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train with AdvancedTrainer
        self.dual_trainer.trainer.train_epoch(train_loader, epochs=epochs)
    
    def _update_strategy_stats(self, game: MultiStrategyGame):
        """Update strategy statistics."""
        # White strategy stats
        w_stats = self.strategy_stats[game.white_strategy.value]
        w_stats["games_played"] += 1
        
        b_stats = self.strategy_stats[game.black_strategy.value]
        b_stats["games_played"] += 1
        
        # Results
        if "1-0" in game.result:
            w_stats["wins"] += 1
            b_stats["losses"] += 1
        elif "0-1" in game.result:
            w_stats["losses"] += 1
            b_stats["wins"] += 1
        else:
            w_stats["draws"] += 1
            b_stats["draws"] += 1
    
    def _get_cycle_stats(self, games: List[MultiStrategyGame]) -> Dict:
        """Get stats for training cycle."""
        stats = {
            "total_games": len(games),
            "total_moves": sum(len(g.moves) for g in games),
            "total_duration": sum(g.duration for g in games),
            "strategy_results": {}
        }
        
        for strat in ChessStrategy:
            strat_key = strat.value
            strat_games = [g for g in games 
                          if g.white_strategy == strat or g.black_strategy == strat]
            
            if strat_games:
                stats["strategy_results"][strat_key] = {
                    "games": len(strat_games),
                    "avg_moves": sum(len(g.moves) for g in strat_games) / len(strat_games)
                }
        
        return stats
    
    def _save_checkpoint(self) -> str:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"model_strategy_{timestamp}.pt"
        torch.save(self.model.state_dict(), str(checkpoint_path))
        return str(checkpoint_path)
    
    def _print_stats(self, stats: Dict):
        """Print training cycle stats."""
        print(f"\n{'='*80}")
        print("CYCLE STATISTICS")
        print(f"{'='*80}")
        print(f"Total Games: {stats['total_games']}")
        print(f"Total Moves: {stats['total_moves']}")
        print(f"Total Duration: {stats['total_duration']:.1f}s")
        print(f"Avg Game Length: {stats['total_moves']/stats['total_games']:.0f} moves")
        print(f"{'='*80}\n")
    
    def save_games_to_file(self, filename: str = "comprehensive_strategy_games.jsonl"):
        """Save all games."""
        with open(filename, 'a') as f:
            for game in self.all_games:
                f.write(json.dumps(game.to_dict()) + '\n')
        print(f"[SAVE] {len(self.all_games)} games saved to {filename}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive strategy training")
    parser.add_argument("--games", type=int, default=50,
                       help="Games per cycle (default: 50)")
    parser.add_argument("--cycles", type=int, default=5,
                       help="Training cycles (default: 5)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Epochs per cycle (default: 10)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    print("[INIT] Initializing comprehensive strategy trainer...")
    trainer = ComprehensiveStrategyTrainer()
    
    # Run training cycles
    for cycle in range(args.cycles):
        print(f"\n[CYCLE {cycle+1}/{args.cycles}]")
        trainer.train_cycle(
            num_games=args.games,
            epochs=args.epochs,
            save_checkpoint=True
        )
    
    # Save all games
    trainer.save_games_to_file()
    
    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
