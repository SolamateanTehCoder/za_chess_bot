"""
WCCC Training & Competition Main Loop
Complete pipeline for training and tournament play.
Now with comprehensive strategy support!
"""

import torch
import argparse
from pathlib import Path
from datetime import datetime
import json

from hybrid_player import HybridChessPlayer
from advanced_trainer import TrainingPipeline
from tournament import Tournament, GameResult
from chess_models import ChessNetV2
from comprehensive_trainer import ComprehensiveStrategyTrainer
from chess_strategies import ChessStrategy
import chess


class WCCCMainLoop:
    """Main training and competition loop for WCCC bot."""
    
    def __init__(self, model_path: str = None, use_enhanced_model: bool = True):
        """
        Initialize main loop.
        
        Args:
            model_path: Path to model checkpoint
            use_enhanced_model: Use ChessNetV2 (True) or SimpleChessNet (False)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        # Initialize player
        self.player = HybridChessPlayer(
            model_path=model_path,
            use_enhanced_model=use_enhanced_model,
            device=self.device
        )
        
        # Initialize trainer
        self.trainer = TrainingPipeline(self.player.model, self.device)
        
        # Results tracking
        self.results = {
            "games_played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "training_sessions": []
        }
    
    def generate_self_play_games(self, num_games: int = 100, 
                                output_file: str = "self_play_games.jsonl") -> str:
        """
        Generate self-play games for training.
        
        Args:
            num_games: Number of games to generate
            output_file: Output file for games
            
        Returns:
            Path to games file
        """
        print(f"\n=== Generating {num_games} Self-Play Games ===\n")
        
        opponent_player = HybridChessPlayer(
            model=self.player.model,
            use_enhanced_model=self.player.use_enhanced_model,
            device=self.device
        )
        
        games_data = []
        
        for game_num in range(num_games):
            board = chess.Board()
            moves = []
            game_start = datetime.now()
            
            move_count = 0
            while not board.is_game_over() and move_count < 200:
                # White move
                white_move = self.player.select_move(board, remaining_time_ms=5000)
                if not white_move:
                    break
                board.push_uci(white_move)
                moves.append(white_move)
                move_count += 1
                
                if board.is_game_over():
                    break
                
                # Black move (opponent)
                black_move = opponent_player.select_move(board, remaining_time_ms=5000)
                if not black_move:
                    break
                board.push_uci(black_move)
                moves.append(black_move)
                move_count += 1
            
            # Determine result
            if board.is_checkmate():
                if board.turn:
                    result = "0-1"
                else:
                    result = "1-0"
            elif board.is_stalemate():
                result = "1/2-1/2"
            else:
                result = "1/2-1/2"
            
            games_data.append({
                "game_num": game_num + 1,
                "moves": moves,
                "result": result,
                "move_count": len(moves),
                "duration": (datetime.now() - game_start).total_seconds()
            })
            
            if (game_num + 1) % 10 == 0:
                print(f"[INFO] Generated {game_num + 1}/{num_games} games")
                if result == "1-0":
                    print(f"       Result: White Win")
                elif result == "0-1":
                    print(f"       Result: Black Win")
                else:
                    print(f"       Result: Draw")
        
        # Save games
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for game in games_data:
                f.write(json.dumps(game) + "\n")
        
        print(f"\n[INFO] Saved {num_games} games to {output_file}")
        
        return str(output_path)
    
    def train_from_games(self, games_file: str = "self_play_games.jsonl",
                        num_epochs: int = 10, batch_size: int = 32):
        """
        Train model from generated games.
        
        Args:
            games_file: File with game data
            num_epochs: Training epochs
            batch_size: Batch size
        """
        print(f"\n=== Training on {games_file} ===\n")
        
        results = self.trainer.train(
            games_file=games_file,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        self.results["training_sessions"].append({
            "timestamp": datetime.now().isoformat(),
            "games_file": games_file,
            "epochs": num_epochs,
            "results": results
        })
        
        return results
    
    def play_tournament(self, opponent_name: str = "StockfishReference",
                       num_games: int = 10, tournament_name: str = "WCCC_Trial") -> Tournament:
        """
        Play tournament games against reference engine.
        
        Args:
            opponent_name: Name of opponent
            num_games: Number of games to play
            tournament_name: Tournament name
            
        Returns:
            Completed tournament
        """
        print(f"\n=== Playing Tournament: {tournament_name} ===\n")
        print(f"Opponent: {opponent_name}")
        print(f"Games: {num_games}\n")
        
        tournament = Tournament(tournament_name)
        tournament.add_player("Za Chess Bot")
        tournament.add_player(opponent_name)
        
        for game_num in range(num_games):
            # Alternate colors
            if game_num % 2 == 0:
                white_player = "Za Chess Bot"
                black_player = opponent_name
            else:
                white_player = opponent_name
                black_player = "Za Chess Bot"
            
            # Play game
            board = chess.Board()
            moves = []
            game_start = datetime.now()
            
            while not board.is_game_over():
                if board.turn == chess.WHITE and white_player == "Za Chess Bot":
                    move = self.player.select_move(board, remaining_time_ms=30000)
                elif board.turn == chess.BLACK and black_player == "Za Chess Bot":
                    move = self.player.select_move(board, remaining_time_ms=30000)
                else:
                    # Opponent move (would be generated by actual opponent engine)
                    break
                
                if not move:
                    break
                
                board.push_uci(move)
                moves.append(chess.Move.from_uci(move))
            
            # Determine result
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    result = GameResult.BLACK_WIN
                else:
                    result = GameResult.WHITE_WIN
            else:
                result = GameResult.DRAW
            
            tournament.record_game(white_player, black_player, result,
                                  round_num=game_num + 1, moves=moves)
            
            print(f"[Game {game_num+1}/{num_games}] {white_player} vs {black_player}: {result.value}")
        
        tournament.print_standings()
        tournament.export_pgn(f"tournaments/{tournament_name}.pgn")
        tournament.export_json(f"tournaments/{tournament_name}.json")
        
        return tournament
    
    def run_training_cycle(self, num_games: int = 100, num_epochs: int = 10,
                          num_tournament_games: int = 20):
        """
        Run complete training cycle: generate games -> train -> test.
        
        Args:
            num_games: Self-play games to generate
            num_epochs: Training epochs
            num_tournament_games: Tournament test games
        """
        print("\n" + "="*60)
        print("WCCC BOT - COMPLETE TRAINING CYCLE")
        print("="*60)
        
        cycle_start = datetime.now()
        
        # Step 1: Generate self-play games
        games_file = self.generate_self_play_games(num_games)
        
        # Step 2: Train model
        train_results = self.train_from_games(games_file, num_epochs)
        
        # Step 3: Test with tournament
        tournament = self.play_tournament(
            num_games=num_tournament_games,
            tournament_name=f"WCCC_Trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        cycle_duration = (datetime.now() - cycle_start).total_seconds()
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING CYCLE SUMMARY")
        print("="*60)
        print(f"Total Duration: {cycle_duration:.1f}s")
        print(f"Self-play Games: {num_games}")
        print(f"Training Epochs: {num_epochs}")
        print(f"Tournament Games: {num_tournament_games}")
        
        standings = tournament.get_standings()
        for rank, (player, wins, draws, losses, score, rating) in enumerate(standings, 1):
            print(f"{rank}. {player}: {score:.1f}/{wins + draws + losses} ({rating:.0f} Elo)")
        
        print("="*60 + "\n")
    
    def interactive_play(self):
        """Play interactive games (for testing/demo)."""
        print("\n=== Interactive Mode ===\n")
        print("Type moves in UCI format (e.g., e2e4)")
        print("Type 'quit' to exit\n")
        
        board = chess.Board()
        
        while not board.is_game_over():
            print(board)
            print()
            
            if board.turn == chess.WHITE:
                # Human plays white
                while True:
                    move_str = input("Your move: ").strip().lower()
                    
                    if move_str == "quit":
                        return
                    
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            board.push(move)
                            break
                        else:
                            print("Illegal move!")
                    except:
                        print("Invalid format!")
            else:
                # AI plays black
                move = self.player.select_move(board, remaining_time_ms=10000)
                if move:
                    print(f"AI plays: {move}")
                    board.push_uci(move)
                else:
                    break
        
        print("\nGame over!")
        print(board)
    
    def run_strategy_training(self, num_games: int = 50, 
                            num_cycles: int = 3,
                            epochs_per_cycle: int = 10):
        """
        Run comprehensive strategy training.
        Plays all strategies against each other and learns from both sides.
        
        Args:
            num_games: Games per cycle
            num_cycles: Number of training cycles
            epochs_per_cycle: Training epochs per cycle
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE STRATEGY TRAINING")
        print("="*80)
        print(f"Games per Cycle: {num_games}")
        print(f"Training Cycles: {num_cycles}")
        print(f"Epochs per Cycle: {epochs_per_cycle}\n")
        
        # Initialize strategy trainer
        strategy_trainer = ComprehensiveStrategyTrainer(device=self.device)
        
        # Run training cycles
        all_stats = []
        for cycle in range(num_cycles):
            print(f"\n[CYCLE {cycle+1}/{num_cycles}]")
            
            stats = strategy_trainer.train_cycle(
                num_games=num_games,
                epochs=epochs_per_cycle,
                save_checkpoint=True
            )
            
            all_stats.append(stats)
        
        # Save all games
        strategy_trainer.save_games_to_file()
        
        # Print overall summary
        print("\n" + "="*80)
        print("STRATEGY TRAINING SUMMARY")
        print("="*80)
        print(f"Total Cycles: {num_cycles}")
        print(f"Total Training Games: {num_cycles * num_games}")
        print(f"Training Examples Generated: {len(strategy_trainer.training_data)}")
        
        print("\nStrategy Statistics:")
        for strat in sorted(strategy_trainer.strategy_stats.keys()):
            stats = strategy_trainer.strategy_stats[strat]
            if stats["games_played"] > 0:
                win_rate = stats["wins"] / stats["games_played"]
                print(f"  {strat:15s}: {win_rate:.1%} win rate ({stats['wins']}/{stats['games_played']})")
        
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WCCC Chess Bot Training & Competition")
    
    parser.add_argument("--mode", default="train",
                       choices=["train", "play", "interactive", "tournament", "strategy"],
                       help="Mode to run")
    parser.add_argument("--games", type=int, default=100,
                       help="Number of self-play games")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--tournament-games", type=int, default=20,
                       help="Tournament test games")
    parser.add_argument("--strategy-cycles", type=int, default=3,
                       help="Strategy training cycles")
    parser.add_argument("--strategy-games", type=int, default=50,
                       help="Strategy games per cycle")
    
    args = parser.parse_args()
    
    # Initialize
    wccc = WCCCMainLoop(model_path=args.model, use_enhanced_model=True)
    
    # Run requested mode
    if args.mode == "train":
        wccc.run_training_cycle(
            num_games=args.games,
            num_epochs=args.epochs,
            num_tournament_games=args.tournament_games
        )
    elif args.mode == "play":
        wccc.generate_self_play_games(args.games)
    elif args.mode == "interactive":
        wccc.interactive_play()
    elif args.mode == "tournament":
        wccc.play_tournament(num_games=args.tournament_games)
    elif args.mode == "strategy":
        wccc.run_strategy_training(
            num_games=args.strategy_games,
            num_cycles=args.strategy_cycles,
            epochs_per_cycle=args.epochs
        )
    
    # Cleanup
    wccc.player.close()


if __name__ == "__main__":
    main()
