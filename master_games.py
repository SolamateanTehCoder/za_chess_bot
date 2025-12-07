"""
Master games data loader and processor.
Learns from grandmaster games to improve opening and middlegame play.
Key component for reaching WCCC level.
"""

import chess
import chess.pgn
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json
from datetime import datetime


class MasterGamesDatabase:
    """Database of master games for training."""
    
    def __init__(self, min_rating: int = 2400, max_games: Optional[int] = None):
        """
        Initialize master games database.
        
        Args:
            min_rating: Minimum player rating to include
            max_games: Maximum games to load (None = all)
        """
        self.min_rating = min_rating
        self.max_games = max_games
        self.games: List[Dict] = []
        self.statistics = {
            "total_games": 0,
            "total_positions": 0,
            "rating_range": (0, 0),
            "time_span": ("", "")
        }
    
    def load_pgn_file(self, pgn_path: str, max_moves: int = 100) -> int:
        """
        Load games from PGN file.
        
        Args:
            pgn_path: Path to PGN file
            max_moves: Maximum moves to extract per game
            
        Returns:
            Number of games loaded
        """
        path = Path(pgn_path)
        if not path.exists():
            print(f"[ERROR] PGN file not found: {pgn_path}")
            return 0
        
        games_loaded = 0
        games_skipped = 0
        
        with open(path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                if self.max_games and games_loaded >= self.max_games:
                    break
                
                # Check ratings
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0"))
                    black_elo = int(game.headers.get("BlackElo", "0"))
                    min_elo = min(white_elo, black_elo)
                    
                    if min_elo < self.min_rating:
                        games_skipped += 1
                        continue
                except:
                    games_skipped += 1
                    continue
                
                # Extract game data
                game_data = self._extract_game_data(game, max_moves)
                if game_data:
                    self.games.append(game_data)
                    games_loaded += 1
                    
                    if games_loaded % 100 == 0:
                        print(f"[INFO] Loaded {games_loaded} games...")
        
        print(f"[INFO] Loaded {games_loaded} games (skipped {games_skipped} below rating threshold)")
        return games_loaded
    
    def _extract_game_data(self, game: chess.pgn.Game, max_moves: int) -> Optional[Dict]:
        """Extract structured data from PGN game."""
        try:
            positions = []
            moves = []
            board = chess.Board()
            
            move_count = 0
            for move in game.mainline_moves():
                if move_count >= max_moves:
                    break
                
                fen = board.fen()
                positions.append(fen)
                moves.append(move.uci())
                board.push(move)
                move_count += 1
            
            return {
                "white": game.headers.get("White", "Unknown"),
                "black": game.headers.get("Black", "Unknown"),
                "white_elo": int(game.headers.get("WhiteElo", "0")),
                "black_elo": int(game.headers.get("BlackElo", "0")),
                "result": game.headers.get("Result", "*"),
                "eco": game.headers.get("ECO", ""),
                "opening": game.headers.get("Opening", ""),
                "date": game.headers.get("Date", ""),
                "positions": positions,
                "moves": moves
            }
        except:
            return None
    
    def get_best_moves_by_position(self, fen: str, min_games: int = 3) -> Dict[str, int]:
        """
        Get move statistics for a position from master games.
        
        Args:
            fen: Position FEN
            min_games: Minimum games required to include a move
            
        Returns:
            Dictionary of move -> frequency
        """
        move_counts = Counter()
        
        for game in self.games:
            try:
                idx = game["positions"].index(fen)
                move = game["moves"][idx]
                move_counts[move] += 1
            except (ValueError, IndexError):
                continue
        
        # Filter by minimum games
        return {move: count for move, count in move_counts.items() if count >= min_games}
    
    def get_opening_statistics(self) -> Dict[str, Tuple[int, float]]:
        """
        Get statistics by opening.
        
        Returns:
            Dict of opening -> (game_count, white_score)
        """
        opening_stats = {}
        
        for game in self.games:
            opening = game["opening"] or "Unknown"
            
            # Calculate white score
            if game["result"] == "1-0":
                white_score = 1.0
            elif game["result"] == "0-1":
                white_score = 0.0
            else:  # Draw
                white_score = 0.5
            
            if opening not in opening_stats:
                opening_stats[opening] = {"count": 0, "white_score": 0.0}
            
            opening_stats[opening]["count"] += 1
            opening_stats[opening]["white_score"] += white_score
        
        # Calculate averages
        return {
            opening: (stats["count"], stats["white_score"] / stats["count"])
            for opening, stats in opening_stats.items()
        }
    
    def export_training_data(self, output_path: str, positions_per_game: int = 50):
        """
        Export games as training data (positions with moves).
        
        Args:
            output_path: Path to save training data
            positions_per_game: Positions to extract per game
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        training_data = []
        
        for game_idx, game in enumerate(self.games):
            for pos_idx, (fen, move) in enumerate(zip(game["positions"], game["moves"])):
                if pos_idx >= positions_per_game:
                    break
                
                # Calculate game result from this position's perspective
                board = chess.Board(fen)
                is_white = board.turn == chess.WHITE
                
                if game["result"] == "1-0":
                    game_result = 1.0 if is_white else -1.0
                elif game["result"] == "0-1":
                    game_result = -1.0 if is_white else 1.0
                else:
                    game_result = 0.0
                
                training_data.append({
                    "fen": fen,
                    "move": move,
                    "result": game_result,
                    "eco": game["eco"],
                    "player_elo": game["white_elo"] if is_white else game["black_elo"]
                })
        
        # Save as JSONL
        with open(output_path, 'w') as f:
            for entry in training_data:
                f.write(json.dumps(entry) + "\n")
        
        print(f"[INFO] Exported {len(training_data)} training positions to {output_path}")
        
        return training_data
    
    def get_opening_repetoire(self, player_name: str) -> Dict[str, List[str]]:
        """
        Get opening repertoire for a specific player.
        
        Args:
            player_name: Player name
            
        Returns:
            Dict of opening -> list of move sequences
        """
        repertoire = {}
        
        for game in self.games:
            if game["white"] == player_name or game["black"] == player_name:
                opening = game["opening"] or "Unknown"
                moves_seq = " ".join(game["moves"][:20])  # First 20 half-moves
                
                if opening not in repertoire:
                    repertoire[opening] = []
                
                repertoire[opening].append(moves_seq)
        
        # Keep only most frequent variations
        return {
            opening: list(set(variations))[:5]  # Top 5 variations per opening
            for opening, variations in repertoire.items()
        }


class TrainingDataGenerator:
    """Generate training data from master games and self-play."""
    
    @staticmethod
    def create_mixed_dataset(master_games: MasterGamesDatabase, 
                            self_play_games: List[str],
                            output_path: str,
                            master_weight: float = 0.3):
        """
        Create training dataset combining master games and self-play.
        
        Args:
            master_games: Master games database
            self_play_games: Paths to self-play game files
            output_path: Output path for training data
            master_weight: Weight for master games (0.0-1.0)
        """
        training_data = []
        
        # Add master games
        master_data = master_games.export_training_data(output_path + ".master.jsonl")
        master_sample_size = int(len(master_data) * master_weight)
        
        # Randomly sample master games (with preference for high-rated games)
        master_sample = np.random.choice(
            len(master_data),
            size=min(master_sample_size, len(master_data)),
            replace=False
        )
        
        for idx in master_sample:
            training_data.append(master_data[idx])
        
        print(f"[INFO] Added {len(master_sample)} master game positions")
        
        # Add self-play games
        self_play_sample_size = int(len(training_data) / (1 - master_weight) * (1 - master_weight)) if master_weight < 1 else len(training_data)
        
        for game_path in self_play_games:
            # Load self-play games (implement based on your format)
            # This is a placeholder
            pass
        
        print(f"[INFO] Created mixed dataset with {len(training_data)} positions")
        
        # Save mixed dataset
        with open(output_path, 'w') as f:
            for entry in training_data:
                f.write(json.dumps(entry) + "\n")
        
        return training_data
