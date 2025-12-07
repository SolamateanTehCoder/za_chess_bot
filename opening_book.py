"""
Opening book system for competitive chess play.
Supports PGN-based opening learning, ECO classification, and weighted move selection.
Provides fast lookups during game play with fallback to neural network.
"""

import chess
import chess.pgn
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from typing import Optional, Tuple, List, Dict
from datetime import datetime


class OpeningBook:
    """
    Opening book with ECO classification and move weighting.
    Learns from master games and tracks move frequencies/ratings.
    """
    
    def __init__(self, max_depth: int = 20):
        """
        Initialize opening book.
        
        Args:
            max_depth: Maximum number of half-moves to store (10 full moves)
        """
        self.max_depth = max_depth
        self.positions = {}  # FEN -> {"moves": {uci: weight, ...}, "eco": str, "rating": float}
        self.eco_names = {}  # ECO code -> name
        self.stats = {
            "total_positions": 0,
            "total_games_learned": 0,
            "last_updated": None
        }
    
    def learn_from_pgn(self, pgn_file: str, max_games: int = None, min_rating: int = 2000):
        """
        Learn opening repertoire from PGN file.
        
        Args:
            pgn_file: Path to PGN file
            max_games: Maximum number of games to process
            min_rating: Minimum player rating to learn from
        """
        pgn_path = Path(pgn_file)
        if not pgn_path.exists():
            print(f"[WARN] PGN file not found: {pgn_file}")
            return
        
        games_processed = 0
        
        with open(pgn_path) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                if max_games and games_processed >= max_games:
                    break
                
                # Check player ratings (get higher of two)
                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")
                try:
                    max_elo = max(int(white_elo), int(black_elo))
                    if max_elo < min_rating:
                        continue
                except:
                    pass
                
                # Get ECO code
                eco = game.headers.get("ECO", "")
                opening = game.headers.get("Opening", "")
                
                # Process moves up to max_depth or game end
                board = chess.Board()
                move_count = 0
                game_result = game.headers.get("Result", "*")
                
                for move in game.mainline_moves():
                    if move_count >= self.max_depth:
                        break
                    
                    fen = board.fen()
                    move_uci = move.uci()
                    
                    # Initialize position if not seen
                    if fen not in self.positions:
                        self.positions[fen] = {
                            "moves": defaultdict(float),
                            "eco": eco,
                            "opening": opening,
                            "frequency": 0,
                            "avg_rating": 0,
                            "games": []
                        }
                    
                    # Update move weight
                    self.positions[fen]["moves"][move_uci] += 1.0
                    self.positions[fen]["frequency"] += 1
                    
                    # Update average rating
                    if max_elo > 0:
                        pos_data = self.positions[fen]
                        prev_avg = pos_data["avg_rating"]
                        prev_count = max(1, pos_data["frequency"] - 1)
                        pos_data["avg_rating"] = (prev_avg * prev_count + max_elo) / pos_data["frequency"]
                    
                    board.push(move)
                    move_count += 1
                
                games_processed += 1
                if games_processed % 500 == 0:
                    print(f"[INFO] Processed {games_processed} games...")
        
        self.stats["total_positions"] = len(self.positions)
        self.stats["total_games_learned"] = games_processed
        self.stats["last_updated"] = datetime.now().isoformat()
        
        print(f"[INFO] Learned from {games_processed} games")
        print(f"[INFO] Opening book: {len(self.positions)} positions")
    
    def get_book_move(self, board: chess.Board, temperature: float = 0.3) -> Optional[str]:
        """
        Get opening book move with probability weighting.
        
        Args:
            board: Current board position
            temperature: Softmax temperature (lower = more deterministic)
            
        Returns:
            Best move UCI or None if not in book
        """
        fen = board.fen()
        
        if fen not in self.positions:
            return None
        
        pos_data = self.positions[fen]
        moves_dict = pos_data["moves"]
        
        if not moves_dict:
            return None
        
        # Convert moves to list for probability calculation
        moves = list(moves_dict.keys())
        weights = np.array([moves_dict[m] for m in moves])
        
        # Softmax with temperature
        exp_weights = np.exp(weights / (temperature * np.max(weights) + 1e-8))
        probabilities = exp_weights / np.sum(exp_weights)
        
        # Sample from distribution (but with high temperature preference for best)
        selected_idx = np.argmax(probabilities)
        
        return moves[selected_idx]
    
    def is_in_book(self, board: chess.Board) -> bool:
        """Check if position is in opening book."""
        return board.fen() in self.positions
    
    def get_eco(self, board: chess.Board) -> Tuple[Optional[str], Optional[str]]:
        """
        Get ECO classification for current position.
        
        Returns:
            Tuple of (eco_code, opening_name) or (None, None)
        """
        fen = board.fen()
        if fen in self.positions:
            data = self.positions[fen]
            return data.get("eco"), data.get("opening")
        return None, None
    
    def save_book(self, filepath: str):
        """Save opening book to JSON file."""
        # Convert defaultdict to regular dict for JSON serialization
        data = {
            "positions": {},
            "stats": self.stats
        }
        
        for fen, pos_data in self.positions.items():
            data["positions"][fen] = {
                "moves": dict(pos_data["moves"]),  # Convert defaultdict to dict
                "eco": pos_data["eco"],
                "opening": pos_data["opening"],
                "frequency": pos_data["frequency"],
                "avg_rating": pos_data["avg_rating"]
            }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Saved opening book to {filepath}")
    
    def load_book(self, filepath: str):
        """Load opening book from JSON file."""
        path = Path(filepath)
        if not path.exists():
            print(f"[WARN] Opening book not found: {filepath}")
            return False
        
        with open(filepath) as f:
            data = json.load(f)
        
        self.stats = data.get("stats", {})
        
        for fen, pos_data in data.get("positions", {}).items():
            self.positions[fen] = {
                "moves": defaultdict(float, pos_data["moves"]),
                "eco": pos_data["eco"],
                "opening": pos_data["opening"],
                "frequency": pos_data["frequency"],
                "avg_rating": pos_data["avg_rating"],
                "games": []
            }
        
        print(f"[INFO] Loaded opening book: {len(self.positions)} positions")
        return True
    
    def get_stats(self) -> Dict:
        """Get opening book statistics."""
        return {
            **self.stats,
            "positions_count": len(self.positions),
            "avg_frequency": np.mean([p["frequency"] for p in self.positions.values()]) if self.positions else 0
        }
    
    def get_book_depth(self, board: chess.Board) -> int:
        """
        Get book depth remaining (moves until out of opening book).
        
        Returns:
            Number of half-moves remaining in book
        """
        depth = 0
        temp_board = board.copy()
        
        while temp_board.fen() in self.positions and depth < self.max_depth:
            moves = self.positions[temp_board.fen()]["moves"]
            if not moves:
                break
            
            # Pick best move
            best_move = max(moves, key=moves.get)
            try:
                temp_board.push_uci(best_move)
                depth += 1
            except:
                break
        
        return depth


class OpeningLineGenerator:
    """Generate diverse opening lines from opening book."""
    
    def __init__(self, book: OpeningBook):
        """
        Initialize line generator.
        
        Args:
            book: OpeningBook instance
        """
        self.book = book
    
    def generate_line(self, max_moves: int = 10, temperature: float = 0.5) -> List[str]:
        """
        Generate random opening line from book.
        
        Args:
            max_moves: Maximum full moves to generate
            temperature: Softmax temperature for move selection
            
        Returns:
            List of moves in UCI format
        """
        board = chess.Board()
        line = []
        
        for _ in range(max_moves * 2):  # Half-moves
            if not self.book.is_in_book(board):
                break
            
            move = self.book.get_book_move(board, temperature)
            if not move:
                break
            
            try:
                board.push_uci(move)
                line.append(move)
            except:
                break
        
        return line
