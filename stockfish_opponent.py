"""Stockfish engine integration."""

import chess
import chess.engine
from config import STOCKFISH_PATH, STOCKFISH_SKILL_LEVEL, STOCKFISH_DEPTH


class StockfishOpponent:
    """
    Wrapper for Stockfish chess engine to serve as an opponent.
    """
    
    def __init__(self, skill_level=STOCKFISH_SKILL_LEVEL, depth=STOCKFISH_DEPTH):
        """
        Initialize Stockfish engine.
        
        Args:
            skill_level: Skill level for Stockfish (0-20)
            depth: Search depth for Stockfish
        """
        self.skill_level = skill_level
        self.depth = depth
        self.engine = None
        
    def start(self):
        """Start the Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
            self.engine.configure({"Skill Level": self.skill_level})
        except Exception as e:
            print(f"Error starting Stockfish: {e}")
            print(f"Make sure Stockfish is installed and the path is correct: {STOCKFISH_PATH}")
            print("You can download Stockfish from: https://stockfishchess.org/download/")
            raise
    
    def get_move(self, board):
        """
        Get a move from Stockfish for the given board position.
        
        Args:
            board: chess.Board object
            
        Returns:
            chess.Move object
        """
        if self.engine is None:
            self.start()
        
        result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        return result.move
    
    def close(self):
        """Close the Stockfish engine."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None
    
    def __del__(self):
        """Destructor to ensure engine is closed."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
