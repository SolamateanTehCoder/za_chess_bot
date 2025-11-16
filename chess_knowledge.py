"""Opening book for chess engine."""

import chess
import random

class OpeningBook:
    """
    Opening book with common chess openings.
    Provides moves for the opening phase to help the AI start games strongly.
    """
    
    def __init__(self):
        # Dictionary of opening lines: position_fen -> list of good moves
        self.book = self._build_opening_book()
        self.use_book = True
        
    def _build_opening_book(self):
        """Build a comprehensive opening book."""
        book = {}
        
        # Starting position - most popular first moves
        book[chess.STARTING_FEN] = [
            chess.Move.from_uci("e2e4"),  # King's Pawn
            chess.Move.from_uci("d2d4"),  # Queen's Pawn
            chess.Move.from_uci("c2c4"),  # English Opening
            chess.Move.from_uci("g1f3"),  # Reti Opening
        ]
        
        # Response to 1.e4 as Black
        book[self._get_fen_after("e2e4")] = [
            chess.Move.from_uci("e7e5"),  # Open Game
            chess.Move.from_uci("c7c5"),  # Sicilian Defense
            chess.Move.from_uci("e7e6"),  # French Defense
            chess.Move.from_uci("c7c6"),  # Caro-Kann Defense
        ]
        
        # Response to 1.d4 as Black
        book[self._get_fen_after("d2d4")] = [
            chess.Move.from_uci("d7d5"),  # Closed Game
            chess.Move.from_uci("g8f6"),  # Indian Defenses
            chess.Move.from_uci("e7e6"),  # French via d4
            chess.Move.from_uci("f7f5"),  # Dutch Defense
        ]
        
        # After 1.e4 e5
        book[self._get_fen_after("e2e4", "e7e5")] = [
            chess.Move.from_uci("g1f3"),  # King's Knight
            chess.Move.from_uci("f2f4"),  # King's Gambit
            chess.Move.from_uci("b1c3"),  # Vienna Game
        ]
        
        # After 1.e4 e5 2.Nf3
        book[self._get_fen_after("e2e4", "e7e5", "g1f3")] = [
            chess.Move.from_uci("b8c6"),  # Most common
            chess.Move.from_uci("g8f6"),  # Petrov's Defense
        ]
        
        # After 1.e4 e5 2.Nf3 Nc6
        book[self._get_fen_after("e2e4", "e7e5", "g1f3", "b8c6")] = [
            chess.Move.from_uci("f1b5"),  # Ruy Lopez
            chess.Move.from_uci("f1c4"),  # Italian Game
            chess.Move.from_uci("d2d4"),  # Scotch Game
            chess.Move.from_uci("b1c3"),  # Four Knights
        ]
        
        # After 1.d4 d5
        book[self._get_fen_after("d2d4", "d7d5")] = [
            chess.Move.from_uci("c2c4"),  # Queen's Gambit
            chess.Move.from_uci("g1f3"),  # London System prep
            chess.Move.from_uci("c1f4"),  # London System
        ]
        
        # After 1.d4 Nf6
        book[self._get_fen_after("d2d4", "g8f6")] = [
            chess.Move.from_uci("c2c4"),  # Indian Game
            chess.Move.from_uci("g1f3"),  # Standard development
            chess.Move.from_uci("c1f4"),  # London System
        ]
        
        # Sicilian Defense: 1.e4 c5
        book[self._get_fen_after("e2e4", "c7c5")] = [
            chess.Move.from_uci("g1f3"),  # Open Sicilian
            chess.Move.from_uci("b1c3"),  # Closed Sicilian
            chess.Move.from_uci("c2c3"),  # Alapin Variation
        ]
        
        return book
    
    def _get_fen_after(self, *moves):
        """Get FEN position after a sequence of moves."""
        board = chess.Board()
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
        return board.fen()
    
    def get_book_move(self, board):
        """
        Get a move from the opening book for the current position.
        
        Args:
            board: chess.Board object
            
        Returns:
            chess.Move or None if position not in book
        """
        if not self.use_book:
            return None
        
        fen = board.fen()
        
        # Check if position is in book
        if fen in self.book:
            # Get candidate moves
            candidate_moves = self.book[fen]
            
            # Filter for legal moves only
            legal_candidates = [m for m in candidate_moves if m in board.legal_moves]
            
            if legal_candidates:
                # Return a random book move (with slight preference for first moves)
                weights = [2.0 / (i + 1) for i in range(len(legal_candidates))]
                return random.choices(legal_candidates, weights=weights, k=1)[0]
        
        return None
    
    def is_in_book(self, board):
        """Check if current position is in the opening book."""
        return board.fen() in self.book
    
    def set_enabled(self, enabled):
        """Enable or disable the opening book."""
        self.use_book = enabled


class EndgameTablebase:
    """
    Simplified endgame knowledge.
    Provides guidance for common endgame patterns.
    """
    
    def __init__(self):
        self.enabled = True
    
    def evaluate_endgame(self, board):
        """
        Evaluate endgame positions and suggest moves.
        
        Args:
            board: chess.Board object
            
        Returns:
            tuple: (is_endgame, suggested_move, evaluation)
        """
        if not self.enabled:
            return False, None, 0.0
        
        # Count pieces
        piece_count = len(board.piece_map())
        
        # Not an endgame if too many pieces
        if piece_count > 10:
            return False, None, 0.0
        
        # Basic endgame principles
        is_endgame = True
        suggested_move = None
        evaluation = 0.0
        
        # King activity in endgame is crucial
        # Prefer king moves toward center in endgame
        for move in board.legal_moves:
            if board.piece_at(move.from_square) == chess.KING:
                # Evaluate king centralization
                to_square = move.to_square
                center_distance = self._distance_to_center(to_square)
                if center_distance < 2:
                    suggested_move = move
                    evaluation = 0.3
                    break
        
        return is_endgame, suggested_move, evaluation
    
    def _distance_to_center(self, square):
        """Calculate Manhattan distance from square to center."""
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        center_rank = 3.5
        center_file = 3.5
        return abs(rank - center_rank) + abs(file - center_file)


class ChessKnowledge:
    """
    Chess knowledge system combining opening book and endgame knowledge.
    """
    
    def __init__(self, use_opening_book=True, use_endgame_knowledge=True):
        self.opening_book = OpeningBook()
        self.endgame_tablebase = EndgameTablebase()
        
        self.opening_book.set_enabled(use_opening_book)
        self.endgame_tablebase.enabled = use_endgame_knowledge
        
        self.book_moves_used = 0
        self.endgame_assists = 0
    
    def get_assisted_move(self, board):
        """
        Get move suggestion using chess knowledge.
        
        Args:
            board: chess.Board object
            
        Returns:
            tuple: (move, source) where source is 'book', 'endgame', or None
        """
        # Try opening book first
        book_move = self.opening_book.get_book_move(board)
        if book_move:
            self.book_moves_used += 1
            return book_move, 'book'
        
        # Try endgame knowledge
        is_endgame, endgame_move, eval_score = self.endgame_tablebase.evaluate_endgame(board)
        if is_endgame and endgame_move and eval_score > 0.2:
            self.endgame_assists += 1
            return endgame_move, 'endgame'
        
        return None, None
    
    def get_stats(self):
        """Get statistics about knowledge usage."""
        return {
            'book_moves_used': self.book_moves_used,
            'endgame_assists': self.endgame_assists
        }
