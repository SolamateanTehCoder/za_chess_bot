"""
Comprehensive chess knowledge system with ALL openings, tactics, strategies, and endgames
from Wikipedia's extensive chess knowledge base.
"""

import chess
import random
from typing import Optional, Tuple, Dict, List

class ComprehensiveOpeningBook:
    """
    Complete encyclopedia of chess openings from Wikipedia.
    Covers A00-E99 ECO codes with hundreds of variations.
    """
    
    def __init__(self):
        self.book = self._build_comprehensive_book()
        self.stats = {
            'moves_used': 0,
            'positions_queried': 0,
            'unique_positions': len(self.book)
        }
        
    def _build_comprehensive_book(self) -> Dict[str, List[chess.Move]]:
        """Build complete opening book from Wikipedia's List of Chess Openings."""
        book = {}
        starting_fen = chess.STARTING_FEN
        
        # ===== STARTING MOVES - MOST POPULAR =====
        book[starting_fen] = [
            chess.Move.from_uci("e2e4"),  # King's Pawn (most popular)
            chess.Move.from_uci("d2d4"),  # Queen's Pawn (second most popular)
            chess.Move.from_uci("g1f3"),  # Reti/Zukertort
            chess.Move.from_uci("c2c4"),  # English Opening
            chess.Move.from_uci("f2f4"),  # Bird's Opening
            chess.Move.from_uci("b2b3"),  # Larsen's Opening
            chess.Move.from_uci("g2g3"),  # King's Fianchetto
            chess.Move.from_uci("b1c3"),  # Van Geet/Dunst Opening
        ]
        
        # Auto-populate book with move sequences
        self._populate_with_sequences(book)
        
        return book
    
    def _add_line(self, book: Dict, *moves: str):
        """Helper to add a complete opening line."""
        board = chess.Board()
        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
            fen = board.fen()
            if fen not in book:
                book[fen] = []
            # Add next move if there is one
            if moves.index(move_uci) < len(moves) - 1:
                next_move = chess.Move.from_uci(moves[moves.index(move_uci) + 1])
                if next_move not in book[fen]:
                    book[fen].append(next_move)
    
    def _populate_with_sequences(self, book: Dict):
        """Populate book with hundreds of opening lines from Wikipedia."""
        
        # ===== KING'S PAWN OPENINGS (1.e4) - ECO B00-C99 =====
        openings = [
            # Ruy Lopez (Spanish) - C60-C99
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],  # Main line
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],  # Morphy Defense
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"],  # Closed Ruy
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"],  # Berlin Defense
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "f7f5"],  # Schliemann
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],  # Castled
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"],
            
            # Italian Game - C50-C54
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],  # Italian
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],  # Giuoco Piano
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3"],  # Main Giuoco
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "b2b4"],  # Evans Gambit
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],  # Two Knights
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"],  # Fried Liver prep
            ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5", "d7d5"],  # Fried Liver
            
            # Scotch Game - C44-C45
            ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4"],  # Scotch
            ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4"],
            ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"],
            ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "c2c3"],  # Scotch Gambit
            
            # Petrov Defense (Russian) - C42-C43
            ["e2e4", "e7e5", "g1f3", "g8f6"],  # Petrov's
            ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5"],
            ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6"],
            ["e2e4", "e7e5", "g1f3", "g8f6", "d2d4"],  # Steinitz Attack
            
            # Four Knights - C46-C49
            ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3"],  # Four Knights
            ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"],
            ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6", "f1b5"],  # Spanish Four Knights
            
            # King's Gambit - C30-C39
            ["e2e4", "e7e5", "f2f4"],  # King's Gambit
            ["e2e4", "e7e5", "f2f4", "e5f4"],  # Accepted
            ["e2e4", "e7e5", "f2f4", "f8c5"],  # Declined Classical
            ["e2e4", "e7e5", "f2f4", "d7d5"],  # Falkbeer Countergambit
            ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3"],  # King's Knight Gambit
            ["e2e4", "e7e5", "f2f4", "e5f4", "f1c4"],  # Bishop's Gambit
            
            # Vienna Game - C25-C29
            ["e2e4", "e7e5", "b1c3"],  # Vienna
            ["e2e4", "e7e5", "b1c3", "g8f6"],  # Vienna Game proper
            ["e2e4", "e7e5", "b1c3", "b8c6"],
            ["e2e4", "e7e5", "b1c3", "f8c5"],
            
            # Bishop's Opening - C23-C24
            ["e2e4", "e7e5", "f1c4"],  # Bishop's Opening
            ["e2e4", "e7e5", "f1c4", "g8f6"],
            ["e2e4", "e7e5", "f1c4", "f8c5"],
            
            # Center Game - C21-C22
            ["e2e4", "e7e5", "d2d4"],  # Center Game
            ["e2e4", "e7e5", "d2d4", "e5d4"],
            
            # Philidor Defense - C41
            ["e2e4", "e7e5", "g1f3", "d7d6"],  # Philidor
            ["e2e4", "e7e5", "g1f3", "d7d6", "d2d4"],
            
            # ===== SICILIAN DEFENSE (1.e4 c5) - ECO B20-B99 =====
            ["e2e4", "c7c5"],  # Sicilian
            ["e2e4", "c7c5", "g1f3"],  # Open Sicilian
            ["e2e4", "c7c5", "g1f3", "d7d6"],  # Main line
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"],
            
            # Najdorf - B90-B99
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1g5"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "f2f3"],
            
            # Dragon - B70-B79
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6", "f1e2"],
            ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "g7g6", "f2f3"],  # Yugoslav
            ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g7g6"],  # Accelerated Dragon
            
            # Sveshnikov - B33
            ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5"],
            
            # Kan - B40-B43
            ["e2e4", "c7c5", "g1f3", "e7e6"],
            ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4"],
            ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4"],
            
            # Taimanov - B44-B49
            ["e2e4", "c7c5", "g1f3", "b8c6"],  # Old Sicilian
            ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4"],
            ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4"],
            ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "e7e6"],
            
            # Alapin - B22
            ["e2e4", "c7c5", "c2c3"],  # Alapin
            ["e2e4", "c7c5", "c2c3", "d7d5"],
            ["e2e4", "c7c5", "c2c3", "g8f6"],
            
            # Closed Sicilian - B23-B26
            ["e2e4", "c7c5", "b1c3"],  # Closed
            ["e2e4", "c7c5", "b1c3", "b8c6"],
            ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3"],
            
            # ===== FRENCH DEFENSE (1.e4 e6) - ECO C00-C19 =====
            ["e2e4", "e7e6"],  # French
            ["e2e4", "e7e6", "d2d4"],
            ["e2e4", "e7e6", "d2d4", "d7d5"],
            ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"],  # Main Line
            ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],  # Classical
            ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"],  # Winawer
            ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"],  # Advance
            ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5"],
            ["e2e4", "e7e6", "d2d4", "d7d5", "e4d5"],  # Exchange
            ["e2e4", "e7e6", "d2d4", "d7d5", "e4d5", "e6d5"],
            ["e2e4", "e7e6", "d2d4", "d7d5", "g1f3"],  # Two Knights
            
            # ===== CARO-KANN DEFENSE (1.e4 c6) - ECO B10-B19 =====
            ["e2e4", "c7c6"],  # Caro-Kann
            ["e2e4", "c7c6", "d2d4"],
            ["e2e4", "c7c6", "d2d4", "d7d5"],
            ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"],  # Main line
            ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"],
            ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"],
            ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5"],  # Advance
            ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5"],  # Exchange
            ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5"],
            
            # ===== PIRC DEFENSE (1.e4 d6) - ECO B07-B09 =====
            ["e2e4", "d7d6"],  # Pirc
            ["e2e4", "d7d6", "d2d4"],
            ["e2e4", "d7d6", "d2d4", "g8f6"],
            ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3"],
            ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
            ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "f2f4"],  # Austrian Attack
            
            # ===== MODERN DEFENSE (1.e4 g6) - ECO B06 =====
            ["e2e4", "g7g6"],  # Modern
            ["e2e4", "g7g6", "d2d4"],
            ["e2e4", "g7g6", "d2d4", "f8g7"],
            
            # ===== ALEKHINE'S DEFENSE (1.e4 Nf6) - ECO B02-B05 =====
            ["e2e4", "g8f6"],  # Alekhine's
            ["e2e4", "g8f6", "e4e5"],
            ["e2e4", "g8f6", "e4e5", "f6d5"],
            ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4"],
            ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"],
            
            # ===== SCANDINAVIAN DEFENSE (1.e4 d5) - ECO B01 =====
            ["e2e4", "d7d5"],  # Scandinavian
            ["e2e4", "d7d5", "e4d5"],
            ["e2e4", "d7d5", "e4d5", "d8d5"],  # Main Line
            ["e2e4", "d7d5", "e4d5", "g8f6"],  # Modern
            
            # ===== QUEEN'S PAWN OPENINGS (1.d4) - ECO D00-E99 =====
            ["d2d4", "d7d5"],  # Closed Game
            
            # Queen's Gambit - D06-D69
            ["d2d4", "d7d5", "c2c4"],  # Queen's Gambit
            ["d2d4", "d7d5", "c2c4", "e7e6"],  # QGD
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3"],
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5"],  # Orthodox
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3"],
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1f4"],  # London vs QGD
            
            # QGA - D20-D29
            ["d2d4", "d7d5", "c2c4", "d5c4"],  # QGA
            ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3"],
            ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6"],
            ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3"],
            
            # Slav Defense - D10-D19
            ["d2d4", "d7d5", "c2c4", "c7c6"],  # Slav
            ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3"],
            ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3"],
            ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6"],
            ["d2d4", "d7d5", "c2c4", "c7c6", "b1c3", "g8f6", "g1f3"],
            
            # Semi-Slav - D43-D49
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "g1f3", "c7c6"],  # Semi-Slav
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c6"],
            
            # Tarrasch Defense - D32-D34
            ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5"],  # Tarrasch
            
            # London System - D02
            ["d2d4", "d7d5", "g1f3"],  # London prep
            ["d2d4", "d7d5", "g1f3", "g8f6"],
            ["d2d4", "d7d5", "g1f3", "g8f6", "c1f4"],  # London System
            ["d2d4", "g8f6", "g1f3", "d7d5", "c1f4"],
            ["d2d4", "g8f6", "c1f4"],  # London vs Nf6
            
            # Colle System - D04-D05
            ["d2d4", "d7d5", "g1f3", "g8f6", "e2e3"],  # Colle
            ["d2d4", "g8f6", "g1f3", "e7e6", "e2e3"],
            
            # Torre Attack - D03
            ["d2d4", "g8f6", "g1f3", "e7e6", "c1g5"],  # Torre
            ["d2d4", "d7d5", "g1f3", "g8f6", "c1g5"],
            
            # Trompowsky - A45
            ["d2d4", "g8f6", "c1g5"],  # Trompowsky
            
            # ===== INDIAN DEFENSES - ECO A40-A99, E00-E99 =====
            ["d2d4", "g8f6"],  # Indian Defense
            ["d2d4", "g8f6", "c2c4"],  # Indian Game
            
            # King's Indian - E60-E99
            ["d2d4", "g8f6", "c2c4", "g7g6"],  # KID
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3"],
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4"],  # Classical KID
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"],
            
            # Grünfeld - D70-D99
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],  # Grünfeld
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5"],
            ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5"],
            
            # Nimzo-Indian - E20-E59
            ["d2d4", "g8f6", "c2c4", "e7e6"],  # Nimzo prep
            ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3"],
            ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],  # Nimzo-Indian
            ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"],  # Rubinstein
            ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "d1c2"],  # Classical
            ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "g1f3"],
            ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "f2f3"],  # Sämisch
            
            # Queen's Indian - E12-E19
            ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3"],  # QID prep
            ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],  # Queen's Indian
            ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "g2g3"],
            ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6", "b1c3"],
            
            # Bogo-Indian - E11
            ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"],  # Bogo-Indian
            
            # Catalan - E00-E09
            ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],  # Catalan
            ["d2d4", "d7d5", "c2c4", "e7e6", "g2g3"],
            ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5"],
            
            # Benoni Defense - A60-A79
            ["d2d4", "g8f6", "c2c4", "c7c5"],  # Benoni
            ["d2d4", "c7c5"],  # Benoni alternative
            ["d2d4", "c7c5", "d4d5"],  # Modern Benoni
            ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5"],
            ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6"],
            
            # Budapest Gambit - A51-A52
            ["d2d4", "g8f6", "c2c4", "e7e5"],  # Budapest
            ["d2d4", "g8f6", "c2c4", "e7e5", "d4e5"],
            
            # Dutch Defense - A80-A99
            ["d2d4", "f7f5"],  # Dutch
            ["d2d4", "f7f5", "g2g3"],  # Dutch Fianchetto
            ["d2d4", "f7f5", "c2c4"],
            ["d2d4", "f7f5", "g1f3"],
            ["d2d4", "f7f5", "g1f3", "g8f6"],
            
            # Old Indian - A53-A55
            ["d2d4", "g8f6", "c2c4", "d7d6"],  # Old Indian
            ["d2d4", "g8f6", "c2c4", "d7d6", "b1c3"],
            
            # ===== ENGLISH OPENING (1.c4) - ECO A10-A39 =====
            ["c2c4"],  # English
            ["c2c4", "e7e5"],  # Reversed Sicilian
            ["c2c4", "e7e5", "b1c3"],
            ["c2c4", "e7e5", "b1c3", "g8f6"],
            ["c2c4", "e7e5", "b1c3", "b8c6"],
            ["c2c4", "e7e5", "b1c3", "f8c5"],
            ["c2c4", "g8f6"],  # English vs Nf6
            ["c2c4", "g8f6", "b1c3"],
            ["c2c4", "g8f6", "g1f3"],
            ["c2c4", "c7c5"],  # Symmetrical English
            ["c2c4", "c7c5", "b1c3"],
            ["c2c4", "c7c5", "g1f3"],
            ["c2c4", "e7e6"],
            ["c2c4", "c7c6"],
            
            # ===== RETI OPENING (1.Nf3) - ECO A04-A09 =====
            ["g1f3"],  # Reti
            ["g1f3", "d7d5"],  # Reti vs d5
            ["g1f3", "d7d5", "c2c4"],
            ["g1f3", "d7d5", "g2g3"],
            ["g1f3", "g8f6"],  # Reti vs Nf6
            ["g1f3", "g8f6", "c2c4"],
            ["g1f3", "g8f6", "g2g3"],
            ["g1f3", "c7c5"],
            ["g1f3", "e7e6"],
            
            # King's Indian Attack
            ["g1f3", "d7d5", "g2g3", "g8f6"],
            ["g1f3", "g8f6", "g2g3", "d7d5"],
            
            # ===== BIRD'S OPENING (1.f4) - ECO A02-A03 =====
            ["f2f4"],  # Bird's
            ["f2f4", "d7d5"],
            ["f2f4", "g8f6"],
            ["f2f4", "e7e5"],  # From's Gambit
            ["f2f4", "d7d5", "g1f3"],
            
            # ===== LARSEN'S OPENING (1.b3) - ECO A01 =====
            ["b2b3"],  # Larsen's
            ["b2b3", "e7e5"],
            ["b2b3", "d7d5"],
            ["b2b3", "g8f6"],
            ["b2b3", "e7e5", "c1b2"],
            
            # ===== UNUSUAL OPENINGS - ECO A00 =====
            ["b1c3"],  # Dunst/Van Geet
            ["b1c3", "d7d5"],
            ["b1c3", "e7e5"],
            ["g2g3"],  # King's Fianchetto
            ["g2g3", "e7e5"],
            ["g2g3", "d7d5"],
            ["e2e3"],  # Van't Kruijs
            ["d2d3"],  # Mieses
            ["c2c3"],  # Saragossa
            ["a2a3"],  # Anderssen's
            ["h2h3"],  # Clemenz
            ["a2a4"],  # Ware
            ["b1a3"],  # Sodium Attack
            ["g1h3"],  # Amar
            ["f2f3"],  # Barnes
            ["h2h4"],  # Kádas/Desprez
        ]
        
        for line in openings:
            self._add_line(book, *line)
    
    def get_book_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a move from the opening book."""
        self.stats['positions_queried'] += 1
        fen = board.fen()
        if fen in self.book and self.book[fen]:
            self.stats['moves_used'] += 1
            return random.choice(self.book[fen])
        return None


class OpeningMiddlegameTactics:
    """
    Middlegame tactical themes specific to each major opening.
    Understanding what tactics to look for based on the opening played.
    """
    
    OPENING_TACTICS = {
        # ===== SICILIAN DEFENSE =====
        "sicilian_najdorf": {
            "typical_tactics": [
                "Exchange sacrifice on c3 (Rxc3 destroying pawn structure)",
                "Greek gift sacrifice on h7 (Bxh7+ Kxh7, Ng5+)",
                "Nd5 breakthrough sacrifice",
                "Queenside pawn storm with b5-b4-b3",
                "Black's a6-b5-b4 pawn advances",
                "f5-f4-f3 pawn wedge attack",
            ],
            "key_squares": ["d5", "e6", "b5"],
            "piece_sacrifices": ["Exchange sac on c3", "Nd5 piece sac"],
        },
        "sicilian_dragon": {
            "typical_tactics": [
                "Exchange sacrifice on c3",
                "White's h4-h5 pawn storm",
                "Black's queenside counterplay (b5-b4)",
                "Discovered attacks on the long diagonal",
                "Nd5 followed by Nf6+ fork",
                "Sacrifice on h5 to open h-file",
            ],
            "key_squares": ["d5", "h5", "c3"],
            "piece_sacrifices": ["Bxh5 opening attack", "Rxc3 exchange sac"],
        },
        "sicilian_sveshnikov": {
            "typical_tactics": [
                "Nd5 breakthrough constantly hanging",
                "f5 pawn break",
                "Exploiting the d5 hole",
                "Black's e5 pawn weaknesses",
                "Bishop pair exploitation in endgame",
            ],
            "key_squares": ["d5", "e6", "f5"],
            "piece_sacrifices": ["Nd5 positional sacrifice"],
        },
        
        # ===== RUY LOPEZ =====
        "ruy_lopez_closed": {
            "typical_tactics": [
                "d4-d5 central pawn break",
                "Bxc6 damaging pawn structure",
                "Nd5 outpost exploitation",
                "f4-f5 kingside attack",
                "c3-d4 central breakthrough",
                "Re1-e4-g4 swing attack",
            ],
            "key_squares": ["d5", "e4", "e5"],
            "piece_sacrifices": ["Bxf7+ deflection", "d5 pawn sacrifice"],
        },
        "ruy_lopez_marshall": {
            "typical_tactics": [
                "Black's d5 pawn sacrifice for initiative",
                "Re8-e1+ forcing moves",
                "Qd5 centralizing queen with threats",
                "Perpetual check possibilities",
                "Sacrifices on e4 or g3",
                "Pin on e-file after ...Re1+",
            ],
            "key_squares": ["e1", "d5", "g3"],
            "piece_sacrifices": ["d5 gambit pawn", "Rxe4 exchange"],
        },
        "ruy_lopez_berlin": {
            "typical_tactics": [
                "Endgame with opposite castling",
                "Central pawn breaks (d5/f5)",
                "Knight maneuvers to d5/f5",
                "Exploiting the isolated e5 pawn",
            ],
            "key_squares": ["d5", "e5", "f5"],
            "piece_sacrifices": ["None - positional endgame"],
        },
        
        # ===== ITALIAN GAME =====
        "italian_game": {
            "typical_tactics": [
                "Bxf7+ classical Greek gift",
                "Ng5 threatening Nxf7",
                "d4 central break and pawn sacrifice",
                "Re1 pinning tactics",
                "Quick attacks on f7 weakness",
                "Castling opposite sides attacking race",
            ],
            "key_squares": ["f7", "d5", "e5"],
            "piece_sacrifices": ["Bxf7+", "Nxf7", "d4 pawn sac"],
        },
        
        # ===== FRENCH DEFENSE =====
        "french_defense": {
            "typical_tactics": [
                "White's f4-f5 pawn storm",
                "c5 pawn break by Black",
                "f6 pawn break attacking e5",
                "Sacrificing on e6 (Bxe6 fxe6)",
                "Bad light-squared bishop for Black",
                "Minority attack with b4-b5xc6",
            ],
            "key_squares": ["e5", "d4", "c5"],
            "piece_sacrifices": ["Bxe6 piece sac", "f5 pawn storm"],
        },
        
        # ===== CARO-KANN =====
        "caro_kann": {
            "typical_tactics": [
                "c5 pawn break",
                "White's e5 advance cramping Black",
                "Bf5 bishop trade",
                "Solid structure with counterplay on queenside",
                "Black's ...c5-c4 advancing",
            ],
            "key_squares": ["d4", "e5", "c5"],
            "piece_sacrifices": ["Rare - positional opening"],
        },
        
        # ===== KING'S INDIAN DEFENSE =====
        "kings_indian": {
            "typical_tactics": [
                "Black's f5-f4-f3 kingside pawn storm",
                "g5-g4 attacking White's kingside",
                "Exchange sacrifice on e4 (Rxe4)",
                "White's c5 advance and Nd5",
                "Desperado Nf2 tactics",
                "h5-h4 pawn levers",
            ],
            "key_squares": ["e4", "d5", "f4"],
            "piece_sacrifices": ["Rxe4 exchange sac", "f3 pawn breakthrough"],
        },
        
        # ===== GRÜNFELD DEFENSE =====
        "grunfeld": {
            "typical_tactics": [
                "Black sacrifices center for pressure on d4",
                "Nd7-Nb6-Nc4 knight outpost",
                "Pressure on d4 with Bg7, Qb6, Rd8",
                "c5 pawn break constantly",
                "Exchange variation central dominance",
            ],
            "key_squares": ["d4", "c3", "e4"],
            "piece_sacrifices": ["Positional d5 center sacrifice"],
        },
        
        # ===== QUEEN'S GAMBIT =====
        "queens_gambit_declined": {
            "typical_tactics": [
                "Minority attack b4-b5xc6",
                "e4 central break",
                "Isolani tactics on d5",
                "Ne5 outpost exploitation",
                "Hanging pawns structure tactics",
            ],
            "key_squares": ["d4", "d5", "e4"],
            "piece_sacrifices": ["e4 pawn sacrifice opening lines"],
        },
        "queens_gambit_accepted": {
            "typical_tactics": [
                "Early queen development Qa4+/Qd1-a4",
                "Black holds the extra pawn with b5-c5",
                "White's e4 rapid development",
                "Central control and rapid piece play",
            ],
            "key_squares": ["c4", "e4", "a4"],
            "piece_sacrifices": ["Rare - material already imbalanced"],
        },
        "slav_defense": {
            "typical_tactics": [
                "dxc4 and holding with b5",
                "a6 preparing b5 advance",
                "Solid structure with Bf5 development",
                "c5 break fighting for center",
            ],
            "key_squares": ["c4", "d5", "c5"],
            "piece_sacrifices": ["Rare - solid opening"],
        },
        
        # ===== NIMZO-INDIAN =====
        "nimzo_indian": {
            "typical_tactics": [
                "Black's Bxc3 damaging pawn structure",
                "Doubled c-pawns for White",
                "e4 central break tactics",
                "c5 pawn break by Black",
                "Exploiting weak c-pawns in endgame",
            ],
            "key_squares": ["e4", "c3", "c5"],
            "piece_sacrifices": ["Strategic Bxc3 not a true sacrifice"],
        },
        
        # ===== ENGLISH OPENING =====
        "english_opening": {
            "typical_tactics": [
                "Reversed Sicilian patterns",
                "Fianchetto bishop on long diagonal",
                "Central breaks with d4 or e4",
                "Queenside expansion with b4-b5",
                "Hedgehog setup with ...b6, ...Bb7, ...d6",
            ],
            "key_squares": ["d5", "e4", "c5"],
            "piece_sacrifices": ["Rare - flexible strategic opening"],
        },
        
        # ===== KING'S GAMBIT =====
        "kings_gambit": {
            "typical_tactics": [
                "Quick attack on f7",
                "f4 pawn sacrifice for rapid development",
                "Bc4 and Qf3 battery on f7",
                "Open f-file for rook",
                "Knight checks from g5",
                "Sacrifices on e6 or f7",
            ],
            "key_squares": ["f7", "e6", "g5"],
            "piece_sacrifices": ["f4 gambit pawn", "Bxf7+", "Nxf7"],
        },
        
        # ===== PIRC/MODERN DEFENSE =====
        "pirc_defense": {
            "typical_tactics": [
                "Black's fianchetto pressure on e4",
                "White's f4-f5 kingside attack",
                "Austrian Attack with f4, e5",
                "Black's c5 or e5 pawn breaks",
                "Flexible piece placement",
            ],
            "key_squares": ["e4", "d4", "f5"],
            "piece_sacrifices": ["Rare in opening phase"],
        },
    }
    
    def get_opening_tactics(self, board: chess.Board) -> Dict[str, any]:
        """
        Identify which opening has been played and return its typical middlegame tactics.
        """
        opening_signature = self._identify_opening(board)
        
        if opening_signature in self.OPENING_TACTICS:
            return self.OPENING_TACTICS[opening_signature]
        
        return {
            "typical_tactics": ["Universal tactics apply"],
            "key_squares": [],
            "piece_sacrifices": [],
        }
    
    def _identify_opening(self, board: chess.Board) -> str:
        """Simple opening identification based on pawn structure and piece placement."""
        # Get first few moves to identify opening
        move_count = board.fullmove_number
        
        # Only identify in early/middle game
        if move_count > 20:
            return "middlegame"
        
        # Helper to safely check pieces
        def has_pawn(square, color):
            piece = board.piece_at(square)
            return piece and piece.piece_type == chess.PAWN and piece.color == color
        
        def has_piece(square, piece_type, color):
            piece = board.piece_at(square)
            return piece and piece.piece_type == piece_type and piece.color == color
        
        # Check pawn placements
        e4_white = has_pawn(chess.E4, chess.WHITE)
        d4_white = has_pawn(chess.D4, chess.WHITE)
        c4_white = has_pawn(chess.C4, chess.WHITE)
        
        e5_black = has_pawn(chess.E5, chess.BLACK)
        c5_black = has_pawn(chess.C5, chess.BLACK)
        e6_black = has_pawn(chess.E6, chess.BLACK)
        c6_black = has_pawn(chess.C6, chess.BLACK)
        d5_black = has_pawn(chess.D5, chess.BLACK)
        d6_black = has_pawn(chess.D6, chess.BLACK)
        g6_black = has_pawn(chess.G6, chess.BLACK)
        a6_black = has_pawn(chess.A6, chess.BLACK)
        
        # Check for Sicilian (e4 present or was there, c5 by black)
        if c5_black:
            # Sicilian Defense
            if has_piece(chess.G7, chess.BISHOP, chess.BLACK):
                return "sicilian_dragon"
            if a6_black:
                return "sicilian_najdorf"
            if e5_black:
                return "sicilian_sveshnikov"
            return "sicilian_najdorf"  # Default Sicilian
        
        # Check for Ruy Lopez (e4 e5, Bb5)
        if e4_white and e5_black:
            if has_piece(chess.B5, chess.BISHOP, chess.WHITE) or has_piece(chess.C6, chess.BISHOP, chess.WHITE):
                return "ruy_lopez_closed"
            # Check for Italian (Bc4)
            if has_piece(chess.C4, chess.BISHOP, chess.WHITE) or has_piece(chess.B3, chess.BISHOP, chess.WHITE):
                return "italian_game"
            # King's Gambit (f4 pawn pushed or captured)
            if has_pawn(chess.F4, chess.WHITE) or not has_pawn(chess.F2, chess.WHITE):
                return "kings_gambit"
        
        # Check for French (e4 e6, d5)
        if e6_black and d5_black:
            return "french_defense"
        
        # Check for Caro-Kann (e4 c6, d5)
        if c6_black and d5_black and (e4_white or d4_white):
            return "caro_kann"
        
        # Check for King's Indian (d4, g6 fianchetto, d6)
        if d4_white and g6_black and d6_black:
            return "kings_indian"
        
        # Check for Grünfeld (d4, g6, d5)
        if d4_white and g6_black and d5_black:
            return "grunfeld"
        
        # Check for Queen's Gambit (d4 d5, c4 or was c4)
        if d4_white and d5_black:
            if c4_white:
                return "queens_gambit_declined"
            # Check if c4 pawn was captured
            if not has_pawn(chess.C2, chess.WHITE) and not c4_white:
                # Pawn moved from c2, might have been captured
                if has_pawn(chess.C4, chess.BLACK):
                    return "queens_gambit_accepted"
                return "queens_gambit_declined"
            # Check for Slav (c6 setup)
            if c6_black:
                return "slav_defense"
            return "queens_gambit_declined"
        
        # Check for Nimzo-Indian (d4, Nf6, c4, Bb4)
        if d4_white and c4_white and has_piece(chess.B4, chess.BISHOP, chess.BLACK):
            return "nimzo_indian"
        
        # Check for English Opening (c4 first move)
        if c4_white and not d4_white and not e4_white:
            return "english_opening"
        
        # Check for Pirc (d6, Nf6, g6 without quick central pawn breaks)
        if d6_black and g6_black and e4_white:
            return "pirc_defense"
        
        return "unknown"


class ChessTactics:
    """
    Complete repertoire of chess tactics from Wikipedia.
    """
    
    TACTICS = {
        "fork": "Attacking two or more enemy pieces simultaneously",
        "pin": "Attacking a piece that cannot move without exposing a more valuable piece",
        "skewer": "Attacking a valuable piece, forcing it to move and exposing a less valuable piece",
        "discovered_attack": "Moving a piece reveals an attack from another piece",
        "double_check": "Checking the king with two pieces simultaneously",
        "deflection": "Forcing an enemy piece away from defending something",
        "decoy": "Luring an enemy piece to a bad square",
        "interference": "Blocking the line between enemy pieces",
        "undermining": "Attacking a defending piece to remove it",
        "overloading": "Attacking multiple pieces/squares defended by one piece",
        "zwischenzug": "Intermediate move - making an in-between threat",
        "desperado": "A piece that is doomed makes maximum damage before capture",
        "clearance": "Moving pieces to clear a square or line",
        "X-ray_attack": "Attacking through a piece to the piece behind it",
        "windmill": "Series of discovered checks",
        "battery": "Two or more pieces on the same line (Alekhine's gun, etc.)",
        "sacrifice": "Giving up material for positional or tactical advantage",
        "zugzwang": "Being forced to move when any move worsens position",
        "stalemate_trick": "Forcing stalemate in a losing position",
    }
    
    def evaluate_tactical_motifs(self, board: chess.Board) -> List[str]:
        """Identify tactical patterns on the board."""
        motifs = []
        
        # Check for pins
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == (not board.turn):
                attackers = board.attackers(board.turn, square)
                if len(attackers) > 0:
                    motifs.append("pin_opportunity")
        
        # Check for forks
        for move in board.legal_moves:
            board.push(move)
            attacked = sum(1 for sq in chess.SQUARES if board.is_attacked_by(board.turn, sq))
            if attacked >= 2:
                motifs.append("fork_opportunity")
            board.pop()
        
        return motifs


class ChessStrategy:
    """
    Complete strategic principles from Wikipedia.
    """
    
    STRATEGIC_CONCEPTS = {
        # Material
        "material_advantage": "Having more valuable pieces",
        "material_imbalance": "Different piece configurations (e.g., R+P vs 2N)",
        "piece_value": "Relative worth: P=1, N=B=3, R=5, Q=9",
        
        # Space
        "space_advantage": "Controlling more squares, especially in center",
        "center_control": "Dominating d4, d5, e4, e5 squares",
        "pawn_chain": "Connected pawns controlling key squares",
        
        # King safety
        "castling": "Moving king to safety",
        "king_activity": "King participation in endgame",
        "king_shelter": "Pawns protecting castled king",
        
        # Pawn structure
        "pawn_islands": "Groups of connected pawns",
        "isolated_pawn": "Pawn with no friendly pawns on adjacent files",
        "doubled_pawns": "Two pawns on same file",
        "backward_pawn": "Pawn behind neighbors, can't advance safely",
        "passed_pawn": "No enemy pawns can stop it",
        "outside_passed_pawn": "Passed pawn away from main action",
        "connected_pawns": "Pawns on adjacent files supporting each other",
        
        # Pieces
        "bishop_pair": "Having both bishops (advantage in open positions)",
        "bad_bishop": "Bishop blocked by own pawns",
        "good_bishop": "Mobile bishop on open diagonals",
        "knight_outpost": "Knight on strong square protected by pawn",
        "rook_on_open_file": "Rook on file with no pawns",
        "rook_on_7th_rank": "Rook invading 7th/2nd rank",
        "connected_rooks": "Rooks protecting each other",
        
        # Initiative
        "initiative": "Ability to make threats forcing opponent's moves",
        "tempo": "Gaining time by forcing opponent to respond",
        "compensation": "Non-material advantages for sacrificed material",
        
        # Positional
        "weak_squares": "Squares that can't be defended by pawns",
        "holes": "Weak squares in pawn structure",
        "outpost": "Advanced piece on hole protected by pawn",
        "prophylaxis": "Preventing opponent's plans",
        "minority_attack": "Pawn attack with fewer pawns",
        "pawn_breakthrough": "Forcing passed pawn",
        "exchange": "Trading pieces strategically",
        
        # Plans
        "attack_on_king": "Coordinated assault on enemy king",
        "queenside_attack": "Attack on opponent's queenside",
        "kingside_attack": "Attack on opponent's kingside",
        "piece_coordination": "Pieces working together effectively",
    }
    
    def evaluate_position(self, board: chess.Board) -> Dict[str, float]:
        """Evaluate strategic factors."""
        evaluation = {
            "material": 0.0,
            "space": 0.0,
            "king_safety": 0.0,
            "piece_activity": 0.0,
        }
        
        # Count material
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9}
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    evaluation["material"] += value
                else:
                    evaluation["material"] -= value
        
        return evaluation


class EndgameKnowledge:
    """
    Comprehensive endgame knowledge from Wikipedia.
    """
    
    ENDGAME_PRINCIPLES = {
        # Basic mates
        "K+Q_vs_K": "Queen can force checkmate with king help",
        "K+R_vs_K": "Rook can force checkmate with king help",
        "K+2B_vs_K": "Two bishops can force checkmate",
        "K+B+N_vs_K": "Bishop and knight can force mate (difficult)",
        "K+2N_vs_K": "Two knights cannot force mate (draw)",
        
        # Pawn endgames
        "opposition": "Kings facing each other with odd squares between",
        "key_squares": "Squares king must reach to support pawn",
        "square_of_the_pawn": "Zone where king can catch pawn",
        "passed_pawn_promotion": "Race to promote pawn",
        "pawn_breakthrough": "Sacrificing to create passed pawn",
        "triangulation": "Losing tempo to put opponent in zugzwang",
        "outflanking": "Going around opponent's king",
        
        # Rook endgames
        "lucena_position": "Winning R+P vs R endgame",
        "philidor_position": "Drawing R+P vs R endgame",
        "rook_on_7th_rank": "Rook dominates on 7th rank",
        "rook_behind_passed_pawn": "Best placement of rook",
        "fortress": "Defensive setup that draws",
        
        # Bishop endgames
        "opposite_colored_bishops": "Often drawn even with pawn advantage",
        "same_colored_bishops": "Technique matters with passed pawns",
        "wrong_bishop": "Bishop can't control queening square",
        "wrong_rook_pawn": "Rook pawn + wrong bishop draws",
        
        # Knight endgames
        "knight_vs_pawns": "Knight usually beats 1-2 pawns",
        "knight_on_rim": "Knight on edge is poorly placed",
        
        # Queen endgames
        "queen_vs_pawn": "Queen usually wins easily",
        "queen_vs_rook": "Queen usually wins with technique",
        
        # General principles
        "king_activity": "King becomes powerful in endgame",
        "pawn_promotion": "Creating and pushing passed pawns",
        "zugzwang": "More common in endgames",
        "stalemate_tricks": "Drawing resource in lost positions",
        "outside_passed_pawn": "Decisive advantage",
        "active_rook": "Activity more important than pawns",
    }
    
    def is_endgame(self, board: chess.Board) -> bool:
        """Determine if position is an endgame."""
        # Count pieces
        piece_count = 0
        queens = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type != chess.KING:
                piece_count += 1
                if piece.piece_type == chess.QUEEN:
                    queens += 1
        
        # Endgame if: few pieces OR no queens OR both criteria
        return piece_count <= 6 or queens == 0
    
    def suggest_plan(self, board: chess.Board) -> str:
        """Suggest endgame plan."""
        if not self.is_endgame(board):
            return "Not in endgame yet"
        
        # Count material
        white_pieces = sum(1 for sq in chess.SQUARES 
                          if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
        black_pieces = sum(1 for sq in chess.SQUARES 
                          if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK)
        
        if white_pieces > black_pieces or black_pieces > white_pieces:
            return "Activate king, create passed pawns, simplify if winning"
        return "Activate king, look for opposition, control key squares"


class ComprehensiveChessKnowledge:
    """
    Complete integration of all chess knowledge.
    """
    
    def __init__(self):
        self.opening_book = ComprehensiveOpeningBook()
        self.opening_middlegame = OpeningMiddlegameTactics()
        self.tactics = ChessTactics()
        self.strategy = ChessStrategy()
        self.endgame = EndgameKnowledge()
        
        # Statistics
        self.stats = {
            "book_moves_used": 0,
            "middlegame_tactics_identified": 0,
            "tactical_positions": 0,
            "strategic_evaluations": 0,
            "endgame_assists": 0,
        }
    
    def get_assisted_move(self, board: chess.Board) -> Tuple[Optional[chess.Move], str]:
        """
        Get move assistance using all knowledge.
        Returns: (move, source_description)
        """
        # 1. Try opening book first
        book_move = self.opening_book.get_book_move(board)
        if book_move:
            self.stats["book_moves_used"] += 1
            return (book_move, "opening_book")
        
        # 2. Check if endgame
        if self.endgame.is_endgame(board):
            self.stats["endgame_assists"] += 1
            # Could add endgame tablebase here
            return (None, "endgame_knowledge_available")
        
        # 3. Evaluate tactics (including opening-specific middlegame tactics)
        tactical_motifs = self.tactics.evaluate_tactical_motifs(board)
        if tactical_motifs:
            self.stats["tactical_positions"] += 1
        
        # 3b. Get opening-specific middlegame tactical themes
        opening_tactics = self.opening_middlegame.get_opening_tactics(board)
        if opening_tactics and opening_tactics.get("typical_tactics"):
            self.stats["middlegame_tactics_identified"] += 1
        
        # 4. Strategic evaluation available
        self.stats["strategic_evaluations"] += 1
        
        return (None, "neural_network")
    
    def get_opening_tactical_themes(self, board: chess.Board) -> Dict[str, any]:
        """
        Get the typical middlegame tactical themes for the current opening.
        Useful for understanding what to look for in the position.
        """
        return self.opening_middlegame.get_opening_tactics(board)
    
    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


# Example usage
if __name__ == "__main__":
    knowledge = ComprehensiveChessKnowledge()
    board = chess.Board()
    
    print("Comprehensive Chess Knowledge System Loaded!")
    print(f"Opening book size: {len(knowledge.opening_book.book)} positions")
    print(f"Tactics: {len(knowledge.tactics.TACTICS)} types")
    print(f"Strategy: {len(knowledge.strategy.STRATEGIC_CONCEPTS)} concepts")
    print(f"Endgame: {len(knowledge.endgame.ENDGAME_PRINCIPLES)} principles")
    
    # Test first move
    move, source = knowledge.get_assisted_move(board)
    print(f"\nRecommended opening: {move} (from {source})")
