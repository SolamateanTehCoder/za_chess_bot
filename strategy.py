#!/usr/bin/env python3
"""
Chess Strategy System - Multiple strategic playing styles for the bot.
Implements 8 different chess strategies: Aggressive, Defensive, Positional,
Tactical, Endgame, Opening, Balanced, and Machine Learning-focused.
"""

import chess
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class StrategyType(Enum):
    """Chess strategy types."""
    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    POSITIONAL = "positional"
    TACTICAL = "tactical"
    ENDGAME = "endgame"
    OPENING = "opening"
    BALANCED = "balanced"
    MACHINE_LEARNING = "machine_learning"


class ChessStrategy:
    """Base class for chess strategies with configurable parameters."""
    
    def __init__(self, name: str, config: Dict):
        """Initialize strategy with configuration."""
        self.name = name
        self.config = config
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
    
    @staticmethod
    def from_config(strategy_name: str) -> "ChessStrategy":
        """Create strategy from config dictionary."""
        if strategy_name not in STRATEGY_CONFIGS:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        config = STRATEGY_CONFIGS[strategy_name]
        return ChessStrategy(strategy_name, config)
    
    def evaluate_move(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate a move based on strategy parameters.
        Returns score between 0 and 100.
        """
        score = 50.0  # Base neutral score
        
        # Check if move is capturing
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                capture_value = self.piece_values.get(captured_piece.piece_type, 0)
            else:
                # En passant capture (always a pawn)
                capture_value = self.piece_values.get(chess.PAWN, 1)
            score += self.config['capture_weight'] * capture_value
        
        # Check if move gives check
        if board.gives_check(move):
            score += self.config['check_weight']
        
        # Check for threats
        board_copy = board.copy()
        board_copy.push(move)
        
        if self._is_king_safe(board_copy, board.turn):
            score += self.config['safety_weight']
        
        # Piece activity evaluation
        piece = board.piece_at(move.from_square)
        if piece:
            activity = self._evaluate_piece_activity(board_copy, move.to_square, piece.color)
            score += self.config['activity_weight'] * activity
        
        # Positional evaluation
        position_score = self._evaluate_position(board_copy, board.turn)
        score += self.config['position_weight'] * position_score
        
        # Endgame specific
        if len(board_copy.pieces(chess.QUEEN, chess.WHITE)) == 0 and \
           len(board_copy.pieces(chess.QUEEN, chess.BLACK)) == 0:
            score += self.config['endgame_weight'] * 10
        
        return max(0, min(100, score))  # Clamp between 0 and 100
    
    def select_best_move(self, board: chess.Board, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        """
        Select best move based on strategy evaluation.
        Returns the highest scoring legal move.
        """
        if not legal_moves:
            return None
        
        # Evaluate all legal moves
        move_scores = []
        for move in legal_moves:
            score = self.evaluate_move(board, move)
            move_scores.append((move, score))
        
        # Sort by score descending
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best move
        return move_scores[0][0]
    
    def _is_king_safe(self, board: chess.Board, color: bool) -> bool:
        """Check if king of given color is safe (not under attack)."""
        king_square = board.king(color)
        if king_square is None:
            return False
        
        opponent_color = not color
        # Check if any opponent piece attacks the king
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == opponent_color:
                # Create a test move from this piece to king
                test_move = chess.Move(square, king_square)
                if test_move in board.legal_moves:
                    return False
        
        return True
    
    def _evaluate_piece_activity(self, board: chess.Board, square: int, color: bool) -> float:
        """Evaluate how active a piece is (0-10 scale)."""
        piece = board.piece_at(square)
        if piece is None:
            return 0.0
        
        piece_type = piece.piece_type
        
        # Count possible moves for this piece
        temp_moves = 0
        for move in board.legal_moves:
            if move.from_square == square:
                temp_moves += 1
        
        # Normalize to 0-10 scale
        max_moves = {
            chess.PAWN: 2,
            chess.KNIGHT: 8,
            chess.BISHOP: 13,
            chess.ROOK: 14,
            chess.QUEEN: 27,
            chess.KING: 8
        }
        
        max_m = max_moves.get(piece_type, 1)
        return min(10, (temp_moves / max_m) * 10)
    
    def _evaluate_position(self, board: chess.Board, color: bool) -> float:
        """
        Evaluate position based on material and control.
        Returns score between -10 and 10.
        """
        score = 0.0
        
        # Material balance
        for piece_type in chess.PIECE_TYPES:
            our_pieces = len(board.pieces(piece_type, color))
            opp_pieces = len(board.pieces(piece_type, not color))
            piece_value = self.piece_values[piece_type]
            
            if piece_value > 0:
                score += (our_pieces - opp_pieces) * piece_value / 10.0
        
        # Center control (rough estimation)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += 0.5
            elif piece and piece.color != color:
                score -= 0.5
        
        return max(-10, min(10, score))


# Strategy Configurations - Tuned parameters for each strategy type
STRATEGY_CONFIGS = {
    "aggressive": {
        "name": "Aggressive",
        "description": "Prioritizes attacks, checks, and captures",
        "capture_weight": 4.0,      # High capture priority
        "check_weight": 8.0,         # Aggressive checking
        "safety_weight": 2.0,        # Lower safety concern
        "activity_weight": 3.0,      # Active piece play
        "position_weight": 2.0,      # Less positional
        "endgame_weight": 3.0,       # Aggressive in endgame
        "best_for": "Early middle game, attacking positions"
    },
    
    "defensive": {
        "name": "Defensive",
        "description": "Prioritizes safety, material preservation",
        "capture_weight": 2.0,       # Selective captures
        "check_weight": 2.0,         # Avoid risky checks
        "safety_weight": 8.0,        # Extreme safety focus
        "activity_weight": 2.0,      # Conservative piece movement
        "position_weight": 4.0,      # Positional awareness
        "endgame_weight": 2.0,       # Defensive endgame
        "best_for": "Losing positions, material disadvantage"
    },
    
    "positional": {
        "name": "Positional",
        "description": "Emphasizes structure, control, long-term advantage",
        "capture_weight": 2.0,       # Selective captures
        "check_weight": 1.0,         # Avoid random checks
        "safety_weight": 5.0,        # Moderate safety
        "activity_weight": 4.0,      # Piece coordination
        "position_weight": 8.0,      # Heavy positional focus
        "endgame_weight": 4.0,       # Positional endgame
        "best_for": "Long strategic games, building advantage"
    },
    
    "tactical": {
        "name": "Tactical",
        "description": "Looks for immediate tactical opportunities",
        "capture_weight": 5.0,       # High capture priority
        "check_weight": 6.0,         # Tactical checks
        "safety_weight": 3.0,        # Some safety concern
        "activity_weight": 5.0,      # Active tactics
        "position_weight": 2.0,      # Minimal positional concern
        "endgame_weight": 2.0,       # Quick tactical wins
        "best_for": "Middlegame with tactical opportunities"
    },
    
    "endgame": {
        "name": "Endgame",
        "description": "Optimized for endgame play, piece promotion",
        "capture_weight": 3.0,       # Selective captures
        "check_weight": 4.0,         # Strategic checks
        "safety_weight": 4.0,        # Moderate safety
        "activity_weight": 6.0,      # Activate pieces for promotion
        "position_weight": 6.0,      # Positional endgame understanding
        "endgame_weight": 9.0,       # Heavy endgame focus
        "best_for": "Endgame positions, few pieces remaining"
    },
    
    "opening": {
        "name": "Opening",
        "description": "Prioritizes principled opening development",
        "capture_weight": 1.0,       # Avoid early captures
        "check_weight": 1.0,         # Avoid early checks
        "safety_weight": 6.0,        # Development safety
        "activity_weight": 6.0,      # Piece development
        "position_weight": 7.0,      # Opening principles
        "endgame_weight": 1.0,       # Not relevant in opening
        "best_for": "Opening phase, principled development"
    },
    
    "balanced": {
        "name": "Balanced",
        "description": "Balanced approach considering all factors",
        "capture_weight": 3.0,       # Moderate captures
        "check_weight": 3.0,         # Balanced checking
        "safety_weight": 5.0,        # Balanced safety
        "activity_weight": 4.0,      # Balanced activity
        "position_weight": 5.0,      # Balanced position
        "endgame_weight": 4.0,       # Balanced endgame
        "best_for": "All positions, versatile play"
    },
    
    "machine_learning": {
        "name": "Machine Learning",
        "description": "Neural network focused, minimal heuristics",
        "capture_weight": 0.5,       # Minimal heuristic weight
        "check_weight": 0.5,         # Let NN decide
        "safety_weight": 1.0,        # Basic safety
        "activity_weight": 0.5,      # Minimal heuristics
        "position_weight": 0.5,      # Let NN learn
        "endgame_weight": 0.5,       # Let NN learn
        "best_for": "Pure neural network play"
    }
}


class StrategyAnalyzer:
    """Analyze strategy performance and effectiveness."""
    
    def __init__(self):
        self.strategy_stats = {name: {
            'games': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'total_moves': 0,
            'avg_eval': 0
        } for name in STRATEGY_CONFIGS.keys()}
    
    def record_game(self, white_strategy: str, black_strategy: str, 
                   result: str, moves: int):
        """Record game result for strategy analysis."""
        if white_strategy in self.strategy_stats:
            self.strategy_stats[white_strategy]['games'] += 1
            self.strategy_stats[white_strategy]['total_moves'] += moves
            
            if result == "1-0":
                self.strategy_stats[white_strategy]['wins'] += 1
            elif result == "1/2-1/2":
                self.strategy_stats[white_strategy]['draws'] += 1
            else:
                self.strategy_stats[white_strategy]['losses'] += 1
        
        if black_strategy in self.strategy_stats:
            self.strategy_stats[black_strategy]['games'] += 1
            self.strategy_stats[black_strategy]['total_moves'] += moves
            
            if result == "0-1":
                self.strategy_stats[black_strategy]['wins'] += 1
            elif result == "1/2-1/2":
                self.strategy_stats[black_strategy]['draws'] += 1
            else:
                self.strategy_stats[black_strategy]['losses'] += 1
    
    def get_win_rate(self, strategy: str) -> float:
        """Get win rate for a strategy."""
        stats = self.strategy_stats.get(strategy, {})
        games = stats.get('games', 0)
        if games == 0:
            return 0.0
        
        wins = stats.get('wins', 0)
        return (wins / games) * 100
    
    def get_best_strategy(self) -> Optional[str]:
        """Get the best performing strategy."""
        best = None
        best_win_rate = -1
        
        for strategy, stats in self.strategy_stats.items():
            if stats['games'] > 0:
                win_rate = (stats['wins'] / stats['games']) * 100
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best = strategy
        
        return best
    
    def print_summary(self):
        """Print strategy performance summary."""
        print("\n" + "="*70)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("="*70)
        
        for strategy, stats in sorted(self.strategy_stats.items()):
            if stats['games'] > 0:
                win_rate = (stats['wins'] / stats['games']) * 100
                print(f"\n{strategy.upper():20} | Games: {stats['games']:3} | "
                      f"W: {stats['wins']:2} D: {stats['draws']:2} L: {stats['losses']:2} | "
                      f"Win Rate: {win_rate:.1f}%")
