"""
Comprehensive Chess Strategies for Za Chess Bot.
Implements 10+ major chess strategic concepts and playstyles.
Allows bot to play against itself using different strategies.
"""

import chess
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
import random


class ChessStrategy(Enum):
    """All major chess strategies implemented."""
    AGGRESSIVE = "aggressive"           # Maximize piece activity and attacks
    DEFENSIVE = "defensive"             # Prioritize piece safety and structure
    POSITIONAL = "positional"           # Control key squares and space
    TACTICAL = "tactical"               # Maximize tactical opportunities
    MATERIAL = "material"               # Win material and favor favorable trades
    ENDGAME = "endgame"                 # Transition to winning endgames
    OPENING = "opening"                 # Strong opening play and development
    HYPERMODERN = "hypermodern"         # Control center from distance
    PROPHYLAXIS = "prophylaxis"         # Prevent opponent threats first
    FIANCHETTO = "fianchetto"           # Bishop fianchetto setup
    SOLIDIFYING = "solidifying"         # Build solid pawn structure
    SACRIFICIAL = "sacrificial"         # Material sacrifice for activity
    FORTRESS = "fortress"               # Defend seemingly lost positions


class StrategyEvaluator:
    """
    Evaluates positions and suggests moves based on specific strategies.
    Can be used to guide move selection in hybrid player.
    """
    
    def __init__(self):
        """Initialize strategy evaluator."""
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        # Center squares
        self.center_squares = [
            chess.D4, chess.E4, chess.D5, chess.E5,
            chess.C3, chess.C4, chess.C5, chess.C6,
            chess.F3, chess.F4, chess.F5, chess.F6
        ]
        
        # Extended center
        self.extended_center = [
            chess.B2, chess.B3, chess.B4, chess.B5, chess.B6, chess.B7,
            chess.G2, chess.G3, chess.G4, chess.G5, chess.G6, chess.G7,
            chess.C2, chess.C7, chess.F2, chess.F7,
            chess.D3, chess.D6, chess.E3, chess.E6
        ]
    
    def evaluate_position(self, board: chess.Board, strategy: ChessStrategy) -> float:
        """
        Evaluate position based on specific strategy.
        Returns a score from -100 to 100 (better for side to move).
        """
        if strategy == ChessStrategy.AGGRESSIVE:
            return self._evaluate_aggressive(board)
        elif strategy == ChessStrategy.DEFENSIVE:
            return self._evaluate_defensive(board)
        elif strategy == ChessStrategy.POSITIONAL:
            return self._evaluate_positional(board)
        elif strategy == ChessStrategy.TACTICAL:
            return self._evaluate_tactical(board)
        elif strategy == ChessStrategy.MATERIAL:
            return self._evaluate_material(board)
        elif strategy == ChessStrategy.ENDGAME:
            return self._evaluate_endgame(board)
        elif strategy == ChessStrategy.OPENING:
            return self._evaluate_opening(board)
        elif strategy == ChessStrategy.HYPERMODERN:
            return self._evaluate_hypermodern(board)
        elif strategy == ChessStrategy.PROPHYLAXIS:
            return self._evaluate_prophylaxis(board)
        elif strategy == ChessStrategy.FIANCHETTO:
            return self._evaluate_fianchetto(board)
        elif strategy == ChessStrategy.SOLIDIFYING:
            return self._evaluate_solidifying(board)
        elif strategy == ChessStrategy.SACRIFICIAL:
            return self._evaluate_sacrificial(board)
        elif strategy == ChessStrategy.FORTRESS:
            return self._evaluate_fortress(board)
        return 0.0
    
    def rank_moves_by_strategy(self, board: chess.Board, 
                               strategy: ChessStrategy, 
                               moves: Optional[List[chess.Move]] = None) -> List[chess.Move]:
        """
        Rank legal moves by strategy strength.
        Returns moves sorted from best to worst for the strategy.
        """
        if moves is None:
            moves = list(board.legal_moves)
        
        if not moves:
            return []
        
        # Evaluate each move
        move_scores = []
        for move in moves:
            board.push(move)
            score = self.evaluate_position(board, strategy)
            board.pop()
            move_scores.append((move, score))
        
        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]
    
    # Strategy-specific evaluation functions
    
    def _evaluate_aggressive(self, board: chess.Board) -> float:
        """
        Aggressive strategy: Maximize piece activity and attack potential.
        Rewards: Active pieces, checks, threats, pawn advances.
        """
        score = 0.0
        our_side = board.turn
        
        # Count checking moves available
        for move in board.legal_moves:
            if board.is_check():
                score += 5
        
        # Bonus for pieces on active squares
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, our_side):
                # Central pieces worth more
                if square in self.center_squares:
                    score += 2
                # Advanced pieces worth more
                if piece != chess.PAWN and chess.square_rank(square) >= 4:
                    score += 1
                # Attacking pieces
                if board.attackers(our_side, square):
                    score += 1.5
        
        # Penalties
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, not our_side):
                if board.attackers(our_side, square):
                    score += 2  # Can capture
        
        return min(score, 100.0)
    
    def _evaluate_defensive(self, board: chess.Board) -> float:
        """
        Defensive strategy: Prioritize piece safety and solid structure.
        Rewards: Protected pieces, pawn shields, safe pieces.
        """
        score = 0.0
        our_side = board.turn
        
        # Count protected pieces
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, our_side):
                if board.attackers(our_side, square):
                    score += 2  # Protected piece
        
        # Pawn structure
        pawn_squares = board.pieces(chess.PAWN, our_side)
        for square in pawn_squares:
            # Connected pawns
            neighbors = [square - 1, square + 1]
            if any(n in pawn_squares for n in neighbors if n >= 0):
                score += 1
            # Backward protected pawns
            if board.attackers(our_side, square):
                score += 1.5
        
        # Penalty for attacked undefended pieces
        for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            for square in board.pieces(piece, our_side):
                attackers = board.attackers(not our_side, square)
                defenders = board.attackers(our_side, square)
                if attackers and not defenders:
                    score -= 5
        
        return max(min(score, 100.0), -100.0)
    
    def _evaluate_positional(self, board: chess.Board) -> float:
        """
        Positional strategy: Control key squares and superior piece placement.
        Rewards: Center control, outposts, piece coordination.
        """
        score = 0.0
        our_side = board.turn
        opp_side = not our_side
        
        # Center control
        our_center = sum(1 for sq in self.center_squares 
                        if board.piece_at(sq) and board.piece_at(sq).color == our_side)
        opp_center = sum(1 for sq in self.center_squares 
                        if board.piece_at(sq) and board.piece_at(sq).color == opp_side)
        score += (our_center - opp_center) * 3
        
        # Extended center (important squares)
        our_extended = sum(1 for sq in self.extended_center 
                          if board.piece_at(sq) and board.piece_at(sq).color == our_side)
        opp_extended = sum(1 for sq in self.extended_center 
                          if board.piece_at(sq) and board.piece_at(sq).color == opp_side)
        score += (our_extended - opp_extended) * 1.5
        
        # Outposts (squares not attacked by opponent pawns)
        opp_pawns = board.pieces(chess.PAWN, opp_side)
        opp_pawn_control = set()
        for pawn_sq in opp_pawns:
            # Pawns attack diagonally
            attacks = [pawn_sq + 7, pawn_sq + 9] if opp_side else [pawn_sq - 7, pawn_sq - 9]
            opp_pawn_control.update(a for a in attacks if 0 <= a < 64)
        
        for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece, our_side):
                if square not in opp_pawn_control:
                    score += 2.5
        
        return min(score, 100.0)
    
    def _evaluate_tactical(self, board: chess.Board) -> float:
        """
        Tactical strategy: Maximize tactical opportunities and material gain.
        Rewards: Forks, pins, skewers, wins.
        """
        score = 0.0
        our_side = board.turn
        
        # Check for tactical opportunities
        for move in board.legal_moves:
            board.push(move)
            
            # Material gain
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                score += self.piece_values.get(captured.piece_type, 1) * 3
            
            # Check gives tempo
            if board.is_check():
                score += 2
            
            board.pop()
        
        # Penalty for hanging pieces
        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, our_side):
                if piece != chess.KING:
                    attackers = board.attackers(not our_side, square)
                    defenders = board.attackers(our_side, square)
                    if attackers and not defenders:
                        score -= self.piece_values.get(piece, 1) * 4
        
        return max(min(score, 100.0), -100.0)
    
    def _evaluate_material(self, board: chess.Board) -> float:
        """
        Material strategy: Maximize material advantage and favorable trades.
        Rewards: Material up, good trades.
        """
        score = 0.0
        our_side = board.turn
        
        # Calculate material balance
        for piece in chess.PIECE_TYPES:
            our_count = len(board.pieces(piece, our_side))
            opp_count = len(board.pieces(piece, not our_side))
            piece_val = self.piece_values.get(piece, 1)
            score += (our_count - opp_count) * piece_val * 10
        
        return max(min(score, 100.0), -100.0)
    
    def _evaluate_endgame(self, board: chess.Board) -> float:
        """
        Endgame strategy: Transition to favorable endgames.
        Rewards: Activating king, passed pawns, simplification when ahead.
        """
        score = 0.0
        our_side = board.turn
        
        # Check if endgame
        total_material = sum(len(board.pieces(p, our_side)) + len(board.pieces(p, not our_side))
                           for p in chess.PIECE_TYPES)
        
        # Passed pawns are very valuable in endgame
        for square in board.pieces(chess.PAWN, our_side):
            # Simple passed pawn check
            pawn_file = chess.square_file(square)
            pawn_rank = chess.square_rank(square)
            
            # Check if no enemy pawns on adjacent files
            enemy_pawns = board.pieces(chess.PAWN, not our_side)
            is_passed = True
            for ep in enemy_pawns:
                if abs(chess.square_file(ep) - pawn_file) <= 1:
                    if (chess.square_rank(ep) > pawn_rank if our_side else 
                        chess.square_rank(ep) < pawn_rank):
                        is_passed = False
                        break
            
            if is_passed:
                score += 15 + pawn_rank * 2
        
        # Activate king in endgame
        king_square = board.king(our_side)
        if total_material < 20:  # Endgame
            king_rank = chess.square_rank(king_square)
            king_file = chess.square_file(king_square)
            score += (3 - abs(king_file - 3.5)) * 2  # Closer to center
            score += king_rank * 3 if our_side else (7 - king_rank) * 3
        
        return min(score, 100.0)
    
    def _evaluate_opening(self, board: chess.Board) -> float:
        """
        Opening strategy: Prioritize development and piece coordination.
        Rewards: Developed pieces, castling readiness.
        """
        score = 0.0
        our_side = board.turn
        
        # Count developed pieces (not on starting squares)
        starting_squares = {
            True: [chess.B1, chess.C1, chess.F1, chess.G1, chess.B8, chess.C8, chess.F8, chess.G8],
            False: [chess.B1, chess.C1, chess.F1, chess.G1, chess.B8, chess.C8, chess.F8, chess.G8]
        }
        
        developed = 0
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.color == our_side:
                if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
                    if square not in starting_squares[our_side]:
                        developed += 1
                        score += 3
        
        # Reward castling
        if board.has_kingside_castling_rights(our_side):
            score += 2
        if board.has_queenside_castling_rights(our_side):
            score += 1.5
        
        # Control center
        for sq in self.center_squares:
            piece = board.piece_at(sq)
            if piece and piece.color == our_side:
                score += 1
        
        return min(score, 100.0)
    
    def _evaluate_hypermodern(self, board: chess.Board) -> float:
        """
        Hypermodern strategy: Control center from distance with pieces.
        Rewards: Long-range piece control, fianchettoed bishops.
        """
        score = 0.0
        our_side = board.turn
        
        # Long-range piece control
        for piece in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece, our_side):
                # Count controlled center squares
                piece_moves = len(list(board.attacks(square)))
                center_control = sum(1 for move in board.attacks(square) 
                                    if move in self.center_squares)
                score += center_control * 2
        
        # Fianchetto bonus
        for color in [chess.WHITE, chess.BLACK]:
            fianchetto_file = 6 if color == chess.WHITE else 1
            bishop_square = chess.F2 if color == chess.WHITE else chess.F7
            if (board.piece_at(bishop_square) and 
                board.piece_at(bishop_square).piece_type == chess.BISHOP):
                score += 4
        
        return min(score, 100.0)
    
    def _evaluate_prophylaxis(self, board: chess.Board) -> float:
        """
        Prophylaxis: Prevent opponent threats before they develop.
        Rewards: Blocking threats, restricting opponent pieces.
        """
        score = 0.0
        our_side = board.turn
        opp_side = not our_side
        
        # Find opponent threats and reward blocking them
        for move in board.legal_moves:
            board.push(move)
            
            # Check if move reduces opponent attacking possibilities
            opp_threat_count = 0
            for opp_move in board.legal_moves:
                if board.is_attacked_by(opp_side, board.king(our_side)):
                    opp_threat_count += 1
            
            if opp_threat_count > 0:
                score -= opp_threat_count * 0.5
            
            board.pop()
        
        return max(min(score, 100.0), -100.0)
    
    def _evaluate_fianchetto(self, board: chess.Board) -> float:
        """
        Fianchetto strategy: Use fianchettoed bishops on long diagonals.
        Rewards: Fianchetto setup, long diagonal control.
        """
        score = 0.0
        our_side = board.turn
        
        # Check fianchetto setups
        for color in [chess.WHITE, chess.BLACK]:
            if color != our_side:
                continue
            
            if color == chess.WHITE:
                fianchetto_pawn = chess.G2
                bishop_diagonal = chess.G2
            else:
                fianchetto_pawn = chess.G7
                bishop_diagonal = chess.G7
            
            # Pawn on g2/g7
            if board.piece_at(fianchetto_pawn) and \
               board.piece_at(fianchetto_pawn).piece_type == chess.PAWN:
                score += 3
            
            # Bishop on fianchetto
            bishop_sq = chess.F1 if color == chess.WHITE else chess.F8
            if board.piece_at(bishop_sq) and \
               board.piece_at(bishop_sq).piece_type == chess.BISHOP:
                score += 4
        
        return min(score, 100.0)
    
    def _evaluate_solidifying(self, board: chess.Board) -> float:
        """
        Solidifying: Build solid pawn structure and defense.
        Rewards: Pawn chains, protected pieces, structure.
        """
        score = 0.0
        our_side = board.turn
        
        # Pawn chains
        pawn_squares = board.pieces(chess.PAWN, our_side)
        for square in pawn_squares:
            # Check for pawn support
            supporters = 0
            if square + 7 in pawn_squares:
                supporters += 1
            if square + 9 in pawn_squares:
                supporters += 1
            score += supporters * 2
        
        # Doubled/tripled pawns penalty
        file_pawns = {}
        for square in pawn_squares:
            file = chess.square_file(square)
            file_pawns[file] = file_pawns.get(file, 0) + 1
        
        for file, count in file_pawns.items():
            if count > 1:
                score -= (count - 1) * 3
        
        return max(min(score, 100.0), -100.0)
    
    def _evaluate_sacrificial(self, board: chess.Board) -> float:
        """
        Sacrificial strategy: Value material sacrifice for initiative.
        Rewards: Sacrifices with compensation, attacking chances.
        """
        score = 0.0
        
        # Look for moves that sacrifice material
        for move in board.legal_moves:
            board.push(move)
            
            if board.is_capture(move):
                # Sacrifice = capture our piece first
                score -= self.piece_values.get(board.piece_at(move.to_square).piece_type, 1) * 5
            
            # Reward checks and attacks
            if board.is_check():
                score += 8
            
            # Reward move to active squares
            if move.to_square in self.center_squares:
                score += 3
            
            board.pop()
        
        return max(min(score, 100.0), -100.0)
    
    def _evaluate_fortress(self, board: chess.Board) -> float:
        """
        Fortress: Defend seemingly lost positions and hold draws.
        Rewards: Compact defense, piece coordination, fortress patterns.
        """
        score = 0.0
        our_side = board.turn
        opp_side = not our_side
        
        # Check material imbalance (need it for fortress)
        our_material = sum(self.piece_values.get(p, 1) * len(board.pieces(p, our_side))
                          for p in chess.PIECE_TYPES)
        opp_material = sum(self.piece_values.get(p, 1) * len(board.pieces(p, opp_side))
                          for p in chess.PIECE_TYPES)
        
        if opp_material > our_material:  # We're down material
            # Reward compact, defended positions
            for piece in chess.PIECE_TYPES:
                for square in board.pieces(piece, our_side):
                    defenders = board.attackers(our_side, square)
                    if defenders:
                        score += 2
            
            # Reward restricted opponent moves
            legal_moves = len(list(board.legal_moves))
            score += (35 - legal_moves) * 0.5  # Fewer moves = better fortress
        
        return min(score, 100.0)


class StrategyPlayer:
    """
    Wrapper around HybridChessPlayer that uses strategies to guide move selection.
    Can play different strategies against each other.
    """
    
    def __init__(self, base_player, strategy: ChessStrategy):
        """
        Initialize strategy player.
        
        Args:
            base_player: HybridChessPlayer instance
            strategy: ChessStrategy to use
        """
        self.base_player = base_player
        self.strategy = strategy
        self.evaluator = StrategyEvaluator()
    
    def select_move_with_strategy(self, board: chess.Board, 
                                 temperature: float = 0.5) -> chess.Move:
        """
        Select move using strategy guidance.
        Temperature: 0 = always best move, 1 = more variety
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Rank moves by strategy
        ranked_moves = self.evaluator.rank_moves_by_strategy(
            board, self.strategy, legal_moves
        )
        
        # Use temperature to select (explore vs exploit)
        if random.random() < temperature:
            # Explore: random from top moves
            return random.choice(ranked_moves[:max(3, len(ranked_moves) // 3)])
        else:
            # Exploit: use best move from base player with strategy bias
            uci_move = self.base_player.select_move(board)
            return chess.Move.from_uci(uci_move) if uci_move else None
    
    def play_game_with_strategy(self, board: chess.Board = None, 
                               max_moves: int = 300) -> Tuple[str, int]:
        """
        Play a game using this strategy.
        Returns: (result, move_count)
        """
        if board is None:
            board = chess.Board()
        
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves:
            move = self.select_move_with_strategy(board)
            if move is None:
                break
            board.push(move)
            move_count += 1
        
        return board.result(), move_count
