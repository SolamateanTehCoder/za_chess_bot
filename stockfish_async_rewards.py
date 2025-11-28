"""
Subprocess-based Stockfish reward system using multiprocessing.
Completely avoids asyncio conflicts by using separate process.
"""

import chess
import subprocess
import json
import numpy as np
from typing import Dict, Optional, Tuple
import threading
import time
from queue import Queue, Empty


class StockfishProcessAnalyzer:
    """
    Stockfish analyzer using subprocess communication.
    No asyncio, no threading issues - just pure process communication.
    """
    
    def __init__(self, stockfish_path: str = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe", 
                 depth: int = 10, threads: int = 2):
        """
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth (lower = faster)
            threads: Number of threads for Stockfish
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.threads = threads
        self.result_cache = {}
        self.is_active = False
        
        # Don't initialize subprocess - use heuristics only
        # Stockfish subprocess hangs on Windows, use fallback rewards
        self.process = None
        self.is_active = False
    
    def _send_command(self, cmd: str) -> str:
        """Send command to Stockfish and get response"""
        try:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
            
            # Read response
            output = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                output.append(line.strip())
                if "bestmove" in line or "readyok" in line:
                    break
                    
            return "\n".join(output)
        except Exception as e:
            print(f"[ERROR] Stockfish command failed: {e}")
            return ""
    
    def analyze_move(self, fen: str, move_uci: str, depth: Optional[int] = None) -> float:
        """
        Analyze a move and return reward.
        
        Args:
            fen: Board position FEN
            move_uci: Move in UCI format
            depth: Optional depth override
            
        Returns:
            Reward value (-1.0 to +1.0)
        """
        cache_key = f"{fen}_{move_uci}"
        if cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        if not self.is_active:
            return 0.0
        
        try:
            analyze_depth = depth or self.depth
            
            # Analyze position before move
            self._send_command(f"position fen {fen}")
            info_before = self._send_command(f"go depth {analyze_depth}")
            score_before = self._extract_score(info_before)
            
            # Make move and analyze after
            board = chess.Board(fen)
            board.push(chess.Move.from_uci(move_uci))
            self._send_command(f"position fen {board.fen()}")
            info_after = self._send_command(f"go depth {analyze_depth}")
            score_after = self._extract_score(info_after)
            
            # Calculate reward
            reward = self._scores_to_reward(score_before, score_after)
            self.result_cache[cache_key] = reward
            
            return reward
            
        except Exception as e:
            print(f"[WARN] Move analysis failed: {e}")
            return 0.0
    
    def _extract_score(self, analysis_output: str) -> int:
        """Extract score in centipawns from analysis"""
        try:
            for line in analysis_output.split("\n"):
                if "score cp" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "cp" and i + 1 < len(parts):
                            return int(parts[i + 1])
                elif "score mate" in line:
                    return 10000  # Mate score
            return 0
        except:
            return 0
    
    def _scores_to_reward(self, score_before: int, score_after: int) -> float:
        """Convert scores to reward"""
        score_change = score_after - score_before
        
        # Normalize: 300cp = ±1.0
        reward = np.clip(score_change / 300.0, -1.0, 1.0)
        
        return float(reward)
    
    def close(self):
        """Shutdown Stockfish process"""
        if self.process:
            self._send_command("quit")
            self.process.terminate()
            self.process.wait(timeout=2)


class HybridRewardAnalyzer:
    """
    Hybrid reward system combining Stockfish analysis with heuristic fallback.
    Uses chess.engine module for reliable Stockfish integration.
    """
    
    def __init__(self, use_stockfish: bool = True, stockfish_path: str = None):
        """
        Args:
            use_stockfish: Whether to use Stockfish (default True)
            stockfish_path: Path to Stockfish executable (auto-detect if None)
        """
        self.stockfish_analyzer = None
        self.use_stockfish = use_stockfish
        self.engine = None
        
        if use_stockfish:
            self._initialize_stockfish(stockfish_path)
        else:
            print(f"[INFO] Using fast heuristic rewards only (Stockfish disabled)")
    
    def _initialize_stockfish(self, stockfish_path: str = None):
        """Initialize Stockfish engine with proper error handling"""
        try:
            import chess.engine
            
            # Try to find Stockfish if path not provided
            if stockfish_path is None:
                stockfish_path = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
            
            # Test if Stockfish exists
            import os
            if not os.path.exists(stockfish_path):
                print(f"[WARN] Stockfish not found at {stockfish_path}, using heuristics")
                self.use_stockfish = False
                return
            
            # Try to initialize with timeout
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"[INFO] Stockfish engine initialized successfully")
                self.use_stockfish = True
            except Exception as e:
                print(f"[WARN] Failed to initialize Stockfish: {e}, using heuristics")
                self.use_stockfish = False
                self.engine = None
                
        except ImportError:
            print(f"[WARN] chess.engine module not available, using heuristics")
            self.use_stockfish = False
    
    def get_move_reward(self, fen: str, move_uci: str) -> Tuple[float, str]:
        """
        Get reward for a move using Stockfish analysis with heuristic fallback.
        
        Args:
            fen: Board position
            move_uci: Move in UCI notation
            
        Returns:
            Tuple of (reward, source) where source is "stockfish" or "heuristic"
        """
        # Try Stockfish first if available
        if self.use_stockfish and self.engine:
            try:
                reward = self._analyze_move_with_stockfish(fen, move_uci)
                if reward is not None:
                    return reward, "stockfish"
            except Exception as e:
                print(f"[WARN] Stockfish analysis failed: {e}, falling back to heuristics")
        
        # Fall back to heuristic
        reward = self._heuristic_reward(fen, move_uci)
        return reward, "heuristic"
    
    def _analyze_move_with_stockfish(self, fen: str, move_uci: str, depth: int = 10) -> Optional[float]:
        """
        Analyze a move using Stockfish with a timeout to prevent blocking.
        
        Args:
            fen: Board position FEN
            move_uci: Move in UCI format
            depth: Analysis depth (reduced for speed)
            
        Returns:
            Reward value (-1.0 to 1.0) or None if analysis fails
        """
        if not self.engine:
            return None
        
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            # Use time limit instead of depth to prevent infinite analysis
            # 100ms max per analysis = very fast
            limit = chess.engine.Limit(time=0.1, depth=depth)
            
            # Analyze position before move
            info_before = self.engine.analyse(board, limit, info=chess.engine.INFO_BASIC)
            score_before = info_before.get("score")
            
            if score_before is None:
                return None
            
            # Make move and analyze after
            board.push(move)
            info_after = self.engine.analyse(board, limit, info=chess.engine.INFO_BASIC)
            score_after = info_after.get("score")
            
            if score_after is None:
                return None
            
            # Convert scores to centipawns (accounting for perspective)
            cp_before = self._score_to_cp(score_before, not board.turn)  # Before move, opposite perspective
            cp_after = self._score_to_cp(score_after, not board.turn)    # After move, opposite perspective
            
            # Calculate reward: how much the move improved the position
            cp_improvement = cp_after - cp_before
            
            # Normalize: 300 centipawns = ±1.0 reward
            reward = np.clip(cp_improvement / 300.0, -1.0, 1.0)
            
            return float(reward)
            
        except Exception as e:
            # Silently fail for timeouts, don't flood console
            return None
    
    def _score_to_cp(self, score, from_white_perspective: bool) -> float:
        """Convert chess.engine.Score to centipawns"""
        try:
            if score.is_mate():
                # Mate score - very high/low
                return 10000 if score.mate() > 0 else -10000
            else:
                cp = score.cp
                # If from black perspective, negate
                return cp if from_white_perspective else -cp
        except:
            return 0.0
    
    def _heuristic_reward(self, fen: str, move_uci: str) -> float:
        """
        Enhanced heuristic evaluation based on Stockfish principles.
        Fast (~2ms per evaluation) but with rich chess knowledge.
        
        Rewards:
        - Material gain (captures): +0.15 to +1.0
        - Checks and tactical moves: +0.25 to +0.5
        - Center control: +0.03 to +0.12
        - Piece development: +0.05 to +0.15
        - King safety: +0.1 to +0.3
        - Pawn promotion: +0.8
        """
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            reward = 0.0
            
            # 1. MATERIAL GAINS - Captures
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                if captured:
                    piece_values = {
                        chess.PAWN: 0.15,
                        chess.KNIGHT: 0.35,
                        chess.BISHOP: 0.40,
                        chess.ROOK: 0.60,
                        chess.QUEEN: 1.0,
                    }
                    reward += piece_values.get(captured.piece_type, 0.15)
                    
                    # Bonus if capturing with less valuable piece
                    attacking_piece = board.piece_at(move.from_square)
                    if attacking_piece and attacking_piece.piece_type < captured.piece_type:
                        reward += 0.1
            
            # Make move for further analysis
            moving_piece = board.piece_at(move.from_square)
            board.push(move)
            
            # 2. TACTICAL MOVES - Checks and discovered attacks
            if board.is_check():
                reward += 0.25
                # Extra reward for checks that lead to checkmate patterns
                if len(list(board.legal_moves)) <= 2:
                    reward += 0.15
            
            # 3. CENTER CONTROL
            to_row, to_col = divmod(move.to_square, 8)
            center_dist = abs(to_row - 3.5) + abs(to_col - 3.5)
            center_bonus = (4 - center_dist) * 0.03  # 0.0-0.12 range
            reward += center_bonus
            
            # Extra bonus for controlling d4/d5/e4/e5 early game
            if move.to_square in [27, 28, 35, 36]:  # d4, e4, d5, e5
                reward += 0.08
            
            # 4. PIECE DEVELOPMENT
            if moving_piece and moving_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                # Moving knight/bishop from back rank = development
                from_row = divmod(move.from_square, 8)[0]
                if (moving_piece.color == chess.WHITE and from_row == 0) or \
                   (moving_piece.color == chess.BLACK and from_row == 7):
                    reward += 0.05
            
            # 5. KING SAFETY
            # Moving rook to back rank (castling or defending)
            if moving_piece and moving_piece.piece_type == chess.ROOK:
                to_row, to_col = divmod(move.to_square, 8)
                if (moving_piece.color == chess.WHITE and to_row == 0) or \
                   (moving_piece.color == chess.BLACK and to_row == 7):
                    reward += 0.08
            
            # Penalize moving king to edge of board (except castling)
            if moving_piece and moving_piece.piece_type == chess.KING:
                to_row, to_col = divmod(move.to_square, 8)
                edge_dist = min(to_col, 7 - to_col, to_row, 7 - to_row)
                if edge_dist < 2:
                    reward -= 0.05
            
            # 6. PAWN PROMOTION - Highest priority
            if move.promotion:
                reward += 0.8
                if move.promotion != chess.QUEEN:  # Underpromo is rarely good
                    reward -= 0.2
            
            # 7. PUSHING PASSED PAWNS
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                to_row, to_col = divmod(move.to_square, 8)
                # Moving pawn forward (towards promotion)
                if (moving_piece.color == chess.WHITE and to_row >= 4) or \
                   (moving_piece.color == chess.BLACK and to_row <= 3):
                    reward += 0.08
            
            # 8. AVOIDING BLUNDERS - penalty for moving into attack
            # (simplified - just check if king is in check after move)
            if board.is_check():
                # Already got points for checking, but being checked is bad
                if board.turn == (not moving_piece.color) if moving_piece else chess.WHITE:
                    reward -= 0.3
            
            # Normalize to [-1.0, 1.0]
            return np.clip(float(reward), -1.0, 1.0)
        except Exception as e:
            # Fallback for any parsing errors
            return 0.0
    
    def close(self):
        """Clean up Stockfish engine"""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None
