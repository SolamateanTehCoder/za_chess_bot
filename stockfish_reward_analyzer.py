"""
Stockfish-based reward analyzer for self-play training.
Analyzes each move and assigns accuracy-based rewards and time-based penalties.
"""

import chess
import chess.engine
import torch
import numpy as np
from typing import Dict, Tuple, Optional
import os
import time
import subprocess
import sys


class StockfishRewardAnalyzer:
    """
    Uses Stockfish to analyze moves and provide accuracy-based rewards.
    Positive reward = good move (high accuracy)
    Negative reward = bad move (low accuracy)
    Additional time penalty for exceeding 1 second per move.
    """
    
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 20, timeout_ms: int = 500):
        """
        Initialize Stockfish analyzer.
        
        Args:
            stockfish_path: Path to Stockfish executable. If None, tries to find it automatically.
            depth: Depth for Stockfish analysis
            timeout_ms: Millisecond timeout for analysis per move
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.timeout_ms = timeout_ms
        self.engine = None
        self.is_active = False
        
        # Reward parameters
        self.max_reward = 1.0  # Reward for perfect move
        self.min_penalty = -1.0  # Penalty for blunder
        self.time_penalty_per_ms = -0.001  # Pain for each millisecond over 1 second
        self.time_limit_ms = 1000  # 1 second baseline
        
        # Try to initialize engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize Stockfish engine."""
        try:
            # Try to find Stockfish if path not provided
            if self.stockfish_path is None:
                possible_paths = [
                    "stockfish",  # Linux/Mac
                    "stockfish.exe",  # Windows
                    "C:\\Program Files\\Stockfish\\stockfish.exe",  # Windows common location
                    "C:\\Program Files (x86)\\Stockfish\\stockfish.exe",  # Windows 32-bit
                    "/usr/bin/stockfish",  # Linux
                    "/opt/homebrew/bin/stockfish",  # Mac ARM
                ]
                
                # Try environment variable
                if 'STOCKFISH_PATH' in os.environ:
                    possible_paths.insert(0, os.environ['STOCKFISH_PATH'])
                
                for path in possible_paths:
                    if os.path.exists(path) or self._command_exists(path):
                        self.stockfish_path = path
                        break
            
            if self.stockfish_path is None:
                print("[WARNING] Stockfish not found. Will use fallback heuristic rewards.")
                self.is_active = False
                return
            
            # Test if Stockfish works
            try:
                result = subprocess.run([self.stockfish_path, "--version"], 
                                       capture_output=True, 
                                       timeout=5,
                                       text=True)
                if result.returncode == 0:
                    print(f"[SUCCESS] Stockfish found: {result.stdout.strip()}")
                    self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                    self.is_active = True
                    print("[INFO] Stockfish reward analyzer initialized and ready")
                else:
                    raise Exception("Stockfish version check failed")
            except Exception as e:
                print(f"[WARNING] Could not initialize Stockfish: {e}")
                print("[INFO] Will use fallback heuristic rewards instead")
                self.is_active = False
        
        except Exception as e:
            print(f"[WARNING] Stockfish initialization error: {e}")
            self.is_active = False
    
    @staticmethod
    def _command_exists(command):
        """Check if a command exists in PATH."""
        try:
            subprocess.run([command, "--version"], 
                          capture_output=True, 
                          timeout=2)
            return True
        except:
            return False
    
    def analyze_move(self, board: chess.Board, move: chess.Move, move_time_ms: int) -> Dict:
        """
        Analyze a move and return reward/penalty and accuracy score.
        
        Args:
            board: Chess board position BEFORE the move
            move: The move to analyze
            move_time_ms: Time taken to make this move in milliseconds
            
        Returns:
            Dictionary with:
                - 'accuracy': 0-100 (100 = perfect move)
                - 'reward': positive (good) or negative (bad)
                - 'eval_before': Stockfish eval before move (from analyzing side's perspective)
                - 'eval_after': Stockfish eval after move (from opponent's perspective)
                - 'best_move': What Stockfish would play
                - 'is_best_move': Whether player played the best move
                - 'time_penalty': Penalty for using too much time
        """
        result = {
            'accuracy': 100.0,
            'reward': 0.0,
            'eval_before': None,
            'eval_after': None,
            'best_move': None,
            'is_best_move': False,
            'time_penalty': 0.0,
            'move': move.uci()
        }
        
        # Calculate time penalty (pain for exceeding 1 second)
        if move_time_ms > self.time_limit_ms:
            excess_ms = move_time_ms - self.time_limit_ms
            result['time_penalty'] = excess_ms * self.time_penalty_per_ms
        
        if not self.is_active:
            # Fallback heuristic if Stockfish unavailable
            return self._heuristic_reward(board, move, move_time_ms)
        
        try:
            # Analyze position before move
            info_before = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            eval_before = info_before.get("score")
            
            # Get best move from Stockfish
            best_move_info = self.engine.play(board, chess.engine.Limit(depth=self.depth, time=self.timeout_ms/1000))
            best_move = best_move_info.move
            result['best_move'] = best_move.uci() if best_move else None
            
            # Make the move
            board_copy = board.copy()
            board_copy.push(move)
            
            # Analyze position after move
            info_after = self.engine.analyse(board_copy, chess.engine.Limit(depth=self.depth))
            eval_after = info_after.get("score")
            
            # Convert evaluations to centipawn scores
            eval_before_cp = self._eval_to_cp(eval_before)
            eval_after_cp = self._eval_to_cp(eval_after)
            
            result['eval_before'] = eval_before_cp
            result['eval_after'] = eval_after_cp
            
            # Check if this was the best move
            result['is_best_move'] = (move == best_move)
            
            # Calculate accuracy based on evaluation change
            # We measure from the perspective of the player moving
            # Positive change in eval = good move
            side_multiplier = 1 if board.turn == chess.WHITE else -1
            eval_change = (eval_after_cp - eval_before_cp) * side_multiplier
            
            # Accuracy: 100 = best move, 0 = catastrophic blunder
            if eval_change >= 50:  # Excellent move
                result['accuracy'] = 100.0
                result['reward'] = 1.0
            elif eval_change >= 20:  # Good move
                result['accuracy'] = 85.0
                result['reward'] = 0.7
            elif eval_change >= 0:  # Okay move
                result['accuracy'] = 60.0
                result['reward'] = 0.3
            elif eval_change >= -50:  # Slight mistake
                result['accuracy'] = 40.0
                result['reward'] = -0.3
            elif eval_change >= -200:  # Mistake
                result['accuracy'] = 20.0
                result['reward'] = -0.7
            else:  # Blunder
                result['accuracy'] = 0.0
                result['reward'] = -1.0
            
            # Add time penalty to reward
            result['reward'] += result['time_penalty']
            
            # Clamp reward to reasonable range
            result['reward'] = max(-1.0, min(1.0, result['reward']))
            
        except Exception as e:
            print(f"[WARNING] Error analyzing move {move.uci()}: {e}")
            # Fall back to heuristic
            return self._heuristic_reward(board, move, move_time_ms)
        
        return result
    
    def _eval_to_cp(self, eval_score) -> int:
        """Convert chess.engine.Evaluation to centipawns."""
        if eval_score is None:
            return 0
        
        if eval_score.is_mate():
            # Mate in X moves
            mate_moves = eval_score.mate()
            # Return a large value, scaled by proximity to mate
            return 10000 if mate_moves > 0 else -10000
        
        return eval_score.cp if hasattr(eval_score, 'cp') else 0
    
    def _heuristic_reward(self, board: chess.Board, move: chess.Move, move_time_ms: int) -> Dict:
        """
        Fallback reward calculation using heuristics when Stockfish is unavailable.
        """
        result = {
            'accuracy': 50.0,  # Unknown accuracy without engine
            'reward': 0.0,
            'eval_before': None,
            'eval_after': None,
            'best_move': None,
            'is_best_move': False,
            'time_penalty': 0.0,
            'move': move.uci()
        }
        
        # Time penalty
        if move_time_ms > self.time_limit_ms:
            excess_ms = move_time_ms - self.time_limit_ms
            result['time_penalty'] = excess_ms * self.time_penalty_per_ms
        
        # Heuristic: penalize moving into check, capture value, etc.
        board_copy = board.copy()
        board_copy.push(move)
        
        # Check if move creates threats or loses material
        if board_copy.is_check():
            result['accuracy'] = 70.0
            result['reward'] = 0.5
        elif board_copy.is_checkmate():
            result['accuracy'] = 100.0
            result['reward'] = 1.0
        else:
            # Check for capture value
            if move.promotion:
                result['accuracy'] = 75.0
                result['reward'] = 0.6
            elif board.is_capture(move):
                result['accuracy'] = 65.0
                result['reward'] = 0.4
            else:
                result['accuracy'] = 50.0
                result['reward'] = 0.1
        
        # Add time penalty
        result['reward'] += result['time_penalty']
        result['reward'] = max(-1.0, min(1.0, result['reward']))
        
        return result
    
    def close(self):
        """Close the Stockfish engine."""
        if self.engine:
            try:
                self.engine.quit()
                self.is_active = False
            except:
                pass
    
    def __del__(self):
        """Ensure engine is closed on deletion."""
        self.close()
