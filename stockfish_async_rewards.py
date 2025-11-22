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
        
        try:
            # Start Stockfish process
            self.process = subprocess.Popen(
                [stockfish_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            
            # Configure Stockfish
            self._send_command(f"setoption name Threads value {threads}")
            self._send_command("isready")
            
            # Verify it's working
            self._send_command("position startpos")
            info = self._send_command("go depth 1")
            
            if info:
                print(f"[SUCCESS] Stockfish process initialized")
                self.is_active = True
            else:
                self.process.terminate()
                self.is_active = False
                
        except Exception as e:
            print(f"[WARN] Stockfish process failed: {e}")
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
    Combines Stockfish analysis with fast heuristics.
    Uses Stockfish when available, falls back to heuristics for speed.
    """
    
    def __init__(self, use_stockfish: bool = True):
        self.stockfish_analyzer = None
        self.use_stockfish = use_stockfish
        
        if use_stockfish:
            try:
                self.stockfish_analyzer = StockfishProcessAnalyzer(depth=10)
            except Exception as e:
                print(f"[WARN] Stockfish disabled: {e}")
                self.use_stockfish = False
    
    def get_move_reward(self, fen: str, move_uci: str) -> Tuple[float, str]:
        """
        Get reward for a move using best available method.
        
        Args:
            fen: Board position
            move_uci: Move in UCI notation
            
        Returns:
            Tuple of (reward, source) where source is "stockfish" or "heuristic"
        """
        # Try Stockfish first
        if self.use_stockfish and self.stockfish_analyzer and self.stockfish_analyzer.is_active:
            try:
                reward = self.stockfish_analyzer.analyze_move(fen, move_uci)
                if reward != 0.0:  # Got a real analysis
                    return reward, "stockfish"
            except Exception as e:
                print(f"[WARN] Stockfish analysis failed: {e}")
        
        # Fall back to heuristic
        reward = self._heuristic_reward(fen, move_uci)
        return reward, "heuristic"
    
    def _heuristic_reward(self, fen: str, move_uci: str) -> float:
        """Fast heuristic evaluation"""
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            reward = 0.0
            
            # Captures
            if board.is_capture(move):
                captured = board.piece_at(move.to_square)
                piece_values = {
                    chess.PAWN: 0.1,
                    chess.KNIGHT: 0.3,
                    chess.BISHOP: 0.35,
                    chess.ROOK: 0.5,
                    chess.QUEEN: 0.9,
                }
                reward += piece_values.get(captured.piece_type, 0.1)
            
            # Checks
            board.push(move)
            if board.is_check():
                reward += 0.2
            
            # Center control
            to_row, to_col = divmod(move.to_square, 8)
            center_dist = abs(to_row - 3.5) + abs(to_col - 3.5)
            center_bonus = (4 - center_dist) * 0.02
            reward += center_bonus
            
            return np.clip(float(reward), -1.0, 1.0)
        except:
            return 0.0
    
    def close(self):
        """Cleanup"""
        if self.stockfish_analyzer:
            self.stockfish_analyzer.close()
