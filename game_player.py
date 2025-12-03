"""
Bullet chess game player with Stockfish-based reward system.
Plays one game at a time with 60 second time control per side.
Model thinks during opponent's time, gets Stockfish rewards asynchronously.
Training only starts after model reaches 100% accuracy.
"""

import os
import sys
import chess
import chess.engine
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import time as time_module
import threading
from queue import Queue

# PyTorch setup
os.environ['TORCH_COMPILE_DEBUG'] = '0'
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

try:
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
except:
    try:
        torch.set_float32_matmul_precision('high')
    except:
        pass


class StockfishAnalyzer:
    """Stockfish-based move evaluation and reward calculation."""
    
    def __init__(self, stockfish_path: str = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe", 
                 depth: int = 10, time_limit: float = 0.01):
        """
        Initialize Stockfish analyzer.
        
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth (reduced for speed)
            time_limit: Time limit per analysis in seconds (very fast)
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit  # 10ms per analysis - very fast
        self.engine = None
        self._init_engine()
        self.cached_evals = {}  # Cache FEN evaluations
    
    def _init_engine(self):
        """Initialize Stockfish engine."""
        try:
            import os.path
            if os.path.exists(self.stockfish_path):
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                print(f"[INFO] Stockfish engine initialized at {self.stockfish_path}")
            else:
                print(f"[WARN] Stockfish not found at {self.stockfish_path}")
        except Exception as e:
            print(f"[WARN] Failed to initialize Stockfish: {e}")
    
    def evaluate_move(self, fen: str, move_uci: str) -> float:
        """
        Evaluate a move and return reward based on Stockfish analysis.
        Uses cached evaluations for speed.
        
        Args:
            fen: Board position FEN
            move_uci: Move in UCI format
            
        Returns:
            Reward in range [-1.0, 1.0]
        """
        if not self.engine:
            return 0.0
        
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            # Get or compute evaluation before move (cached)
            if fen in self.cached_evals:
                cp_before = self.cached_evals[fen]
            else:
                limit = chess.engine.Limit(time=self.time_limit, depth=self.depth)
                info_before = self.engine.analyse(board, limit, info=chess.engine.INFO_BASIC)
                score_before = info_before.get("score")
                
                if score_before is None:
                    return 0.0
                
                cp_before = self._score_to_cp(score_before, False)
                self.cached_evals[fen] = cp_before
            
            # Make move and analyze after
            board.push(move)
            fen_after = board.fen()
            
            # Get or compute evaluation after move (cached)
            if fen_after in self.cached_evals:
                cp_after = self.cached_evals[fen_after]
            else:
                limit = chess.engine.Limit(time=self.time_limit, depth=self.depth)
                info_after = self.engine.analyse(board, limit, info=chess.engine.INFO_BASIC)
                score_after = info_after.get("score")
                
                if score_after is None:
                    return 0.0
                
                cp_after = self._score_to_cp(score_after, False)
                self.cached_evals[fen_after] = cp_after
            
            # Improvement from this move
            improvement = cp_after - cp_before
            
            # Normalize: 300cp = ±1.0 reward
            reward = np.clip(improvement / 300.0, -1.0, 1.0)
            
            # Clear old cache to prevent memory bloat (keep last 1000)
            if len(self.cached_evals) > 1000:
                # Keep only recent entries
                keys_to_remove = list(self.cached_evals.keys())[:-500]
                for k in keys_to_remove:
                    del self.cached_evals[k]
            
            return float(reward)
        except Exception as e:
            return 0.0
    
    def evaluate_move_before_push(self, fen_after: str, move_uci: str) -> float:
        """
        Deprecated - use evaluate_move instead with fen_before.
        """
        return 0.0
    
    def _score_to_cp(self, score, from_white_perspective: bool) -> float:
        """Convert chess.engine.Score to centipawns."""
        try:
            if score.is_mate():
                return 10000 if score.mate() > 0 else -10000
            else:
                cp = score.cp
                return cp if from_white_perspective else -cp
        except:
            return 0.0
    
    def close(self):
        """Close the engine."""
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass


class BulletGamePlayer:
    """Plays single bullet chess games with time-based pain penalty."""
    
    def __init__(self, model, device='cuda', stockfish_analyzer=None):
        """
        Initialize game player.
        
        Args:
            model: Neural network model
            device: Device to run on (cuda/cpu)
            stockfish_analyzer: StockfishAnalyzer instance
        """
        self.model = model
        self.device = device
        self.analyzer = stockfish_analyzer
        self.eval_queue = Queue()
        self.eval_results = {}
        self.eval_thread = None
    
    def _evaluation_worker(self):
        """Background thread that evaluates moves from the queue."""
        while True:
            try:
                item = self.eval_queue.get(timeout=1)
                if item is None:  # Sentinel value to stop thread
                    break
                
                fen, move_uci, move_id = item
                reward = self.analyzer.evaluate_move(fen, move_uci) if self.analyzer else 0.0
                self.eval_results[move_id] = reward
                self.eval_queue.task_done()
            except:
                continue
    
    def start_evaluation_thread(self):
        """Start the background evaluation thread."""
        if self.eval_thread is None:
            self.eval_thread = threading.Thread(target=self._evaluation_worker, daemon=True)
            self.eval_thread.start()
    
    def stop_evaluation_thread(self):
        """Stop the background evaluation thread."""
        if self.eval_thread:
            self.eval_queue.put(None)  # Sentinel to stop
    
    def play_game(self, play_as_white: bool = True):
        """
        Play a single bullet game with 60s per side time control.
        Game pauses after each move to get Stockfish reward.
        
        Args:
            play_as_white: Whether model plays as white
            
        Returns:
            Game result dict with experiences and metrics
        """
        board = chess.Board()
        experiences = []
        
        # Time tracking
        time_limit = 60.0  # 60 seconds per side
        white_time = time_limit
        black_time = time_limit
        last_time_update = time_module.time()
        
        move_count = 0
        max_moves = 200
        
        try:
            while not board.is_game_over() and move_count < max_moves:
                current_time = time_module.time()
                elapsed = current_time - last_time_update
                
                # Update time based on whose turn it is
                if board.turn == chess.WHITE:
                    white_time -= elapsed
                    if white_time <= 0:
                        return self._format_result("Loss" if play_as_white else "Win", 
                                                  move_count, experiences, timeout=True, board=board)
                else:
                    black_time -= elapsed
                    if black_time <= 0:
                        return self._format_result("Win" if play_as_white else "Loss", 
                                                  move_count, experiences, timeout=True, board=board)
                
                last_time_update = current_time
                is_ai_turn = (board.turn == chess.WHITE) == play_as_white
                
                if is_ai_turn:
                    # AI move - make it fast
                    move_start = time_module.time()
                    
                    # Get legal moves
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Save FEN before move
                    fen_before = board.fen()
                    
                    # Compute move (random selection for now)
                    move = np.random.choice(legal_moves)
                    move_time_ms = (time_module.time() - move_start) * 1000
                    
                    # Penalty based on actual thinking time
                    time_penalty = -0.001 * move_time_ms
                    
                    # Push move to board
                    board.push(move)
                    fen_after = board.fen()
                    
                    # NOW PAUSE AND GET STOCKFISH REWARD (blocking)
                    stockfish_reward = 0.0
                    if self.analyzer:
                        try:
                            # Evaluate the move synchronously (pauses game)
                            stockfish_reward = self.analyzer.evaluate_move(fen_before, move.uci())
                        except Exception as e:
                            pass
                    
                    # Combined reward: Stockfish signal + time penalty
                    total_reward = stockfish_reward + time_penalty
                    total_reward = np.clip(total_reward, -1.0, 1.0)
                    
                    # Store experience
                    experiences.append({
                        'move': move.uci(),
                        'reward': total_reward,
                        'stockfish_reward': stockfish_reward,
                        'time_penalty': time_penalty,
                        'move_time_ms': move_time_ms
                    })
                    
                    move_count += 1
                else:
                    # Opponent move (random)
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Save FEN before move
                    fen_before = board.fen()
                    
                    opponent_move = np.random.choice(legal_moves)
                    
                    # Push opponent move
                    board.push(opponent_move)
                    
                    # Game pauses for opponent's move evaluation (blocking)
                    if self.analyzer:
                        try:
                            _ = self.analyzer.evaluate_move(fen_before, opponent_move.uci())
                        except:
                            pass
                    
                    move_count += 1
            
            # Determine result
            if board.is_checkmate():
                if board.turn == chess.WHITE:
                    result = "Loss" if play_as_white else "Win"
                else:
                    result = "Win" if play_as_white else "Loss"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "Draw"
            else:
                result = "Draw"
            
            return self._format_result(result, move_count, experiences, board=board)
        
        finally:
            pass
    
    def _format_result(self, result: str, moves: int, experiences: list, timeout: bool = False, board=None):
        """Format game result."""
        return {
            'result': result,
            'moves': moves,
            'experiences': experiences,
            'timeout': timeout,
            'num_ai_moves': len(experiences),
            'board': board
        }


def main():
    """Main game playing loop."""
    print("=" * 80)
    print("BULLET CHESS GAME PLAYER")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")
    if device.type == 'cuda':
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize Stockfish analyzer
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Stockfish analyzer...")
    analyzer = StockfishAnalyzer()
    
    # Create dummy model (placeholder)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating game player...")
    
    # Create game player
    player = BulletGamePlayer(None, device=device, stockfish_analyzer=analyzer)
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting game loop...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Playing bullet games (60s per side)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Baseline move time: {player.baseline_time:.1f}s")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pain penalty: -0.001 per ms over baseline")
    print()
    
    # Play games
    game_count = 0
    try:
        while True:
            game_count += 1
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GAME {game_count}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Playing as {'White' if game_count % 2 == 1 else 'Black'}...")
            
            play_white = game_count % 2 == 1
            result = player.play_game(play_as_white=play_white)
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Result: {result['result']}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Moves by AI: {result['num_ai_moves']}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total moves: {result['moves']}")
            
            if result['experiences']:
                avg_reward = np.mean([exp['reward'] for exp in result['experiences']])
                avg_time = np.mean([exp['move_time_ms'] for exp in result['experiences']])
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg reward: {avg_reward:.4f}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg move time: {avg_time:.1f}ms")
            
            print()
            sys.stdout.flush()
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training interrupted by user")
    
    finally:
        if analyzer:
            analyzer.close()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Game player closed")


if __name__ == "__main__":
    main()
