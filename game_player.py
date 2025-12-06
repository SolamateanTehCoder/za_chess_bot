"""
Bullet chess game player with Stockfish-based reward system.
Plays one game at a time with 60 second time control per side.
Uses neural network policy to select moves, gets Stockfish rewards.
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


def encode_board_state(board: chess.Board) -> np.ndarray:
    """
    Encode chess board state to 768-dimensional vector (8x8x12 planes).
    12 planes: 6 white pieces + 6 black pieces
    
    Args:
        board: Chess board position
        
    Returns:
        768-dimensional numpy array
    """
    piece_map = {
        chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
        chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Create 12 planes (6 white + 6 black)
    board_state = np.zeros((12, 8, 8), dtype=np.float32)
    
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_type = piece_map.get(piece.piece_type, 0)
        plane = color_offset + piece_type
        board_state[plane, rank, file] = 1.0
    
    # Flatten to 768D vector
    return board_state.reshape(-1)


def get_move_indices(board: chess.Board) -> dict:
    """
    Map legal moves to indices in the policy output (4672 moves).
    Uses standard chess move encoding: from_square (64) * to_square (64) + other_moves
    
    Args:
        board: Chess board position
        
    Returns:
        Dict mapping move UCI strings to policy indices
    """
    move_to_index = {}
    
    for move in board.legal_moves:
        from_sq = move.from_square  # 0-63
        to_sq = move.to_square      # 0-63
        
        # Simple encoding: from_square * 64 + to_square
        # This covers basic queen-like moves (64*64 = 4096 possible moves)
        idx = from_sq * 64 + to_sq
        move_to_index[move.uci()] = idx
    
    return move_to_index

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
                 depth: int = 20, time_limit: float = 0.5):
        """
        Initialize Stockfish analyzer.
        
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth (20 for stronger evaluation)
            time_limit: Time limit per analysis in seconds (0.5s = 500ms for quality)
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit  # 500ms per analysis - gives Stockfish proper time
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
                    # AI move - use neural network policy
                    move_start = time_module.time()
                    
                    # Get legal moves
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    
                    # Save FEN before move
                    fen_before = board.fen()
                    
                    # Get move from model or random if no model
                    if self.model is not None:
                        try:
                            # Encode board state
                            board_state = encode_board_state(board)
                            board_tensor = torch.from_numpy(board_state).float().unsqueeze(0).to(self.device)
                            
                            # Get policy from model
                            with torch.no_grad():
                                policy, _ = self.model(board_tensor)
                                policy = policy.cpu().numpy()[0]  # Get first (only) batch
                            
                            # Get indices for legal moves
                            move_indices = get_move_indices(board)
                            legal_move_list = list(legal_moves)
                            
                            # Create mask for legal moves in policy output
                            legal_mask = np.full(4672, -np.inf, dtype=np.float32)
                            for move in legal_move_list:
                                idx = move_indices.get(move.uci(), -1)
                                if idx >= 0 and idx < 4672:
                                    legal_mask[idx] = policy[idx]
                            
                            # Select move with highest policy probability
                            best_idx = np.argmax(legal_mask)
                            
                            # Find which move this corresponds to
                            move = None
                            for m in legal_move_list:
                                idx = move_indices.get(m.uci(), -1)
                                if idx == best_idx:
                                    move = m
                                    break
                            
                            # Fallback to random if no match (shouldn't happen)
                            if move is None:
                                move = np.random.choice(legal_move_list)
                        
                        except Exception as e:
                            # Fallback to random on any error
                            print(f"[WARN] Model inference failed: {e}, using random move")
                            move = np.random.choice(legal_moves)
                    else:
                        # No model - use random moves
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
    """Main game playing loop with neural network move selection."""
    print("=" * 80)
    print("BULLET CHESS GAME PLAYER - NEURAL NETWORK POWERED")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using device: {device}")
    if device.type == 'cuda':
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GPU: {torch.cuda.get_device_name(0)}")
    
    # Import model
    from trainer import SimpleChessNet
    
    # Try to load checkpoint
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model...")
    model = SimpleChessNet().to(device)
    
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        game_checkpoints = list(checkpoint_dir.glob("model_checkpoint_game_*.pt"))
        if game_checkpoints:
            # Sort by game number numerically
            game_checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
            latest = game_checkpoints[-1]
            try:
                checkpoint = torch.load(latest, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded model from {latest.name}")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded checkpoint but using fresh model")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Could not load checkpoint: {e}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using fresh model")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize Stockfish analyzer with improved settings
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Stockfish analyzer...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Analysis depth: 20")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Time per position: 500ms")
    analyzer = StockfishAnalyzer()
    
    # Create game player with model
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating game player...")
    player = BulletGamePlayer(model, device=device, stockfish_analyzer=analyzer)
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting game loop...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Playing bullet games (60s per side)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using neural network for move selection")
    print()
    
    # Play games
    game_count = 0
    results = {'Win': 0, 'Loss': 0, 'Draw': 0}
    
    try:
        while True:
            game_count += 1
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] GAME {game_count}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Playing as {'White' if game_count % 2 == 1 else 'Black'}...")
            
            play_white = game_count % 2 == 1
            result = player.play_game(play_as_white=play_white)
            
            results[result['result']] += 1
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Result: {result['result']}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Moves by AI: {result['num_ai_moves']}")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total moves: {result['moves']}")
            
            if result['experiences']:
                avg_reward = np.mean([exp['reward'] for exp in result['experiences']])
                avg_stockfish = np.mean([exp['stockfish_reward'] for exp in result['experiences']])
                avg_time = np.mean([exp['move_time_ms'] for exp in result['experiences']])
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg reward: {avg_reward:.4f}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg Stockfish reward: {avg_stockfish:.4f}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Avg move time: {avg_time:.1f}ms")
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Record: {results['Win']}W {results['Loss']}L {results['Draw']}D")
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
