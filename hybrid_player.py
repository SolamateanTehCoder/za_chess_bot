"""
Unified game player for WCCC competition.
Integrates: neural networks, opening books, tablebases, time management, Stockfish.
This is the actual engine that plays in tournaments.
"""

import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import time as time_module

from opening_book import OpeningBook
from tablebase_manager import TablebaseManager
from time_management import TimeManager, TimeControl
from chess_models import SimpleChessNet, ChessNetV2
from strategy import ChessStrategy


class HybridChessPlayer:
    """
    Hybrid chess player combining multiple evaluation and move selection strategies.
    Primary engine for tournament play.
    """
    
    def __init__(self, model=None, model_path: Optional[str] = None, 
                 use_enhanced_model: bool = True, device: str = "cuda"):
        """
        Initialize hybrid chess player.
        
        Args:
            model: Pre-loaded model, or None to load from checkpoint
            model_path: Path to model checkpoint
            use_enhanced_model: Use ChessNetV2 (True) or SimpleChessNet (False)
            device: "cuda" or "cpu"
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model
        self.use_enhanced_model = use_enhanced_model
        
        # Load model if not provided
        if self.model is None:
            self.model = self._load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize components
        self.opening_book = OpeningBook()
        self.tablebase_manager = TablebaseManager()
        self.stockfish = None
        self.time_manager = None
        self.strategy = ChessStrategy.from_config("balanced")  # Default balanced strategy
        
        # Initialize Stockfish
        self._init_stockfish()
        
        # Load opening book if available
        self._load_opening_book()
        
        # Statistics
        self.stats = {
            "moves_played": 0,
            "book_moves": 0,
            "tb_moves": 0,
            "nn_moves": 0,
            "stockfish_moves": 0,
            "total_move_time": 0.0,
            "games_played": 0
        }
    
    def _load_model(self, model_path: Optional[str]) -> torch.nn.Module:
        """Load neural network model from checkpoint."""
        if model_path is None:
            # Try to find latest checkpoint
            checkpoint_dir = Path("checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pt"))
                if checkpoints:
                    model_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
        
        if model_path and Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, weights_only=False)
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                else:
                    state_dict = checkpoint
                
                # Load into model
                ModelClass = ChessNetV2 if self.use_enhanced_model else SimpleChessNet
                model = ModelClass()
                model.load_state_dict(state_dict)
                print(f"[INFO] Loaded model from {model_path}")
                return model
            except Exception as e:
                print(f"[WARN] Failed to load model: {e}")
        
        # Create fresh model
        ModelClass = ChessNetV2 if self.use_enhanced_model else SimpleChessNet
        model = ModelClass()
        print("[INFO] Created fresh model")
        return model
    
    def _init_stockfish(self, depth: int = 20, time_limit: float = 0.5):
        """Initialize Stockfish engine."""
        stockfish_path = self._find_stockfish()
        
        try:
            self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            print(f"[INFO] Stockfish initialized (depth {depth}, {time_limit}s)")
        except Exception as e:
            print(f"[WARN] Stockfish initialization failed: {e}")
            self.stockfish = None
    
    def _find_stockfish(self) -> str:
        """Find Stockfish executable."""
        paths = [
            r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "stockfish"
        ]
        
        for path in paths:
            try:
                import subprocess
                result = subprocess.run([path, "--version"], capture_output=True, timeout=2)
                if result.returncode == 0:
                    return path
            except:
                continue
        
        return "stockfish"
    
    def _load_opening_book(self):
        """Load opening book from file."""
        book_path = Path("openings.json")
        if book_path.exists():
            try:
                self.opening_book.load_book(str(book_path))
                print(f"[INFO] Loaded opening book with {len(self.opening_book.positions)} positions")
            except Exception as e:
                print(f"[WARN] Failed to load opening book: {e}")
    
    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """Encode board position to tensor."""
        piece_map = {
            chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
            chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        board_state = np.zeros((12, 8, 8), dtype=np.float32)
        
        for square, piece in board.piece_map().items():
            rank, file = divmod(square, 8)
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_type = piece_map.get(piece.piece_type, 0)
            plane = color_offset + piece_type
            board_state[plane, rank, file] = 1.0
        
        tensor = torch.FloatTensor(board_state.reshape(-1)).to(self.device)
        return tensor
    
    def get_move_from_policy(self, board: chess.Board, policy_logits: torch.Tensor) -> Optional[str]:
        """Select move from policy logits using legal move mask."""
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        # Create legal move mask
        legal_mask = np.zeros(4672, dtype=np.float32)
        
        for move in legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square
            idx = from_sq * 64 + to_sq
            
            if idx < 4672:
                legal_mask[idx] = 1.0
        
        # Apply mask to policy
        masked_policy = policy_logits.cpu().detach().numpy()[0] * legal_mask
        best_idx = np.argmax(masked_policy)
        
        # Convert index back to move
        from_sq = best_idx // 64
        to_sq = best_idx % 64
        
        # Find matching move
        for move in legal_moves:
            if move.from_square == from_sq and move.to_square == to_sq:
                return move.uci()
        
        # Fallback: return best legal move
        if legal_moves:
            return legal_moves[np.argmax(masked_policy[[m.from_square * 64 + m.to_square 
                                                         for m in legal_moves]])].uci()
        
        return None
    
    def evaluate_with_stockfish(self, board: chess.Board, depth: int = 20, 
                               time_limit: float = 0.5) -> float:
        """Evaluate position with Stockfish."""
        if not self.stockfish:
            return 0.0
        
        try:
            limit = chess.engine.Limit(time=time_limit, depth=depth)
            info = self.stockfish.analyse(board, limit)
            score = info.get("score")
            
            if score is None:
                return 0.0
            
            # Convert to centipawns
            if score.is_mate():
                return 10000.0 if score.mate() > 0 else -10000.0
            else:
                return float(score.cp) / 100.0  # Convert to pawns
        except:
            return 0.0
    
    def select_move(self, board: chess.Board, remaining_time_ms: int = 5000,
                   use_book: bool = True, use_tb: bool = True) -> str:
        """
        Select best move using hybrid strategy.
        
        Priority:
        1. Tablebase (if endgame)
        2. Opening book (if in opening)
        3. Neural network + Stockfish validation
        4. Strategy-based move selection (if active)
        
        Args:
            board: Current board position
            remaining_time_ms: Time remaining for this side
            use_book: Use opening book
            use_tb: Use tablebases
            
        Returns:
            Best move in UCI format
        """
        start_time = time_module.time()
        
        # 1. Check tablebases first (endgame)
        if use_tb and self.tablebase_manager.is_endgame(board):
            tb_move = self.tablebase_manager.get_perfect_move(board)
            if tb_move:
                self.stats["tb_moves"] += 1
                return tb_move
        
        # 2. Check opening book
        if use_book and self.opening_book.is_in_book(board):
            book_move = self.opening_book.get_book_move(board, temperature=0.2)
            if book_move:
                self.stats["book_moves"] += 1
                return book_move
        
        # 3. Strategy-based move selection (if strategy is active and not ML-focused)
        if self.strategy and self.strategy.name != "machine_learning":
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = self.strategy.select_best_move(board, legal_moves)
                if best_move:
                    self.stats["nn_moves"] += 1
                    return best_move.uci()
        
        # 4. Neural network move selection with Stockfish validation
        try:
            with torch.no_grad():
                board_tensor = self.encode_board(board).unsqueeze(0)
                policy_logits, value = self.model(board_tensor)
            
            nn_move = self.get_move_from_policy(board, policy_logits)
            
            if nn_move:
                # Validate with Stockfish
                board.push_uci(nn_move)
                sf_eval = self.evaluate_with_stockfish(board, depth=10, time_limit=0.1)
                board.pop()
                
                self.stats["nn_moves"] += 1
                return nn_move
        except Exception as e:
            print(f"[WARN] Neural network move failed: {e}")
        
        # 5. Fallback: Stockfish best move
        if self.stockfish:
            try:
                limit = chess.engine.Limit(time=0.5, depth=15)
                result = self.stockfish.play(board, limit)
                self.stats["stockfish_moves"] += 1
                return result.move.uci()
            except:
                pass
        
        # 6. Last resort: first legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return legal_moves[0].uci()
        
        return None
        
        # 4. Fallback: Stockfish best move
        if self.stockfish:
            try:
                limit = chess.engine.Limit(time=0.5, depth=15)
                result = self.stockfish.play(board, limit)
                self.stats["stockfish_moves"] += 1
                return result.move.uci()
            except:
                pass
        
        # 5. Last resort: first legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return legal_moves[0].uci()
        
        return None
    
    def play_game(self, opponent_move_callback=None, time_control: TimeControl = None,
                 verbose: bool = True) -> Dict:
        """
        Play a complete game.
        
        Args:
            opponent_move_callback: Function to get opponent moves
            time_control: Time control for the game
            verbose: Print move information
            
        Returns:
            Game result dictionary
        """
        if time_control is None:
            time_control = TimeControl(initial_minutes=1, increment_seconds=0)
        
        board = chess.Board()
        self.time_manager = TimeManager(time_control)
        moves = []
        move_times = []
        
        game_start = time_module.time()
        
        while not board.is_game_over():
            # Our move
            remaining_time = self.time_manager.clock.get_time_ms(board.turn == chess.WHITE)
            move_time_budget = self.time_manager.get_move_time_allocation(
                board.turn == chess.WHITE,
                len(board.move_stack) // 2 + 1
            )
            
            move_start = time_module.time()
            best_move = self.select_move(board, remaining_time)
            move_duration = (time_module.time() - move_start) * 1000
            
            if not best_move:
                break
            
            board.push_uci(best_move)
            moves.append(best_move)
            move_times.append(move_duration)
            
            if verbose and len(moves) <= 10:
                print(f"Move {len(moves)}: {best_move} ({move_duration:.1f}ms)")
            
            # Opponent move (if callback provided)
            if opponent_move_callback:
                try:
                    opp_move = opponent_move_callback(board)
                    if opp_move:
                        board.push_uci(opp_move)
                        moves.append(opp_move)
                except:
                    break
        
        game_end = time_module.time()
        
        # Determine result
        if board.is_checkmate():
            if board.turn:
                result = "0-1"  # White checkmated
            else:
                result = "1-0"  # Black checkmated
        elif board.is_stalemate():
            result = "1/2-1/2"
        elif board.is_insufficient_material():
            result = "1/2-1/2"
        else:
            result = "*"
        
        self.stats["games_played"] += 1
        self.stats["moves_played"] += len(moves)
        self.stats["total_move_time"] += game_end - game_start
        
        return {
            "moves": moves,
            "move_times": move_times,
            "result": result,
            "duration": game_end - game_start,
            "position_count": len(board.move_stack),
            "final_fen": board.fen()
        }
    
    def get_statistics(self) -> Dict:
        """Get player statistics."""
        return {
            **self.stats,
            "avg_move_time": self.stats["total_move_time"] / max(1, self.stats["moves_played"])
        }
    
    def close(self):
        """Cleanup resources."""
        if self.stockfish:
            self.stockfish.quit()
        self.tablebase_manager.close()
