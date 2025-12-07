"""
Endgame tablebase integration for perfect play in endgames.
Supports Syzygy tablebases with fast probing and fallback to Stockfish.
Critical for WCCC-level play where endgame accuracy matters.
"""

import chess
import chess.syzygy
from pathlib import Path
from typing import Optional, Tuple, Dict
import os


class TablebaseManager:
    """
    Manager for endgame tablebases.
    Provides perfect move selection in endgames (when pieces <= 6).
    """
    
    def __init__(self, tablebase_paths: list = None, max_pieces: int = 6):
        """
        Initialize tablebase manager.
        
        Args:
            tablebase_paths: List of paths to Syzygy tablebase directories
            max_pieces: Maximum pieces to use tablebases for (6 = standard)
        """
        self.tablebase_paths = tablebase_paths or self._default_paths()
        self.max_pieces = max_pieces
        self.probes = 0
        self.hits = 0
        self.misses = 0
        self.board_probes = {}  # Cache for FEN -> result
        
        self.tablebases = []
        self._load_tablebases()
    
    def _default_paths(self) -> list:
        """Get default tablebase paths."""
        paths = []
        
        # Common Windows paths
        windows_paths = [
            "C:\\Syzygy",
            "C:\\syzygy",
            os.path.expanduser("~\\Syzygy"),
        ]
        
        # Common Linux paths
        linux_paths = [
            "/opt/syzygy",
            os.path.expanduser("~/syzygy"),
            "/usr/share/syzygy",
        ]
        
        all_paths = windows_paths + linux_paths
        for path in all_paths:
            if os.path.isdir(path):
                paths.append(path)
        
        return paths
    
    def _load_tablebases(self):
        """Load tablebases from available paths."""
        loaded_count = 0
        
        for path in self.tablebase_paths:
            if not os.path.isdir(path):
                continue
            
            try:
                # Try to load Syzygy tablebases
                tb = chess.syzygy.open_tablebases(path)
                self.tablebases.append(tb)
                loaded_count += 1
                print(f"[INFO] Loaded tablebases from: {path}")
            except Exception as e:
                print(f"[WARN] Failed to load tablebases from {path}: {e}")
        
        if loaded_count == 0:
            print("[WARN] No Syzygy tablebases found - endgame play will use Stockfish")
            print("[INFO] Recommended: Download Syzygy 6-piece tablebases from")
            print("       https://github.com/syzygy-tables/tb")
    
    def is_endgame(self, board: chess.Board) -> bool:
        """Check if position is in endgame range (suitable for tablebases)."""
        piece_count = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                     len(board.pieces(chess.PAWN, chess.BLACK)) + \
                     len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                     len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                     len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                     len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                     len(board.pieces(chess.ROOK, chess.WHITE)) + \
                     len(board.pieces(chess.ROOK, chess.BLACK)) + \
                     len(board.pieces(chess.QUEEN, chess.WHITE)) + \
                     len(board.pieces(chess.QUEEN, chess.BLACK)) + \
                     len(board.pieces(chess.KING, chess.WHITE)) + \
                     len(board.pieces(chess.KING, chess.BLACK))
        return piece_count <= self.max_pieces
    
    def probe_wdl(self, board: chess.Board) -> Optional[Tuple[int, int, int]]:
        """
        Probe tablebases for Win/Draw/Loss information.
        
        Args:
            board: Chess board position
            
        Returns:
            Tuple of (wins, draws, losses) from white perspective, or None if not in TB
        """
        fen = board.fen()
        self.probes += 1
        
        # Check cache
        if fen in self.board_probes:
            result = self.board_probes[fen]
            if result is not None:
                self.hits += 1
            else:
                self.misses += 1
            return result
        
        # Only probe if endgame
        if not self.is_endgame(board):
            self.board_probes[fen] = None
            self.misses += 1
            return None
        
        # Try to probe tablebases
        for tb in self.tablebases:
            try:
                result = tb.probe_wdl(board)
                if result is not None:
                    self.board_probes[fen] = result
                    self.hits += 1
                    return result
            except:
                continue
        
        self.board_probes[fen] = None
        self.misses += 1
        return None
    
    def probe_dtz(self, board: chess.Board) -> Optional[int]:
        """
        Probe tablebases for DTZ (Distance To Zeroing).
        Used for finding fastest win or avoiding loss.
        
        Args:
            board: Chess board position
            
        Returns:
            DTZ value or None if not in TB
        """
        # Only probe if endgame
        if not self.is_endgame(board):
            return None
        
        for tb in self.tablebases:
            try:
                result = tb.probe_dtz(board)
                return result
            except:
                continue
        
        return None
    
    def get_perfect_move(self, board: chess.Board) -> Optional[str]:
        """
        Get the perfect move from tablebase (best move to win/draw or avoid loss).
        
        Args:
            board: Chess board position
            
        Returns:
            Best move UCI string or None
        """
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        
        wins, draws, losses = wdl
        
        # Find best move based on WDL
        best_move = None
        best_wdl = (losses, draws, wins)  # We want more wins, fewer losses
        
        for move in board.legal_moves:
            board.push(move)
            opp_wdl = self.probe_wdl(board)
            board.pop()
            
            if opp_wdl is None:
                continue
            
            # Convert opponent WDL to our perspective
            opp_wins, opp_draws, opp_losses = opp_wdl
            our_wdl = (opp_losses, opp_draws, opp_wins)
            
            # Comparison tuple: prefer more wins, more draws, fewer losses
            if best_move is None or our_wdl < best_wdl:
                best_move = move
                best_wdl = our_wdl
        
        return best_move.uci() if best_move else None
    
    def evaluate_endgame(self, board: chess.Board) -> Optional[float]:
        """
        Evaluate endgame position using tablebase WDL.
        
        Args:
            board: Chess board position
            
        Returns:
            Float value -1.0 (loss) to 1.0 (win), or None if not in TB
        """
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        
        wins, draws, losses = wdl
        total = wins + draws + losses
        
        if total == 0:
            return 0.0
        
        # Convert to score: win=1.0, draw=0.0, loss=-1.0
        score = (wins - losses) / total
        return float(score)
    
    def get_statistics(self) -> Dict:
        """Get tablebase probe statistics."""
        hit_rate = (self.hits / self.probes * 100) if self.probes > 0 else 0
        return {
            "total_probes": self.probes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.board_probes),
            "tablebases_loaded": len(self.tablebases)
        }
    
    def clear_cache(self):
        """Clear probe cache (for long games to prevent memory bloat)."""
        self.board_probes.clear()
    
    def close(self):
        """Close tablebase connections."""
        for tb in self.tablebases:
            try:
                tb.close()
            except:
                pass
