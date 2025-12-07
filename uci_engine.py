"""
UCI (Universal Chess Interface) protocol implementation.
Enables the bot to play in official WCCC tournaments with proper protocol compliance.
Supports all standard UCI commands and time management variations.
"""

import chess
import chess.engine
import threading
import time
from queue import Queue, Empty
from typing import Optional, Dict, Tuple
from datetime import datetime
import sys


class UCIEngine:
    """
    UCI-compliant chess engine wrapper.
    Handles protocol communication and move generation.
    """
    
    def __init__(self, name: str = "ZaChessBot", author: str = "Arjun"):
        """
        Initialize UCI engine.
        
        Args:
            name: Engine name for UCI identification
            author: Author name for UCI identification
        """
        self.name = name
        self.author = author
        self.is_ready = False
        self.position_history = []
        self.board = chess.Board()
        self.options = {
            "Hash": 256,  # MB
            "Threads": 4,
            "MultiPV": 1,
            "UCI_ShowWDL": False,
            "UCI_ShowCurrLine": False,
        }
        self.go_params = {}
        self.best_move = None
        self.ponder_move = None
        self.search_thread = None
        self.stop_search = False
        self.move_queue = Queue()
        
        print(f"id name {self.name}")
        print(f"id author {self.author}")
        self._print_options()
        print("uciok")
    
    def _print_options(self):
        """Print available UCI options."""
        option_specs = [
            ("Hash", "spin", 256, 16, 1024),
            ("Threads", "spin", 4, 1, 8),
            ("MultiPV", "spin", 1, 1, 5),
            ("UCI_ShowWDL", "check", False),
            ("UCI_ShowCurrLine", "check", False),
        ]
        
        for name, opt_type, default, min_val, max_val in option_specs:
            if opt_type == "spin":
                print(f"option name {name} type {opt_type} default {default} min {min_val} max {max_val}")
            else:
                print(f"option name {name} type {opt_type} default {default}")
    
    def setoption(self, name: str, value=None):
        """
        Set UCI option.
        
        Args:
            name: Option name
            value: Option value (None for boolean options)
        """
        if name in self.options:
            if value is None:
                self.options[name] = True
            else:
                try:
                    # Try to convert to appropriate type
                    if isinstance(self.options[name], bool):
                        self.options[name] = value.lower() == "true"
                    else:
                        self.options[name] = type(self.options[name])(value)
                except:
                    pass
    
    def isready(self) -> bool:
        """Check if engine is ready."""
        return True
    
    def ucinewgame(self):
        """Initialize new game."""
        self.board = chess.Board()
        self.position_history = []
        self.best_move = None
        self.ponder_move = None
    
    def position(self, fen: str = None, moves: list = None):
        """
        Set board position.
        
        Args:
            fen: FEN string (or "startpos" for initial position)
            moves: List of moves in UCI format to apply
        """
        if fen == "startpos":
            self.board = chess.Board()
        elif fen:
            self.board = chess.Board(fen)
        
        self.position_history = [self.board.copy()]
        
        if moves:
            for move_uci in moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    self.board.push(move)
                    self.position_history.append(self.board.copy())
                except:
                    pass
    
    def go(self, **kwargs) -> Tuple[str, Optional[str]]:
        """
        Start search and return best move.
        
        Supported parameters:
        - searchmoves: List of moves to search
        - ponder: Boolean for ponder mode
        - wtime: White time in ms
        - btime: Black time in ms
        - winc: White increment in ms
        - binc: Black increment in ms
        - movestogo: Moves until next time control
        - depth: Maximum search depth
        - nodes: Maximum search nodes
        - mate: Search for mate in N moves
        - movetime: Time per move in ms
        
        Returns:
            Tuple of (best_move, ponder_move) in UCI format
        """
        self.stop_search = False
        self.go_params = kwargs
        
        # Parse time control
        wtime = int(kwargs.get("wtime", 0))
        btime = int(kwargs.get("btime", 0))
        winc = int(kwargs.get("winc", 0))
        binc = int(kwargs.get("binc", 0))
        movetime = int(kwargs.get("movetime", 0))
        depth = int(kwargs.get("depth", 20))
        
        # Allocate time for this move
        time_for_move = self._allocate_time(wtime, btime, winc, binc, movetime)
        
        # Start search in thread (with timeout)
        self.search_thread = threading.Thread(
            target=self._search,
            args=(time_for_move, depth)
        )
        self.search_thread.daemon = True
        self.search_thread.start()
        
        # Wait for result
        try:
            best_move, ponder = self.move_queue.get(timeout=time_for_move / 1000.0 + 1)
            self.best_move = best_move
            self.ponder_move = ponder
            return best_move, ponder
        except Empty:
            # Return any legal move if search times out
            if self.board.legal_moves:
                for move in self.board.legal_moves:
                    return move.uci(), None
            return None, None
    
    def _allocate_time(self, wtime: int, btime: int, winc: int, binc: int, movetime: int) -> int:
        """
        Allocate time for current move.
        
        Args:
            wtime: White time (ms)
            btime: Black time (ms)
            winc: White increment (ms)
            binc: Black increment (ms)
            movetime: Fixed time per move (ms)
            
        Returns:
            Time allocation for this move in ms
        """
        if movetime > 0:
            return movetime
        
        # Get current side's time
        my_time = wtime if self.board.turn == chess.WHITE else btime
        my_inc = winc if self.board.turn == chess.WHITE else binc
        
        if my_time <= 0:
            return 100  # Fallback: 100ms
        
        # Conservative allocation: time / 30 + half increment
        # This is a simple strategy; more sophisticated algorithms exist
        allocated = max(50, my_time // 30 + my_inc // 2)
        
        # Never use more than 80% of remaining time
        allocated = min(allocated, my_time * 4 // 5)
        
        return allocated
    
    def _search(self, time_limit_ms: int, max_depth: int):
        """
        Perform search (placeholder for actual search algorithm).
        
        Args:
            time_limit_ms: Time limit for search
            max_depth: Maximum search depth
        """
        import time
        start_time = time.time()
        time_limit = time_limit_ms / 1000.0
        
        best_move = None
        ponder_move = None
        
        # Simple move selection: use first legal move
        # In real implementation, this would be iterative deepening with NN evaluation
        for move in self.board.legal_moves:
            best_move = move
            break
        
        # Check for ponder move (first legal move after best move)
        if best_move:
            self.board.push(best_move)
            for move in self.board.legal_moves:
                ponder_move = move
                break
            self.board.pop()
        
        self.move_queue.put((best_move.uci() if best_move else None, 
                             ponder_move.uci() if ponder_move else None))
    
    def stop(self):
        """Stop current search."""
        self.stop_search = True
        if self.search_thread:
            self.search_thread.join(timeout=0.5)
    
    def ponderhit(self):
        """Ponder hit - continue search with new move."""
        pass
    
    def quit(self):
        """Quit the engine."""
        self.stop()
        sys.exit(0)


class UCIProtocol:
    """
    UCI protocol handler for stdin/stdout communication.
    Main entry point for tournament play.
    """
    
    def __init__(self):
        """Initialize UCI protocol handler."""
        self.engine = UCIEngine()
        self.running = True
    
    def run(self):
        """Main protocol loop."""
        while self.running:
            try:
                line = input().strip()
                
                if not line:
                    continue
                
                tokens = line.split()
                command = tokens[0].lower()
                
                if command == "uci":
                    # Already printed in engine init, just respond ok
                    pass
                
                elif command == "isready":
                    print("readyok")
                
                elif command == "setoption":
                    # setoption name <id> value <x>
                    if "name" in tokens and "value" in tokens:
                        name_idx = tokens.index("name")
                        value_idx = tokens.index("value")
                        name = tokens[name_idx + 1]
                        value = tokens[value_idx + 1] if value_idx + 1 < len(tokens) else None
                        self.engine.setoption(name, value)
                
                elif command == "ucinewgame":
                    self.engine.ucinewgame()
                
                elif command == "position":
                    # position [fen <fenstring> | startpos ] moves <move1> ... <moveN>
                    fen = None
                    moves = []
                    
                    if "startpos" in tokens:
                        fen = "startpos"
                        moves_idx = tokens.index("moves") if "moves" in tokens else None
                        if moves_idx:
                            moves = tokens[moves_idx + 1:]
                    elif "fen" in tokens:
                        fen_idx = tokens.index("fen")
                        moves_idx = tokens.index("moves") if "moves" in tokens else None
                        
                        if moves_idx:
                            fen = " ".join(tokens[fen_idx + 1:moves_idx])
                            moves = tokens[moves_idx + 1:]
                        else:
                            fen = " ".join(tokens[fen_idx + 1:])
                    
                    self.engine.position(fen, moves)
                
                elif command == "go":
                    # Parse go parameters
                    params = {}
                    i = 1
                    while i < len(tokens):
                        param = tokens[i]
                        if param in ["searchmoves", "ponder", "wtime", "btime", "winc", 
                                    "binc", "movestogo", "depth", "nodes", "mate", "movetime"]:
                            if param == "ponder":
                                params["ponder"] = True
                                i += 1
                            else:
                                params[param] = tokens[i + 1] if i + 1 < len(tokens) else "0"
                                i += 2
                        else:
                            i += 1
                    
                    best_move, ponder = self.engine.go(**params)
                    
                    if best_move:
                        response = f"bestmove {best_move}"
                        if ponder:
                            response += f" ponder {ponder}"
                        print(response)
                    else:
                        print("bestmove 0000")  # No legal move (shouldn't happen)
                
                elif command == "stop":
                    self.engine.stop()
                
                elif command == "ponderhit":
                    self.engine.ponderhit()
                
                elif command == "quit":
                    self.engine.quit()
                    self.running = False
                
            except EOFError:
                break
            except Exception as e:
                print(f"info string error: {e}", file=sys.stderr)


def main():
    """Entry point for UCI protocol."""
    protocol = UCIProtocol()
    protocol.run()


if __name__ == "__main__":
    main()
