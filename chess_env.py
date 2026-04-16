import chess
import numpy as np
from stockfish import Stockfish
import os
import glob

class ChessEnv:
    def __init__(self, stockfish_dir="stockfish_bin", depth=10):
        self.board = chess.Board()
        
        # Locate Stockfish binary dynamically based on Windows extraction
        sf_exe = None
        if os.path.exists(stockfish_dir):
            for f in glob.glob(f"{stockfish_dir}/**/stockfish*.exe", recursive=True):
                sf_exe = f
                break
        
        if not sf_exe:
            print("WARNING: Stockfish executable not found in stockfish_bin/. Fallback to system PATH 'stockfish'.")
            sf_exe = "stockfish"
            
        self.stockfish = Stockfish(path=sf_exe, depth=depth)
        
    def reset(self):
        self.board.reset()
        return self.get_state()
        
    def get_state(self):
        # Convert chess.Board to 14x8x8 tensor
        state = np.zeros((14, 8, 8), dtype=np.float32)
        piece_map = self.board.piece_map()
        
        for sq, piece in piece_map.items():
            # piece type: 1(P), 2(N), 3(B), 4(R), 5(Q), 6(K)
            channel = piece.piece_type - 1
            if not piece.color: # Black
                channel += 6
                
            row = chess.square_rank(sq)
            col = chess.square_file(sq)
            state[channel][row][col] = 1.0
            
        # 13th channel: player turn (all 1s for white, all 0s for black)
        if self.board.turn == chess.WHITE:
            state[12].fill(1.0)
            
        # 14th channel: castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE): state[13][0][7] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE): state[13][0][0] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK): state[13][7][7] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK): state[13][7][0] = 1.0
        
        return state
        
    def get_win_prob_for_player(self, player_color):
        self.stockfish.set_fen_position(self.board.fen())
        eval_info = self.stockfish.get_evaluation()
        
        # Convert Stockfish evaluation format to centipawns
        cp = 0
        if eval_info["type"] == "cp":
            cp = eval_info["value"]
        else:
            # Mate score handling (-/+)
            cp = 10000 if eval_info["value"] > 0 else -10000
            
        # Simple Win Probability curve (elo-based approx.)
        wp_white = 1.0 / (1.0 + 10.0 ** (-cp / 400.0))
        
        if player_color == chess.WHITE:
            return wp_white
        else:
            return 1.0 - wp_white

    def step(self, move_uci):
        player_color = self.board.turn 
        move = chess.Move.from_uci(move_uci)
        
        if move not in self.board.legal_moves:
            # Heavy penalty for illegal move (Negative accuracy mapping)
            return self.get_state(), -10.0, True, {"msg": "illegal move"}
            
        wp_before = self.get_win_prob_for_player(player_color)
        
        self.board.push(move)
        is_done = self.board.is_game_over()
        
        if not is_done:
            # What is the win probability of the SAME player after their move?
            wp_after = self.get_win_prob_for_player(player_color) 
            
            # Calculate accuracy out of 100
            wp_drop = max(0.0, wp_before - wp_after)
            accuracy = max(0.0, 100.0 - (wp_drop * 100.0))
            
            # User specifically requested reward to be: accuracy / 10
            reward = accuracy / 10.0
        else:
            res = self.board.result()
            if res == "1/2-1/2":
                reward = 5.0 # Neutral standard for draw (50% accurate)
            else:
                # Given game ended after this move, and it's decisive,
                # the player who just moved has won.
                reward = 10.0 # 100% accurate
        
        return self.get_state(), reward, is_done, {"msg": "success"}
