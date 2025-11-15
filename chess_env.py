"""Chess environment and board state encoding."""

import chess
import numpy as np
import torch


class ChessEnvironment:
    """
    Chess environment that handles board state representation and game logic.
    """
    
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        
    def reset(self):
        """Reset the board to initial position."""
        self.board = chess.Board()
        self.move_history = []
        return self.encode_board()
    
    def step(self, move):
        """
        Make a move on the board.
        
        Args:
            move: chess.Move object
            
        Returns:
            state: Encoded board state
            reward: Reward for the move
            done: Whether the game is over
            info: Additional information
        """
        if move not in self.board.legal_moves:
            # Illegal move penalty
            return self.encode_board(), -1.0, True, {"result": "illegal_move"}
        
        self.board.push(move)
        self.move_history.append(move)
        
        # Check if game is over
        done = self.board.is_game_over()
        
        # Calculate reward
        reward = self._calculate_reward(done)
        
        info = {}
        if done:
            result = self.board.result()
            info["result"] = result
        
        return self.encode_board(), reward, done, info
    
    def _calculate_reward(self, done):
        """Calculate reward based on game state."""
        if not done:
            return 0.0
        
        result = self.board.result()
        if result == "1-0":  # White wins
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        elif result == "0-1":  # Black wins
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        else:  # Draw
            return 0.0
    
    def encode_board(self):
        """
        Encode the board state as a tensor.
        
        Returns:
            Tensor of shape (119, 8, 8) representing the board state
            
        Encoding:
        - Planes 0-11: Piece positions (6 piece types x 2 colors)
        - Planes 12-19: Repetition counts
        - Plane 20: Current player color
        - Plane 21: Total move count
        - Plane 22: Castling rights (white kingside)
        - Plane 23: Castling rights (white queenside)
        - Plane 24: Castling rights (black kingside)
        - Plane 25: Castling rights (black queenside)
        - Plane 26: En passant square
        - Planes 27-118: Last 8 moves history (8 moves x 2 boards x 6 pieces)
        """
        # Initialize the state tensor
        state = np.zeros((119, 8, 8), dtype=np.float32)
        
        # Encode current piece positions
        piece_idx = 0
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                             chess.ROOK, chess.QUEEN, chess.KING]:
                pieces = self.board.pieces(piece_type, color)
                for square in pieces:
                    row = square // 8
                    col = square % 8
                    state[piece_idx, row, col] = 1.0
                piece_idx += 1
        
        # Encode repetition counts (simplified)
        state[12:20, :, :] = 0.0  # Placeholder for repetition encoding
        
        # Current player (plane 20)
        state[20, :, :] = 1.0 if self.board.turn == chess.WHITE else 0.0
        
        # Move count (plane 21)
        state[21, :, :] = min(self.board.fullmove_number / 100.0, 1.0)
        
        # Castling rights (planes 22-25)
        state[22, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.WHITE) else 0.0
        state[23, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.WHITE) else 0.0
        state[24, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.BLACK) else 0.0
        state[25, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        # En passant square (plane 26)
        if self.board.ep_square is not None:
            row = self.board.ep_square // 8
            col = self.board.ep_square % 8
            state[26, row, col] = 1.0
        
        # Move history (planes 27-118) - simplified encoding
        # In a full implementation, encode the last 8 board positions
        state[27:119, :, :] = 0.0
        
        return torch.FloatTensor(state)
    
    def get_legal_moves(self):
        """Get list of legal moves."""
        return list(self.board.legal_moves)
    
    def get_legal_moves_mask(self):
        """
        Get a boolean mask for legal moves.
        
        Returns:
            Tensor of shape (4672,) with True for legal moves
        """
        mask = torch.zeros(4672, dtype=torch.bool)
        for move in self.board.legal_moves:
            move_idx = self.move_to_index(move)
            if 0 <= move_idx < 4672:
                mask[move_idx] = True
        return mask
    
    def move_to_index(self, move):
        """
        Convert a chess move to an index (0-4671).
        
        Encoding scheme:
        - Queen moves: 56 directions x 64 squares = 3584 moves
        - Knight moves: 8 directions x 64 squares = 512 moves
        - Underpromotions: 9 types x 64 squares = 576 moves
        
        Total: 4672 possible moves
        """
        from_square = move.from_square
        to_square = move.to_square
        
        # Simplified move encoding - just use from_square * 64 + to_square
        # In a full implementation, use proper AlphaZero move encoding
        move_idx = from_square * 64 + to_square
        
        # Handle promotions
        if move.promotion is not None:
            move_idx = 4096 + from_square * 4 + (move.promotion - 2)
        
        return move_idx % 4672
    
    def index_to_move(self, index):
        """
        Convert an index back to a chess move.
        
        Args:
            index: Move index (0-4671)
            
        Returns:
            chess.Move object or None if invalid
        """
        if index < 4096:
            from_square = index // 64
            to_square = index % 64
            move = chess.Move(from_square, to_square)
        else:
            # Promotion move
            promotion_data = index - 4096
            from_square = promotion_data // 4
            promotion_piece = (promotion_data % 4) + 2  # 2=knight, 3=bishop, 4=rook, 5=queen
            # Find the to_square by checking legal moves
            for legal_move in self.board.legal_moves:
                if legal_move.from_square == from_square and legal_move.promotion == promotion_piece:
                    return legal_move
            return None
        
        # Verify the move is legal
        if move in self.board.legal_moves:
            return move
        
        # If not legal, try to find a matching legal move
        for legal_move in self.board.legal_moves:
            if legal_move.from_square == from_square and legal_move.to_square == to_square:
                return legal_move
        
        return None
    
    def render(self):
        """Print the board."""
        print(self.board)
        print()
    
    def get_board_copy(self):
        """Return a copy of the current board."""
        return self.board.copy()
