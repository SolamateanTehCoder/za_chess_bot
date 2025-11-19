"""Self-play opponent - the model plays against itself."""
import chess
import torch
import numpy as np
from typing import Tuple, Optional
import time


class SelfPlayOpponent:
    """
    The AI model playing against itself.
    No external engine needed - uses the same neural network.
    """
    
    def __init__(self, model, device='cpu', temperature=1.0):
        """
        Initialize self-play opponent.
        
        Args:
            model: Neural network model to use
            device: Device to run model on (cpu/cuda)
            temperature: Temperature for move selection (higher = more random)
        """
        self.model = model
        self.device = device
        self.temperature = temperature
        self.is_ready = True
    
    def get_best_move(self, board: chess.Board) -> Tuple[Optional[chess.Move], torch.Tensor]:
        """
        Get best move for current position using the neural network.
        Returns the move and the value prediction.
        
        Args:
            board: Chess board position
            
        Returns:
            Tuple of (move, value_prediction)
        """
        # Get legal moves
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, torch.tensor(0.0)
        
        # For simplicity, return random legal move
        # In a real implementation, you'd use the policy head
        move = np.random.choice(legal_moves)
        
        return move, torch.tensor(0.0)
    
    def get_move(self, board: chess.Board, use_best=False) -> Optional[chess.Move]:
        """
        Get a move for the current position.
        
        Args:
            board: Chess board position
            use_best: If True, use greedy selection; else use sampling
            
        Returns:
            Chess move
        """
        move, _ = self.get_best_move(board)
        return move
    
    def quit(self):
        """Clean up - no engine to quit."""
        pass


class SelfPlayGameWorker:
    """
    Worker for self-play games where the model plays against itself.
    """
    
    def __init__(self, game_id, model, device, use_knowledge=True):
        """
        Initialize self-play game worker.
        
        Args:
            game_id: Unique game identifier
            model: Neural network model
            device: Device to run on
            use_knowledge: Whether to use chess knowledge
        """
        self.game_id = game_id
        self.model = model
        self.device = device
        self.use_knowledge = use_knowledge
        self.opponent = SelfPlayOpponent(model, device)
        self.temperature = 1.0  # For move selection
        
        # Track game as tuple (move, player)
        self.move_history = []
    
    def get_move_from_network(self, board, legal_moves_mask):
        """Get move from neural network."""
        with torch.no_grad():
            policy, value = self.model(board.unsqueeze(0))
            policy = policy.squeeze(0)
            
            # Mask illegal moves
            policy = policy.masked_fill(~legal_moves_mask, float('-inf'))
            
            # Apply temperature
            policy = policy / self.temperature
            
            # Get probabilities
            move_probs = torch.softmax(policy, dim=0)
            
            # Sample move
            move_idx = torch.multinomial(move_probs, 1).item()
            
            return move_idx, policy, value
    
    def play_game(self, env, play_as_white):
        """
        Play a self-play game with bullet time control (60 seconds per side).
        
        Args:
            env: Chess environment
            play_as_white: Whether this instance plays as white
            
        Returns:
            Game result tuple (result, moves_count, experiences, times_exceeded)
        """
        from chess_env import ChessEnvironment
        from comprehensive_chess_knowledge import ComprehensiveChessKnowledge
        
        env.reset()  # Reset environment to get fresh board
        game_knowledge = ComprehensiveChessKnowledge() if self.use_knowledge else None
        
        experiences = []  # List of (state, action, reward, value, log_prob) tuples
        self.move_history = []
        
        move_count = 0
        max_moves = 200
        
        # Bullet time control: 60 seconds per player
        time_limit = 60.0  # seconds
        white_time = time_limit
        black_time = time_limit
        last_move_time = time.time()
        times_exceeded = {'white': False, 'black': False}
        
        while not env.board.is_game_over() and move_count < max_moves:
            current_time = time.time()
            elapsed = current_time - last_move_time
            
            # Update remaining time for current player
            if env.board.turn == chess.WHITE:
                white_time -= elapsed
                if white_time <= 0:
                    times_exceeded['white'] = True
                    result = "Loss" if play_as_white else "Win"  # Timeout is a loss
                    return {
                        'result': result,
                        'moves': move_count,
                        'experiences': experiences,
                        'move_history': self.move_history,
                        'timeout': True,
                        'time_exceeded': times_exceeded
                    }
            else:
                black_time -= elapsed
                if black_time <= 0:
                    times_exceeded['black'] = True
                    result = "Loss" if not play_as_white else "Win"  # Timeout is a loss
                    return {
                        'result': result,
                        'moves': move_count,
                        'experiences': experiences,
                        'move_history': self.move_history,
                        'timeout': True,
                        'time_exceeded': times_exceeded
                    }
            
            last_move_time = current_time
            
            is_ai_turn = (env.board.turn == chess.WHITE) == play_as_white
            
            if is_ai_turn:
                # Get board state
                state_tensor = env.encode_board().to(self.device)
                legal_moves_mask = env.get_legal_moves_mask().to(self.device)
                
                # Get neural network move
                move_idx, policy, value = self.get_move_from_network(state_tensor, legal_moves_mask)
                
                # Check for opening book move
                move = env.index_to_move(move_idx)
                move_source = "self_play"
                
                if game_knowledge:
                    knowledge_move, source = game_knowledge.get_assisted_move(env.board)
                    if knowledge_move:
                        move = knowledge_move
                        move_source = f"opening_book({source})"
                        move_idx = env.move_to_index(move)
                
                # Fallback to random if invalid
                if move is None or move not in env.board.legal_moves:
                    legal_moves = list(env.board.legal_moves)
                    move = np.random.choice(legal_moves)
                    move_idx = env.move_to_index(move)
                
                # Make the move
                next_state, reward, done, info = env.step(move)
                
                # Get log probability
                log_probs = torch.log_softmax(policy, dim=0)
                log_prob = log_probs[move_idx]
                
                # Store experience
                experiences.append({
                    'state': state_tensor.cpu(),
                    'action': move_idx,
                    'reward': reward,
                    'value': value.item(),
                    'log_prob': log_prob.item(),
                    'source': move_source
                })
                
                self.move_history.append((move, f"AI-{move_source}"))
            
            else:
                # Opponent's turn (also using neural network)
                state_tensor = env.encode_board().to(self.device)
                legal_moves_mask = env.get_legal_moves_mask().to(self.device)
                
                move_idx, policy, value = self.get_move_from_network(state_tensor, legal_moves_mask)
                
                move = env.index_to_move(move_idx)
                move_source = "self_play"
                
                # Check for opening book
                if game_knowledge:
                    knowledge_move, source = game_knowledge.get_assisted_move(env.board)
                    if knowledge_move:
                        move = knowledge_move
                        move_source = f"opening_book({source})"
                
                # Fallback to random if invalid
                if move is None or move not in env.board.legal_moves:
                    legal_moves = list(env.board.legal_moves)
                    move = np.random.choice(legal_moves)
                
                next_state, reward, done, info = env.step(move)
                self.move_history.append((move, f"Opponent-{move_source}"))
            
            move_count += 1
        
        # Determine result
        if env.board.is_checkmate():
            result = "Win" if (env.board.turn != chess.WHITE) == play_as_white else "Loss"
        elif env.board.is_stalemate() or env.board.is_insufficient_material() or env.board.is_repetition() or env.board.is_fivefold_repetition():
            result = "Draw"
        else:
            result = "Draw"  # Max moves reached
        
        return {
            'result': result,
            'moves': move_count,
            'experiences': experiences,
            'move_history': self.move_history,
            'timeout': False,
            'time_exceeded': times_exceeded
        }
