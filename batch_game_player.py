"""
Optimized batch game player for parallel GPU evaluation of multiple games.
Enables playing 28 games with GPU-accelerated move selection.
"""

import torch
import torch.nn as nn
import numpy as np
import chess
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class GameState:
    """Represents a single game state"""
    board: chess.Board
    white_time: float
    black_time: float
    move_history: List[chess.Move]
    rewards: List[float]
    white_accuracies: List[float]
    black_accuracies: List[float]


class BatchGamePlayer:
    """
    GPU-accelerated batch game player.
    Evaluates all 28 games' current positions in parallel on GPU.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, batch_size: int = 28):
        """
        Args:
            model: Neural network model
            device: torch device (cuda or cpu)
            batch_size: Number of games to process in parallel
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.eval()  # Set to eval mode for inference
        
    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert chess board to neural network input tensor (8x8x12).
        12 channels: 6 white pieces + 6 black pieces
        """
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        piece_indices = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = divmod(square, 8)
                piece_idx = piece_indices[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                board_array[row, col, piece_idx + color_offset] = 1.0
        
        return torch.from_numpy(board_array)
    
    def select_moves_batch(self, game_states: List[GameState]) -> List[chess.Move]:
        """
        Select moves for multiple games in parallel using GPU batch evaluation.
        
        Args:
            game_states: List of GameState objects (up to batch_size)
            
        Returns:
            List of selected moves (one per game)
        """
        # Convert all boards to tensors
        boards_tensors = []
        legal_moves_per_game = []
        
        for game_state in game_states:
            board_tensor = self.board_to_tensor(game_state.board)
            boards_tensors.append(board_tensor)
            
            # Get legal moves for this position
            legal_moves = list(game_state.board.legal_moves)
            legal_moves_per_game.append(legal_moves)
        
        # Stack all board tensors into a batch
        batch = torch.stack(boards_tensors).to(self.device)
        
        # Forward pass through model (batch evaluation)
        with torch.no_grad():
            # Output shape: (batch_size, 4096) for move probabilities
            move_logits = self.model(batch)  # GPU-accelerated
        
        selected_moves = []
        
        # Select best move for each game based on model output
        for game_idx, (logits, legal_moves) in enumerate(zip(move_logits, legal_moves_per_game)):
            if not legal_moves:
                raise RuntimeError(f"Game {game_idx}: No legal moves available")
            
            # Convert move to index and get score from model output
            move_scores = {}
            for move in legal_moves:
                # Simple scoring: pieces moved toward center get higher scores
                move_score = self._score_move(game_states[game_idx].board, move, logits)
                move_scores[move] = move_score
            
            # Select highest-scoring move
            best_move = max(move_scores, key=move_scores.get)
            selected_moves.append(best_move)
        
        return selected_moves
    
    def _score_move(self, board: chess.Board, move: chess.Move, logits: torch.Tensor) -> float:
        """
        Score a move based on destination square and model logits.
        Simpler than full evaluation but fast for batch processing.
        """
        dest_square = move.to_square
        dest_row, dest_col = divmod(dest_square, 8)
        
        # Bonus for moving toward center (4 central squares)
        center_distance = min(abs(dest_row - 3.5), abs(dest_col - 3.5))
        center_bonus = 0.1 * (4 - center_distance)
        
        # Get model's confidence for this position
        model_confidence = torch.nn.functional.softmax(logits, dim=0).max().item()
        
        # Combine bonuses
        score = model_confidence + center_bonus
        
        # Slight bonus for captures
        if board.is_capture(move):
            score += 0.2
        
        # Slight penalty for moving king (unless capturing)
        if board.piece_at(move.from_square).piece_type == chess.KING and not board.is_capture(move):
            score -= 0.1
        
        return float(score)
    
    def play_games_batch(
        self,
        initial_boards: List[chess.Board],
        max_moves: int = 200,
        time_per_side: float = 60.0,
    ) -> List[Dict]:
        """
        Play multiple games in parallel with batch GPU evaluation.
        
        Args:
            initial_boards: List of initial board positions
            max_moves: Maximum moves per game
            time_per_side: Time control per player in seconds
            
        Returns:
            List of game results with move histories
        """
        num_games = len(initial_boards)
        game_states = [
            GameState(
                board=board.copy(),
                white_time=time_per_side,
                black_time=time_per_side,
                move_history=[],
                rewards=[],
                white_accuracies=[],
                black_accuracies=[]
            )
            for board in initial_boards
        ]
        
        move_count = 0
        active_games = set(range(num_games))
        
        while active_games and move_count < max_moves:
            # Prepare batch of active games
            active_states = [game_states[i] for i in active_games]
            active_indices = list(active_games)
            
            # Batch evaluate all active games
            try:
                selected_moves = self.select_moves_batch(active_states)
            except Exception as e:
                print(f"[ERROR] Batch move selection failed: {e}")
                break
            
            # Apply moves to each game
            completed_games = []
            
            for local_idx, global_idx in enumerate(active_indices):
                game = game_states[global_idx]
                move = selected_moves[local_idx]
                
                # Apply move
                game.board.push(move)
                game.move_history.append(move)
                
                # Update time (simplified - equal time per move)
                move_time = 0.1
                if game.board.turn == chess.BLACK:  # White just moved
                    game.white_time -= move_time
                else:  # Black just moved
                    game.black_time -= move_time
                
                # Check if game is over
                if game.board.is_game_over():
                    completed_games.append(global_idx)
            
            # Remove completed games from active set
            for idx in completed_games:
                active_games.discard(idx)
            
            move_count += 1
        
        # Format results
        results = []
        for i, game in enumerate(game_states):
            result = {
                'game_id': i,
                'moves': game.move_history,
                'final_fen': game.board.fen(),
                'is_game_over': game.board.is_game_over(),
                'result': self._get_game_result(game.board),
                'move_count': len(game.move_history),
                'white_time': game.white_time,
                'black_time': game.black_time,
            }
            results.append(result)
        
        return results
    
    @staticmethod
    def _get_game_result(board: chess.Board) -> str:
        """Determine game result"""
        if not board.is_game_over():
            return "ongoing"
        
        if board.is_checkmate():
            return "White wins" if board.turn == chess.BLACK else "Black wins"
        elif board.is_stalemate():
            return "Stalemate"
        elif board.is_insufficient_material():
            return "Draw (insufficient material)"
        elif board.is_seventyfive_moves():
            return "Draw (75-move rule)"
        elif board.is_fivefold_repetition():
            return "Draw (5-fold repetition)"
        else:
            return "Draw"


class BatchGameEvaluator:
    """Evaluates batch of games for rewards and accuracy"""
    
    @staticmethod
    def evaluate_batch_positions(
        boards: List[chess.Board],
        model: nn.Module,
        device: torch.device,
        reward_analyzer=None
    ) -> Tuple[List[float], List[float]]:
        """
        Evaluate multiple board positions in parallel.
        
        Args:
            boards: List of board positions
            model: Neural network model
            device: torch device
            reward_analyzer: Optional reward system
            
        Returns:
            Tuple of (rewards, accuracies)
        """
        batch_player = BatchGamePlayer(model, device)
        
        # Convert boards to tensors and batch them
        board_tensors = [batch_player.board_to_tensor(board) for board in boards]
        batch = torch.stack(board_tensors).to(device)
        
        # GPU-accelerated evaluation
        with torch.no_grad():
            evaluations = model(batch)
        
        # Process results
        rewards = []
        accuracies = []
        
        for i, (board, eval_score) in enumerate(zip(boards, evaluations)):
            # Get reward from analyzer if available
            if reward_analyzer:
                reward = reward_analyzer.get_move_reward(
                    board,
                    eval_score.cpu().numpy()
                )
            else:
                reward = float(eval_score.mean().cpu().numpy())
            
            rewards.append(reward)
            
            # Simple accuracy metric (based on model confidence)
            accuracy = min(100.0, float(eval_score.max().cpu().numpy()) * 100)
            accuracies.append(accuracy)
        
        return rewards, accuracies
