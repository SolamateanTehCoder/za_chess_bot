"""Parallel game player for training."""

import threading
import queue
import chess
import torch
import numpy as np
from datetime import datetime
from chess_env import ChessEnvironment
from stockfish_opponent import StockfishOpponent
from comprehensive_chess_knowledge import ComprehensiveChessKnowledge
from config import USE_CUDA, GAMES_LOG_FILE


class GameExperience:
    """Container for game experience data."""
    
    def __init__(self, ai_plays_white=True):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.done = False
        self.result = None
        self.ai_plays_white = ai_plays_white
        self.move_history = []  # Store all moves for logging
    
    def add_step(self, state, action, reward, value, log_prob):
        """Add a step to the experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def add_move(self, move, player):
        """Add a move to history for logging."""
        self.move_history.append((move, player))
    
    def set_result(self, result):
        """Set the final game result."""
        self.result = result
        self.done = True


class GameWorker(threading.Thread):
    """
    Worker thread that plays a single game against Stockfish.
    """
    
    def __init__(self, game_id, model, device, result_queue, play_as_white=True, use_knowledge=True):
        """
        Initialize game worker.
        
        Args:
            game_id: Unique identifier for this game
            model: Neural network model
            device: Torch device (CPU or CUDA)
            result_queue: Queue to put game results
            play_as_white: Whether the AI plays as white or black
            use_knowledge: Whether to use opening book and endgame knowledge
        """
        super().__init__()
        self.game_id = game_id
        self.model = model
        self.device = device
        self.result_queue = result_queue
        self.play_as_white = play_as_white
        self.experience = GameExperience(ai_plays_white=play_as_white)
        self.knowledge = ComprehensiveChessKnowledge() if use_knowledge else None
    
    def run(self):
        """Run a complete game and collect experience."""
        env = ChessEnvironment()
        stockfish = StockfishOpponent()
        
        try:
            print(f"Game {self.game_id}: Starting Stockfish...")
            stockfish.start()
            print(f"Game {self.game_id}: Stockfish started successfully")
            state = env.reset()
            
            move_count = 0
            max_moves = 200  # Limit game length
            
            while not env.board.is_game_over() and move_count < max_moves:
                # Check whose turn it is
                is_ai_turn = (env.board.turn == chess.WHITE) == self.play_as_white
                
                if is_ai_turn:
                    # AI's turn
                    move = None
                    move_source = "neural_network"
                    
                    # Try to use chess knowledge first (opening book or endgame)
                    if self.knowledge:
                        knowledge_move, source = self.knowledge.get_assisted_move(env.board)
                        if knowledge_move:
                            move = knowledge_move
                            move_source = source
                    
                    # If no knowledge move, use neural network
                    if move is None:
                        state_tensor = state.to(self.device)
                        legal_moves_mask = env.get_legal_moves_mask().to(self.device)
                        
                        # Get model prediction
                        with torch.no_grad():
                            policy, value = self.model(state_tensor.unsqueeze(0))
                            policy = policy.squeeze(0)
                            
                            # Mask illegal moves
                            policy = policy.masked_fill(~legal_moves_mask, float('-inf'))
                            
                            # Get move probabilities
                            move_probs = torch.softmax(policy, dim=0)
                            
                            # Sample a move
                            move_idx = torch.multinomial(move_probs, 1).item()
                            log_prob = torch.log(move_probs[move_idx] + 1e-8)
                        
                        # Convert index to move
                        move = env.index_to_move(move_idx)
                        
                        # If move is invalid, pick a random legal move
                        if move is None or move not in env.board.legal_moves:
                            legal_moves = list(env.board.legal_moves)
                            move = np.random.choice(legal_moves)
                            move_idx = env.move_to_index(move)
                    
                    # Make the move
                    next_state, reward, done, info = env.step(move)
                    
                    # Record move for logging (with source)
                    self.experience.add_move(move, f"AI-{move_source}")
                    
                    # Store experience only if from neural network
                    if move_source == "neural_network":
                        self.experience.add_step(
                            state.cpu(),
                            move_idx,
                            reward,
                            value.item(),
                            log_prob.item()
                        )
                    
                    state = next_state
                    
                else:
                    # Stockfish's turn
                    move = stockfish.get_move(env.board)
                    state, reward, done, info = env.step(move)
                    
                    # Record move for logging
                    self.experience.add_move(move, "Stockfish")
                
                move_count += 1
            
            # Set final result
            if env.board.is_game_over():
                result = env.board.result()
            else:
                result = "1/2-1/2"  # Draw if max moves reached
            
            self.experience.set_result(result)
            
            # Determine if AI won, lost, or drew
            ai_result = None
            if result == "1-0":
                ai_result = "win" if self.play_as_white else "loss"
            elif result == "0-1":
                ai_result = "loss" if self.play_as_white else "win"
            else:
                ai_result = "draw"
            
            # Get knowledge usage stats
            knowledge_stats = self.knowledge.get_stats() if self.knowledge else {}
            
            # Log game details to games.log
            self._log_game(result, ai_result, move_count, knowledge_stats)
            
            print(f"Game {self.game_id} finished: {result} (AI: {ai_result})")
            
        except Exception as e:
            print(f"Error in game {self.game_id}: {e}")
            import traceback
            traceback.print_exc()
            self.experience.set_result("error")
        
        finally:
            stockfish.close()
            self.result_queue.put((self.game_id, self.experience))
    
    def _log_game(self, result, ai_result, move_count, knowledge_stats=None):
        """Log game details to games.log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = "White" if self.play_as_white else "Black"
        
        # Create detailed game log
        log_lines = []
        log_lines.append("=" * 80)
        log_lines.append(f"[{timestamp}] Game {self.game_id}")
        log_lines.append(f"AI Color: {color} | Result: {result} | Outcome: {ai_result.upper()} | Moves: {move_count}")
        
        # Add knowledge stats if available
        if knowledge_stats:
            log_lines.append(f"Book Moves: {knowledge_stats.get('book_moves_used', 0)} | "
                           f"Endgame Assists: {knowledge_stats.get('endgame_assists', 0)}")
        
        log_lines.append("-" * 80)
        
        # Log each move
        for i, (move, player) in enumerate(self.experience.move_history, 1):
            move_num = (i + 1) // 2
            if i % 2 == 1:
                # Start a new move pair line
                log_lines.append(f"{move_num}. {move.uci():8} ({player})")
            else:
                # Add to same line for paired moves
                log_lines[-1] = log_lines[-1] + f"  {move.uci():8} ({player})"
        
        # Add final result
        log_lines.append("")
        log_lines.append(f"Final Result: {result}")
        log_lines.append("=" * 80)
        log_lines.append("")
        
        try:
            with open(GAMES_LOG_FILE, 'a') as f:
                f.write('\n'.join(log_lines) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to {GAMES_LOG_FILE}: {e}")


class ParallelGamePlayer:
    """
    Manages multiple games playing in parallel using threads.
    """
    
    def __init__(self, model, num_games=10, device=None, use_knowledge=True):
        """
        Initialize parallel game player.
        
        Args:
            model: Neural network model
            num_games: Number of games to play in parallel
            device: Torch device (CPU or CUDA)
            use_knowledge: Whether to use opening book and endgame knowledge
        """
        self.model = model
        self.num_games = num_games
        self.use_knowledge = use_knowledge
        
        if device is None:
            self.device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
    
    def play_games(self):
        """
        Play multiple games in parallel.
        
        Returns:
            List of GameExperience objects
        """
        # Set model to evaluation mode for playing games
        self.model.eval()
        
        result_queue = queue.Queue()
        workers = []
        
        print(f"\nStarting {self.num_games} parallel games...")
        
        # Create and start worker threads
        for i in range(self.num_games):
            # Alternate between playing as white and black
            play_as_white = (i % 2 == 0)
            worker = GameWorker(i, self.model, self.device, result_queue, play_as_white, self.use_knowledge)
            worker.start()
            workers.append(worker)
        
        # Wait for all games to complete
        for worker in workers:
            worker.join()
        
        # Collect all experiences
        experiences = []
        while not result_queue.empty():
            game_id, experience = result_queue.get()
            experiences.append(experience)
        
        print(f"Completed {len(experiences)} games")
        
        return experiences
    
    def collect_training_data(self, experiences):
        """
        Convert game experiences into training data.
        
        Args:
            experiences: List of GameExperience objects
            
        Returns:
            Dictionary with training data
        """
        all_states = []
        all_actions = []
        all_returns = []
        all_advantages = []
        all_log_probs = []
        
        for exp in experiences:
            if not exp.done or len(exp.states) == 0:
                continue
            
            # Calculate returns (discounted cumulative rewards)
            returns = self._calculate_returns(exp.rewards)
            
            # Calculate advantages (returns - values)
            advantages = []
            for ret, val in zip(returns, exp.values):
                advantages.append(ret - val)
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = np.array(advantages)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.tolist()
            
            # Add to training data
            all_states.extend(exp.states)
            all_actions.extend(exp.actions)
            all_returns.extend(returns)
            all_advantages.extend(advantages)
            all_log_probs.extend(exp.log_probs)
        
        return {
            "states": all_states,
            "actions": all_actions,
            "returns": all_returns,
            "advantages": all_advantages,
            "log_probs": all_log_probs
        }
    
    def _calculate_returns(self, rewards, gamma=0.99):
        """
        Calculate discounted returns.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor
            
        Returns:
            List of discounted returns
        """
        returns = []
        R = 0
        
        for reward in reversed(rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        
        return returns
