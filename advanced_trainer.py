"""
Advanced training system for WCCC-level chess bot.
Learns from self-play games, master games, and tournament play.
Implements curriculum learning and multi-task training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
from collections import deque
import chess
from tqdm import tqdm
from hybrid_player import HybridChessPlayer

from chess_models import ChessNetV2, SimpleChessNet


class GameExperienceDataset(Dataset):
    """Dataset for training from game experiences."""

    def __init__(self, games_file: str, max_positions: int = 1000000):
        """
        Initialize dataset from games file.
        
        Args:
            games_file: JSONL file with game data
            max_positions: Maximum positions to load
        """
        self.positions = []
        self.moves = []
        self.rewards = []
        self.game_outcomes = []
        self.encoder = HybridChessPlayer(use_enhanced_model=True, device='cpu')

        self._load_games(games_file, max_positions)

    def _uci_to_policy_index(self, uci_move: str) -> int:
        """Convert UCI move string to policy index."""
        try:
            move = chess.Move.from_uci(uci_move)
            return move.from_square * 64 + move.to_square
        except:
            return -1 # Should not happen with valid UCI

    def _load_games(self, games_file: str, max_positions: int):
        """Load games from JSONL file."""
        if not Path(games_file).exists():
            print(f"[WARN] Games file not found: {games_file}")
            return

        with open(games_file) as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"Loading games from {games_file}"):
                if len(self.positions) >= max_positions:
                    break
                
                try:
                    data = json.loads(line)
                    moves = data.get("moves", [])
                    result = data.get("result", "*")

                    if not moves or result == "*":
                        continue

                    # Determine outcome
                    if result == "1-0":
                        outcome_val = 1.0
                    elif result == "0-1":
                        outcome_val = -1.0
                    else:
                        outcome_val = 0.0

                    board = chess.Board()
                    for move_uci in moves:
                        if len(self.positions) >= max_positions:
                            break

                        # Encode current position
                        encoded_board = self.encoder.encode_board(board).cpu().numpy()

                        # Get policy index for the move
                        policy_index = self._uci_to_policy_index(move_uci)
                        if policy_index == -1:
                            continue # Skip invalid moves

                        # Determine outcome from current player's perspective
                        perspective_outcome = outcome_val if board.turn == chess.WHITE else -outcome_val

                        self.positions.append(encoded_board)
                        self.moves.append(policy_index)
                        self.game_outcomes.append(perspective_outcome)
                        # Reward is 1 for a good move, -1 for a bad one.
                        # We can use the game outcome as a proxy.
                        self.rewards.append(perspective_outcome)

                        board.push_uci(move_uci)

                except Exception as e:
                    # print(f"[WARN] Failed to parse game line: {e}")
                    continue
        
        print(f"[INFO] Loaded {len(self.positions)} positions from {len(lines)} games.")
    
    def __len__(self):
        """Dataset length."""
        return len(self.positions)
    
    def __getitem__(self, idx):
        """Get single item."""
        return {
            "position": torch.FloatTensor(self.positions[idx]),
            "move": torch.LongTensor([self.moves[idx]]),
            "reward": torch.FloatTensor([self.rewards[idx]]),
            "outcome": torch.FloatTensor([self.game_outcomes[idx]])
        }


class AdvancedTrainer:
    """Advanced trainer with curriculum learning and multi-task training."""
    
    def __init__(self, model, device: str = "cuda", learning_rate: float = 0.001):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Learning rate
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        self.losses = {
            "policy": deque(maxlen=100),
            "value": deque(maxlen=100),
            "total": deque(maxlen=100)
        }
        self.training_step = 0
    
    def train_on_batch(self, batch_states: torch.Tensor, batch_moves: torch.Tensor,
                      batch_rewards: torch.Tensor, batch_outcomes: torch.Tensor) -> Dict:
        """
        Train on single batch.
        
        Args:
            batch_states: Batch of board states
            batch_moves: Batch of move indices
            batch_rewards: Batch of move rewards
            batch_outcomes: Batch of game outcomes
            
        Returns:
            Loss dictionary
        """
        batch_states = batch_states.to(self.device)
        batch_moves = batch_moves.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_outcomes = batch_outcomes.to(self.device)
        
        # Forward pass
        policy_logits, values = self.model(batch_states)
        
        # Policy loss: cross-entropy for move selection
        policy_loss = nn.CrossEntropyLoss()(policy_logits, batch_moves)
        
        # Value loss: MSE for outcome prediction
        value_loss = nn.MSELoss()(values.squeeze(-1), batch_outcomes)
        
        # Reward-weighted loss (good moves have higher reward)
        reward_weights = torch.abs(batch_rewards).clamp(min=0.1, max=1.0)
        weighted_policy_loss = policy_loss * reward_weights.mean()
        
        # Total loss with multi-task weighting
        total_loss = weighted_policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Record losses
        self.losses["policy"].append(policy_loss.item())
        self.losses["value"].append(value_loss.item())
        self.losses["total"].append(total_loss.item())
        self.training_step += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def train_epoch(self, train_loader: DataLoader, epochs: int = 1) -> Dict:
        """
        Train for one or more epochs.
        
        Args:
            train_loader: Training data loader
            epochs: Number of epochs
            
        Returns:
            Training statistics
        """
        self.model.train()
        epoch_losses = {"policy": [], "value": [], "total": []}
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                batch_states = batch["position"]
                batch_moves = batch["move"].squeeze(-1)
                batch_rewards = batch["reward"].squeeze(-1)
                batch_outcomes = batch["outcome"].squeeze(-1)
                
                losses = self.train_on_batch(batch_states, batch_moves, batch_rewards, batch_outcomes)
                
                epoch_losses["policy"].append(losses["policy_loss"])
                epoch_losses["value"].append(losses["value_loss"])
                epoch_losses["total"].append(losses["total_loss"])
                
                if batch_idx % 100 == 0:
                    print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx}: Loss {losses['total_loss']:.4f}")
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            "avg_policy_loss": np.mean(epoch_losses["policy"]),
            "avg_value_loss": np.mean(epoch_losses["value"]),
            "avg_total_loss": np.mean(epoch_losses["total"])
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_moves = 0
        total_moves = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch_states = batch["position"].to(self.device)
                batch_moves = batch["move"].squeeze(-1).to(self.device)
                batch_outcomes = batch["outcome"].squeeze(-1).to(self.device)
                
                policy_logits, values = self.model(batch_states)
                
                loss = nn.CrossEntropyLoss()(policy_logits, batch_moves)
                total_loss += loss.item()
                
                # Move accuracy (top-1)
                predictions = policy_logits.argmax(dim=1)
                correct_moves += (predictions == batch_moves).sum().item()
                total_moves += batch_moves.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_moves / total_moves if total_moves > 0 else 0
        
        return {
            "val_loss": avg_loss,
            "move_accuracy": accuracy
        }
    
    def save_checkpoint(self, filepath: str, metadata: Dict = None):
        """Save training checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_step": self.training_step,
            "losses": {k: list(v) for k, v in self.losses.items()},
            "metadata": metadata or {}
        }
        
        torch.save(checkpoint, filepath)
        print(f"[INFO] Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_step = checkpoint["training_step"]
        
        print(f"[INFO] Loaded checkpoint: {filepath}")


class CurriculumLearner:
    """Curriculum learning scheduler."""
    
    def __init__(self, total_steps: int):
        """
        Initialize curriculum learner.
        
        Args:
            total_steps: Total training steps planned
        """
        self.total_steps = total_steps
        self.current_step = 0
    
    def get_difficulty(self) -> float:
        """
        Get current difficulty level (0.0 = easy, 1.0 = hard).
        Gradually increases from 0 to 1 over training.
        """
        return min(1.0, self.current_step / self.total_steps)
    
    def step(self):
        """Advance one step."""
        self.current_step += 1
    
    def get_batch_difficulty_filter(self, batch_difficulties: np.ndarray) -> np.ndarray:
        """
        Filter batch to include only appropriate difficulty levels.
        
        Args:
            batch_difficulties: Array of position difficulties
            
        Returns:
            Boolean mask for appropriate positions
        """
        max_difficulty = self.get_difficulty()
        return batch_difficulties <= max_difficulty


class TrainingPipeline:
    """Complete training pipeline."""
    
    def __init__(self, model, device: str = "cuda"):
        """Initialize training pipeline."""
        self.model = model
        self.device = device
        self.trainer = AdvancedTrainer(model, device)
        self.curriculum = CurriculumLearner(total_steps=100000)
    
    def train(self, games_file: str, num_epochs: int = 10, 
             batch_size: int = 32, validation_split: float = 0.1) -> Dict:
        """
        Complete training pipeline.
        
        Args:
            games_file: Path to games JSONL file
            num_epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation set fraction
            
        Returns:
            Training results
        """
        print(f"\n[INFO] Starting training on {games_file}")
        print(f"[INFO] Epochs: {num_epochs}, Batch size: {batch_size}")
        
        # Load dataset
        dataset = GameExperienceDataset(games_file)
        
        if len(dataset) == 0:
            print("[ERROR] No data loaded!")
            return {}
        
        # Split into train/validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        print(f"[INFO] Train set: {train_size}, Validation set: {val_size}")
        
        # Training loop
        results = {
            "train_losses": [],
            "val_metrics": [],
            "epochs": []
        }
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Train
            train_stats = self.trainer.train_epoch(train_loader, epochs=1)
            print(f"Train - Policy: {train_stats['avg_policy_loss']:.4f}, "
                  f"Value: {train_stats['avg_value_loss']:.4f}, "
                  f"Total: {train_stats['avg_total_loss']:.4f}")
            
            # Validate
            val_stats = self.trainer.evaluate(val_loader)
            print(f"Val - Loss: {val_stats['val_loss']:.4f}, "
                  f"Move Accuracy: {val_stats['move_accuracy']:.2%}")
            
            results["train_losses"].append(train_stats)
            results["val_metrics"].append(val_stats)
            results["epochs"].append(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
                self.trainer.save_checkpoint(checkpoint_path, {
                    "epoch": epoch + 1,
                    "train_loss": train_stats["avg_total_loss"],
                    "val_loss": val_stats["val_loss"],
                    "val_accuracy": val_stats["move_accuracy"]
                })
        
        print(f"\n[INFO] Training complete!")
        
        return results
