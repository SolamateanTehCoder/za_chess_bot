"""
Training module for bullet chess model.
Only trains after model reaches 100% accuracy (wins all games against itself).
Uses simple policy gradient with time-based reward signals.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
from pathlib import Path
import chess
import chess.engine


class SimpleChessNet(nn.Module):
    """Simple neural network for chess move selection."""
    
    def __init__(self, input_size: int = 768, hidden_size: int = 512):
        """
        Initialize network.
        
        Args:
            input_size: Board state encoding size (8x8x12)
            hidden_size: Hidden layer size
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Policy head (move selection)
        self.policy = nn.Linear(hidden_size, 4672)  # All possible moves
        
        # Value head (game outcome prediction)
        self.value = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """Forward pass."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        policy = self.policy(x)
        value = torch.tanh(self.value(x))
        
        return policy, value


class ChessTrainer:
    """Trainer for chess model."""
    
    def __init__(self, model, device='cuda', learning_rate=0.001):
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
        self.policy_losses = []
        self.value_losses = []
    
    def train_on_games(self, game_experiences: list, batch_size: int = 32, epochs: int = 4):
        """
        Train on collected game experiences.
        
        Args:
            game_experiences: List of game experience dicts
            batch_size: Batch size for training
            epochs: Number of training epochs
            
        Returns:
            Dict with loss statistics
        """
        if not game_experiences:
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        # Collect all moves and rewards
        states = []
        actions = []
        rewards = []
        
        for game in game_experiences:
            for move_exp in game.get('experiences', []):
                # Placeholder: in real implementation, encode board state
                states.append(np.zeros(768, dtype=np.float32))
                actions.append(0)  # Placeholder action index
                rewards.append(move_exp['reward'])
        
        if not states:
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        
        # Normalize rewards
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        advantages = (rewards - reward_mean) / reward_std
        
        epoch_policy_losses = []
        epoch_value_losses = []
        
        self.model.train()
        
        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                batch_advantages = advantages[i:i+batch_size]
                batch_rewards = rewards[i:i+batch_size]
                
                # Forward pass
                policy_logits, values = self.model(batch_states)
                values = values.squeeze(-1)
                
                # Policy loss
                log_probs = torch.log_softmax(policy_logits, dim=-1)
                action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                policy_loss = -(action_log_probs * batch_advantages).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, batch_rewards)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
        
        avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0
        avg_value_loss = np.mean(epoch_value_losses) if epoch_value_losses else 0
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_policy_loss + 0.5 * avg_value_loss
        }
    
    def save_checkpoint(self, filepath: str, metadata: dict = None):
        """Save model checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metadata': metadata or {}
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
