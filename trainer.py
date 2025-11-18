"""Training module for the chess engine."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import LEARNING_RATE, GAMMA, BATCH_SIZE, USE_CUDA


class ChessDataset(Dataset):
    """Dataset for chess training data."""
    
    def __init__(self, states, actions, returns, advantages, old_log_probs):
        """
        Initialize dataset.
        
        Args:
            states: List of board states
            actions: List of actions taken
            returns: List of discounted returns
            advantages: List of advantages
            old_log_probs: List of log probabilities from old policy
        """
        self.states = states
        self.actions = actions
        self.returns = returns
        self.advantages = advantages
        self.old_log_probs = old_log_probs
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        elif state.dtype != torch.float32:
            state = state.float()
        
        return {
            'state': state,
            'action': torch.tensor(self.actions[idx], dtype=torch.long),
            'return': torch.tensor(self.returns[idx], dtype=torch.float32),
            'advantage': torch.tensor(self.advantages[idx], dtype=torch.float32),
            'old_log_prob': torch.tensor(self.old_log_probs[idx], dtype=torch.float32)
        }


class ChessTrainer:
    """
    Trainer for the chess engine using policy gradient / PPO.
    """
    
    def __init__(self, model, learning_rate=LEARNING_RATE, device=None):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            learning_rate: Learning rate for optimizer
            device: Torch device (CPU or CUDA)
        """
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss tracking
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []
        
        # Accuracy tracking
        self.win_rates = []
        self.draw_rates = []
        self.loss_rates = []
    
    def train_epoch(self, training_data, batch_size=BATCH_SIZE, ppo_epochs=4, clip_epsilon=0.2):
        """
        Train the model for one epoch using PPO.
        
        Args:
            training_data: Dictionary with training data
            batch_size: Batch size for training
            ppo_epochs: Number of PPO epochs
            clip_epsilon: Clipping parameter for PPO
            
        Returns:
            Dictionary with loss statistics
        """
        states = training_data['states']
        actions = training_data['actions']
        returns = training_data['returns']
        advantages = training_data['advantages']
        old_log_probs = training_data['log_probs']
        
        if len(states) == 0:
            print("No training data available")
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        print(f"Training on {len(states)} samples...")
        
        # Create dataset and dataloader (num_workers=0 to avoid multiprocessing issues on Windows)
        dataset = ChessDataset(states, actions, returns, advantages, old_log_probs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_total_losses = []
        
        self.model.train()
        
        # PPO training loop
        for ppo_epoch in range(ppo_epochs):
            for batch in dataloader:
                # Move batch to device - DataLoader already stacks tensors
                batch_states = batch['state'].to(self.device)
                batch_actions = batch['action'].to(self.device)
                batch_returns = batch['return'].to(self.device)
                batch_advantages = batch['advantage'].to(self.device)
                batch_old_log_probs = batch['old_log_prob'].to(self.device)
                
                # Forward pass
                policy_logits, values = self.model(batch_states)
                values = values.squeeze(-1)
                
                # Get log probabilities for taken actions
                log_probs = torch.log_softmax(policy_logits, dim=-1)
                action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                
                # PPO clipped objective
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                # Track losses
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_total_losses.append(total_loss.item())
        
        # Calculate average losses
        avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0
        avg_value_loss = np.mean(epoch_value_losses) if epoch_value_losses else 0
        avg_total_loss = np.mean(epoch_total_losses) if epoch_total_losses else 0
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.total_losses.append(avg_total_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss
        }
    
    def save_checkpoint(self, filepath, epoch, additional_info=None):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            additional_info: Additional information to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'total_losses': self.total_losses,
            'win_rates': self.win_rates,
            'draw_rates': self.draw_rates,
            'loss_rates': self.loss_rates,
        }
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.total_losses = checkpoint.get('total_losses', [])
        self.win_rates = checkpoint.get('win_rates', [])
        self.draw_rates = checkpoint.get('draw_rates', [])
        self.loss_rates = checkpoint.get('loss_rates', [])
        
        print(f"Checkpoint loaded from {filepath}")
        
        return checkpoint
    
    def get_loss_history(self):
        """Get loss and accuracy history."""
        return {
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'total_losses': self.total_losses,
            'win_rates': self.win_rates,
            'draw_rates': self.draw_rates,
            'loss_rates': self.loss_rates,
        }
    
    def record_game_results(self, wins, draws, losses):
        """Record game results for this epoch."""
        total = wins + draws + losses
        if total > 0:
            self.win_rates.append(100.0 * wins / total)
            self.draw_rates.append(100.0 * draws / total)
            self.loss_rates.append(100.0 * losses / total)
        else:
            self.win_rates.append(0.0)
            self.draw_rates.append(0.0)
            self.loss_rates.append(0.0)
