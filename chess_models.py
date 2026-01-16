"""
Enhanced neural network architecture for WCCC-level chess play.
Implements residual blocks, batch normalization, and multi-head attention.
Supports both policy (move selection) and value (position evaluation) predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """
        Initialize residual block.
        
        Args:
            hidden_size: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out = out + residual  # Residual connection
        out = self.relu(out)
        
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Attention output (batch_size, seq_len, hidden_size)
        """
        batch_size = x.shape[0]
        
        # Linear transformations
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, -1, self.hidden_size)
        
        # Final linear transformation
        output = self.fc_out(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Self-attention transformer encoder layer."""
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ChessNetV2(nn.Module):
    """
    Enhanced chess neural network with residual blocks and attention.
    Supports both move prediction and position evaluation.
    """
    
    def __init__(self, 
                 input_size: int = 832, # 8x8x13 (with extra features)
                 hidden_size: int = 512,
                 num_residual_blocks: int = 8,
                 num_heads: int = 8,
                 num_policies: int = 4672,
                 dropout: float = 0.1):
        """
        Initialize enhanced chess network.
        
        Args:
            input_size: Input board encoding size
            hidden_size: Hidden layer dimension
            num_residual_blocks: Number of residual blocks
            num_heads: Number of attention heads
            num_policies: Number of possible moves (policy output size)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_policies = num_policies
        
        # Input layer with batch norm
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout)
            for _ in range(num_residual_blocks)
        ])
        
        # Attention layer (Transformer Encoder)
        self.attention = TransformerEncoderLayer(hidden_size, num_heads, dropout)
        
        # Policy head (move selection)
        self.policy_layer1 = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_bn = nn.BatchNorm1d(hidden_size // 2)
        self.policy_dropout = nn.Dropout(dropout)
        self.policy_output = nn.Linear(hidden_size // 2, num_policies)
        
        # Value head (position evaluation)
        self.value_layer1 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_bn = nn.BatchNorm1d(hidden_size // 2)
        self.value_dropout = nn.Dropout(dropout)
        self.value_layer2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.value_output = nn.Linear(hidden_size // 4, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_size)
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Input projection
        x = self.input_projection(x)
        x = self.input_bn(x)
        x = F.relu(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Attention Layer
        x = x.unsqueeze(1) # Add sequence dimension
        x = self.attention(x)
        x = x.squeeze(1) # Remove sequence dimension
        
        # Policy head
        policy = self.policy_layer1(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = self.policy_dropout(policy)
        policy = self.policy_output(policy)  # (batch_size, num_policies)
        
        # Value head
        value = self.value_layer1(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = self.value_dropout(value)
        value = self.value_layer2(value)
        value = F.relu(value)
        value = self.value_output(value)  # (batch_size, 1)
        value = torch.tanh(value)  # Bound to [-1, 1]
        
        return policy, value


class SimpleChessNet(nn.Module):
    """
    Backward-compatible simple neural network for chess.
    Kept for compatibility with existing code.
    """
    
    def __init__(self, input_size: int = 768, hidden_size: int = 512):
        """
        Initialize simple network.
        
        Args:
            input_size: Input board encoding size
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
