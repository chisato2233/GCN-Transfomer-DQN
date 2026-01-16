"""
Transformer encoder for temporal feature extraction.

This module implements a Transformer encoder that processes
state history sequences to capture temporal patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_length: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_length, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """
    Transformer encoder for temporal state sequences.
    
    Processes a sequence of past states to extract temporal features
    that capture patterns in routing decisions over time.
    """
    
    def __init__(self, state_dim: int, config: dict):
        super().__init__()
        
        self.state_dim = state_dim
        self.d_model = config.get('d_model', 64)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.d_feedforward = config.get('d_feedforward', 128)
        self.dropout = config.get('dropout', 0.1)
        self.history_length = config.get('history_length', 10)
        self.max_seq_length = config.get('max_seq_length', 100)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.d_model, 
            self.max_seq_length, 
            self.dropout
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_feedforward,
            dropout=self.dropout,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Output projection to get fixed-size output
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state_history: Historical states [batch, history_length, state_dim]
            
        Returns:
            Temporal features [batch, d_model]
        """
        batch_size = state_history.size(0)
        
        # Project input to model dimension
        x = self.input_proj(state_history)  # [batch, history_length, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for padding (zeros in history)
        # Detect padding by checking if all values are zero
        padding_mask = (state_history.abs().sum(dim=-1) == 0)  # [batch, history_length]
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Use the last non-padded position's output
        # Or use mean pooling over non-padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        x_masked = x * mask_expanded
        
        # Mean pooling (avoiding division by zero)
        seq_lengths = mask_expanded.sum(dim=1).clamp(min=1)
        x_pooled = x_masked.sum(dim=1) / seq_lengths
        
        # Output projection
        output = self.output_proj(x_pooled)  # [batch, d_model]
        
        return output


class TemporalAttention(nn.Module):
    """
    Simplified temporal attention (alternative to full Transformer).
    
    Uses attention over state history without full Transformer overhead.
    """
    
    def __init__(self, state_dim: int, config: dict):
        super().__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 64)
        
        # Query, Key, Value projections
        self.query = nn.Linear(state_dim, self.hidden_dim)
        self.key = nn.Linear(state_dim, self.hidden_dim)
        self.value = nn.Linear(state_dim, self.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU()
        )
        
    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: [batch, seq_length, state_dim]
            
        Returns:
            Temporal features [batch, output_dim]
        """
        # Use last state as query
        query = self.query(state_history[:, -1, :])  # [batch, hidden_dim]
        keys = self.key(state_history)  # [batch, seq_length, hidden_dim]
        values = self.value(state_history)  # [batch, seq_length, hidden_dim]
        
        # Attention scores
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))  # [batch, 1, seq_length]
        scores = scores / math.sqrt(self.hidden_dim)
        
        # Create mask for padding
        padding_mask = (state_history.abs().sum(dim=-1) == 0)  # [batch, seq_length]
        scores = scores.masked_fill(padding_mask.unsqueeze(1), float('-inf'))
        
        # Attention weights
        weights = F.softmax(scores, dim=-1)
        
        # Handle all-padding case
        weights = torch.where(
            torch.isnan(weights),
            torch.zeros_like(weights),
            weights
        )
        
        # Weighted sum
        context = torch.bmm(weights, values).squeeze(1)  # [batch, hidden_dim]
        
        # Output
        output = self.output_proj(context)
        
        return output
