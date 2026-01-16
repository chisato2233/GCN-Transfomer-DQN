"""
Transformer encoder for temporal sequence modeling.

Captures temporal dependencies in the historical state sequence.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self,
                 d_model: int,
                 max_len: int = 100,
                 dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""

    def __init__(self,
                 d_model: int,
                 max_len: int = 100,
                 dropout: float = 0.1):
        """Initialize learnable positional encoding."""
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal sequence processing.

    Processes historical state sequences to capture temporal patterns.
    """

    def __init__(self, config: dict):
        """
        Initialize Transformer encoder.

        Args:
            config: Configuration dictionary with:
                - d_model: Model dimension
                - num_heads: Number of attention heads
                - num_layers: Number of encoder layers
                - d_feedforward: Feedforward dimension
                - dropout: Dropout rate
                - history_length: Expected sequence length
                - max_seq_length: Maximum sequence length
                - positional_encoding: Type of positional encoding
        """
        super().__init__()

        self.d_model = config.get('d_model', 64)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.d_feedforward = config.get('d_feedforward', 256)
        self.dropout = config.get('dropout', 0.1)
        self.history_length = config.get('history_length', 10)
        self.max_seq_length = config.get('max_seq_length', 100)

        # Positional encoding
        pe_type = config.get('positional_encoding', 'sinusoidal')
        if pe_type == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(
                self.d_model, self.max_seq_length, self.dropout
            )
        else:
            self.pos_encoder = LearnablePositionalEncoding(
                self.d_model, self.max_seq_length, self.dropout
            )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Output layer normalization
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len]
            src_key_padding_mask: Padding mask [batch, seq_len]

        Returns:
            Encoded features for the last time step [batch, d_model]
        """
        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Take the last time step's output
        x = x[:, -1, :]  # [batch, d_model]

        # Layer normalization
        x = self.norm(x)

        return x

    def forward_all(self,
                    x: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass returning all time steps.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            mask: Attention mask

        Returns:
            Encoded features for all time steps [batch, seq_len, d_model]
        """
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=mask)
        x = self.norm(x)
        return x

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (autoregressive) attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class StateProjector(nn.Module):
    """Project raw state vectors to Transformer dimension."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        """
        Initialize state projector.

        Args:
            input_dim: Raw state dimension
            d_model: Target dimension (Transformer d_model)
            dropout: Dropout rate
        """
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project state to model dimension."""
        return self.proj(x)


class TemporalEncoder(nn.Module):
    """
    Complete temporal encoder combining projection and Transformer.

    Takes raw state history and produces temporal features.
    """

    def __init__(self, state_dim: int, config: dict):
        """
        Initialize temporal encoder.

        Args:
            state_dim: Raw state dimension
            config: Transformer configuration
        """
        super().__init__()

        self.state_dim = state_dim
        self.d_model = config.get('d_model', 64)

        # State projection
        self.state_projector = StateProjector(
            state_dim, self.d_model,
            dropout=config.get('dropout', 0.1)
        )

        # Transformer encoder
        self.transformer = TemporalTransformer(config)

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Process state history.

        Args:
            state_history: Historical states [batch, history_length, state_dim]

        Returns:
            Temporal features [batch, d_model]
        """
        # Project to model dimension
        x = self.state_projector(state_history)  # [batch, history_length, d_model]

        # Transformer encoding
        temporal_features = self.transformer(x)  # [batch, d_model]

        return temporal_features

    def forward_with_states(self,
                           state_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process state history and return intermediate states.

        Useful for visualization and debugging.
        """
        x = self.state_projector(state_history)
        all_states = self.transformer.forward_all(x)
        final_state = all_states[:, -1, :]
        return final_state, all_states


class TemporalAttentionPooling(nn.Module):
    """
    Attention-based temporal pooling.

    Alternative to using only the last time step.
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        """Initialize temporal attention pooling."""
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )

        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool temporal sequence using attention.

        Args:
            x: Sequence [batch, seq_len, d_model]

        Returns:
            Pooled features [batch, d_model]
        """
        batch_size = x.size(0)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Attention pooling
        pooled, _ = self.attention(query, x, x)

        return pooled.squeeze(1)
