"""
Feature fusion modules for combining GCN and Transformer outputs.

Provides different strategies for fusing spatial (GCN) and temporal (Transformer) features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConcatFusion(nn.Module):
    """Simple concatenation-based feature fusion."""

    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize concatenation fusion.

        Args:
            gcn_dim: GCN output dimension
            temporal_dim: Temporal encoder output dimension
            state_dim: State encoder output dimension
            output_dim: Final output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        total_dim = gcn_dim + temporal_dim + state_dim

        # Build fusion MLP
        layers = []
        prev_dim = total_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.fusion = nn.Sequential(*layers)

    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:
        """
        Fuse features via concatenation.

        Args:
            gcn_features: GCN output [batch, gcn_dim]
            temporal_features: Transformer output [batch, temporal_dim]
            current_state: Current state encoding [batch, state_dim]

        Returns:
            Fused features [batch, output_dim]
        """
        combined = torch.cat([gcn_features, temporal_features, current_state], dim=-1)
        return self.fusion(combined)


class AttentionFusion(nn.Module):
    """Attention-based feature fusion."""

    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Initialize attention fusion.

        Args:
            gcn_dim: GCN output dimension
            temporal_dim: Temporal encoder output dimension
            state_dim: State encoder output dimension
            output_dim: Final output dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = max(gcn_dim, temporal_dim, state_dim)

        # Project all features to same dimension
        self.gcn_proj = nn.Linear(gcn_dim, self.d_model)
        self.temporal_proj = nn.Linear(temporal_dim, self.d_model)
        self.state_proj = nn.Linear(state_dim, self.d_model)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, output_dim),
            nn.ReLU()
        )

    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:
        """
        Fuse features via attention.

        Args:
            gcn_features: GCN output [batch, gcn_dim]
            temporal_features: Transformer output [batch, temporal_dim]
            current_state: Current state encoding [batch, state_dim]

        Returns:
            Fused features [batch, output_dim]
        """
        batch_size = gcn_features.size(0)

        # Project to common dimension
        gcn_proj = self.gcn_proj(gcn_features).unsqueeze(1)
        temporal_proj = self.temporal_proj(temporal_features).unsqueeze(1)
        state_proj = self.state_proj(current_state).unsqueeze(1)

        # Stack as sequence [batch, 3, d_model]
        features = torch.cat([gcn_proj, temporal_proj, state_proj], dim=1)

        # Self-attention
        attn_out, attn_weights = self.attention(features, features, features)

        # Residual connection and normalization
        features = self.norm(features + attn_out)

        # Pool across feature types (mean pooling)
        fused = features.mean(dim=1)  # [batch, d_model]

        return self.output_proj(fused)


class GatedFusion(nn.Module):
    """Gated feature fusion with learned importance weights."""

    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 dropout: float = 0.1):
        """
        Initialize gated fusion.

        Args:
            gcn_dim: GCN output dimension
            temporal_dim: Temporal encoder output dimension
            state_dim: State encoder output dimension
            output_dim: Final output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = output_dim

        # Projection layers with LayerNorm for stability
        self.gcn_proj = nn.Sequential(
            nn.Linear(gcn_dim, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, self.d_model),
            nn.LayerNorm(self.d_model)
        )

        # Gate networks (compute importance of each feature type)
        concat_dim = self.d_model * 3
        self.gate_gcn = nn.Sequential(
            nn.Linear(concat_dim, self.d_model),
            nn.Sigmoid()
        )
        self.gate_temporal = nn.Sequential(
            nn.Linear(concat_dim, self.d_model),
            nn.Sigmoid()
        )
        self.gate_state = nn.Sequential(
            nn.Linear(concat_dim, self.d_model),
            nn.Sigmoid()
        )

        # Output projection with LayerNorm
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:
        """
        Fuse features via gating mechanism.

        Args:
            gcn_features: GCN output [batch, gcn_dim]
            temporal_features: Transformer output [batch, temporal_dim]
            current_state: Current state encoding [batch, state_dim]

        Returns:
            Fused features [batch, output_dim]
        """
        # Project to common dimension (with LayerNorm)
        h_gcn = self.gcn_proj(gcn_features)
        h_temporal = self.temporal_proj(temporal_features)
        h_state = self.state_proj(current_state)

        # Concatenate for gate computation
        combined = torch.cat([h_gcn, h_temporal, h_state], dim=-1)

        # Compute gates
        g_gcn = self.gate_gcn(combined)
        g_temporal = self.gate_temporal(combined)
        g_state = self.gate_state(combined)

        # Normalize gates (softmax-like)
        g_sum = g_gcn + g_temporal + g_state + 1e-8
        g_gcn = g_gcn / g_sum
        g_temporal = g_temporal / g_sum
        g_state = g_state / g_sum

        # Weighted combination
        fused = g_gcn * h_gcn + g_temporal * h_temporal + g_state * h_state

        return self.output_proj(fused)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion using state as query."""

    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """Initialize cross-attention fusion."""
        super().__init__()

        self.d_model = max(gcn_dim, temporal_dim, state_dim)

        # Projections
        self.gcn_proj = nn.Linear(gcn_dim, self.d_model)
        self.temporal_proj = nn.Linear(temporal_dim, self.d_model)
        self.state_proj = nn.Linear(state_dim, self.d_model)

        # Cross-attention (state queries GCN and temporal)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output layers
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, output_dim),
            nn.ReLU()
        )

    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:
        """Fuse using cross-attention."""
        # Project features
        gcn_proj = self.gcn_proj(gcn_features).unsqueeze(1)
        temporal_proj = self.temporal_proj(temporal_features).unsqueeze(1)
        state_proj = self.state_proj(current_state).unsqueeze(1)

        # Use state as query, others as key/value
        kv = torch.cat([gcn_proj, temporal_proj], dim=1)  # [batch, 2, d_model]

        # Cross-attention
        attn_out, _ = self.cross_attention(state_proj, kv, kv)

        # Residual and output
        fused = state_proj + attn_out
        fused = fused.squeeze(1)

        return self.output_proj(fused)


class FeatureFusion(nn.Module):
    """
    Main feature fusion module with configurable fusion method.

    Wraps different fusion strategies under a common interface.
    """

    def __init__(self,
                 config: dict,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int):
        """
        Initialize feature fusion.

        Args:
            config: Fusion configuration with:
                - method: Fusion method ('concat', 'attention', 'gated', 'cross_attention')
                - output_dim: Output dimension
                - fc_hidden_dim: Hidden dimension for MLP layers
                - num_fc_layers: Number of FC layers (for concat)
            gcn_dim: GCN output dimension
            temporal_dim: Temporal encoder output dimension
            state_dim: State encoder output dimension
        """
        super().__init__()

        self.method = config.get('method', 'concat')
        self.output_dim = config.get('output_dim', 128)
        hidden_dim = config.get('fc_hidden_dim', 256)
        num_layers = config.get('num_fc_layers', 2)
        dropout = config.get('dropout', 0.1)

        # Create fusion module based on method
        if self.method == 'concat':
            self.fusion = ConcatFusion(
                gcn_dim, temporal_dim, state_dim,
                self.output_dim, hidden_dim, num_layers, dropout
            )
        elif self.method == 'attention':
            self.fusion = AttentionFusion(
                gcn_dim, temporal_dim, state_dim,
                self.output_dim, dropout=dropout
            )
        elif self.method == 'gated':
            self.fusion = GatedFusion(
                gcn_dim, temporal_dim, state_dim,
                self.output_dim, dropout
            )
        elif self.method == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                gcn_dim, temporal_dim, state_dim,
                self.output_dim, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                current_state: torch.Tensor) -> torch.Tensor:
        """
        Fuse features.

        Args:
            gcn_features: GCN output [batch, gcn_dim]
            temporal_features: Transformer output [batch, temporal_dim]
            current_state: Current state encoding [batch, state_dim]

        Returns:
            Fused features [batch, output_dim]
        """
        return self.fusion(gcn_features, temporal_features, current_state)


class SimpleFusion(nn.Module):
    """
    Simple two-input fusion (when only using one encoder).

    Useful for ablation studies.
    """

    def __init__(self,
                 input1_dim: int,
                 input2_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128):
        """Initialize simple fusion."""
        super().__init__()

        total_dim = input1_dim + input2_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self,
                input1: torch.Tensor,
                input2: torch.Tensor) -> torch.Tensor:
        """Fuse two inputs."""
        combined = torch.cat([input1, input2], dim=-1)
        return self.fusion(combined)
