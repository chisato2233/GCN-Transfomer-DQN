"""
Feature fusion modules for combining spatial, temporal, and state features.

This module implements various fusion strategies:
- FeatureFusion: Simple concatenation-based fusion
- GatedFusion: Learnable gated fusion mechanism
- AttentionFusion: Attention-based fusion (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FeatureFusion(nn.Module):
    """
    Simple feature fusion by concatenation and projection.
    """
    
    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.gcn_dim = gcn_dim
        self.temporal_dim = temporal_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        total_input_dim = gcn_dim + temporal_dim + state_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
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
    
    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                state_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features from different sources.
        
        Args:
            gcn_features: [batch, gcn_dim]
            temporal_features: [batch, temporal_dim]
            state_features: [batch, state_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        # Concatenate all features
        combined = torch.cat([gcn_features, temporal_features, state_features], dim=-1)
        
        # Project to output dimension
        output = self.fusion(combined)
        
        return output


class GatedFusion(nn.Module):
    """
    Gated feature fusion mechanism.
    
    Learns to weight different feature sources adaptively based on
    the current context. This allows the model to focus on the most
    relevant information source for each decision.
    """
    
    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.gcn_dim = gcn_dim
        self.temporal_dim = temporal_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # Project each feature to same dimension
        self.gcn_proj = nn.Sequential(
            nn.Linear(gcn_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # Gating network
        total_dim = gcn_dim + temporal_dim + state_dim
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 3),  # 3 gates for 3 feature sources
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                state_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features using learned gates.
        
        Args:
            gcn_features: [batch, gcn_dim]
            temporal_features: [batch, temporal_dim]
            state_features: [batch, state_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        # Project features to same dimension
        gcn_proj = self.gcn_proj(gcn_features)  # [batch, output_dim]
        temporal_proj = self.temporal_proj(temporal_features)  # [batch, output_dim]
        state_proj = self.state_proj(state_features)  # [batch, output_dim]
        
        # Compute gates based on all features
        combined_input = torch.cat([gcn_features, temporal_features, state_features], dim=-1)
        gates = self.gate(combined_input)  # [batch, 3]
        
        # Apply gates
        fused = (
            gates[:, 0:1] * gcn_proj +
            gates[:, 1:2] * temporal_proj +
            gates[:, 2:3] * state_proj
        )
        
        # Output projection
        output = self.output_proj(fused)
        
        return output


class AttentionFusion(nn.Module):
    """
    Attention-based feature fusion.
    
    Uses self-attention to learn relationships between different
    feature sources and fuse them accordingly.
    """
    
    def __init__(self,
                 gcn_dim: int,
                 temporal_dim: int,
                 state_dim: int,
                 output_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # Project all features to same dimension
        self.gcn_proj = nn.Linear(gcn_dim, output_dim)
        self.temporal_proj = nn.Linear(temporal_dim, output_dim)
        self.state_proj = nn.Linear(state_dim, output_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(output_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self,
                gcn_features: torch.Tensor,
                temporal_features: torch.Tensor,
                state_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse features using attention.
        
        Args:
            gcn_features: [batch, gcn_dim]
            temporal_features: [batch, temporal_dim]
            state_features: [batch, state_dim]
            
        Returns:
            Fused features [batch, output_dim]
        """
        batch_size = gcn_features.size(0)
        
        # Project features
        gcn_proj = self.gcn_proj(gcn_features)
        temporal_proj = self.temporal_proj(temporal_features)
        state_proj = self.state_proj(state_features)
        
        # Stack as sequence [batch, 3, output_dim]
        features = torch.stack([gcn_proj, temporal_proj, state_proj], dim=1)
        
        # Self-attention
        attended, _ = self.attention(features, features, features)
        attended = self.norm(attended + features)  # Residual connection
        
        # Flatten and project
        flattened = attended.view(batch_size, -1)
        output = self.output_proj(flattened)
        
        return output
