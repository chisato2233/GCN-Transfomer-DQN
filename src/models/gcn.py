"""
Graph Convolutional Network modules for spatial feature extraction.

This module implements:
- LocalGCN: Local graph convolution on current node and neighbors
- GCNEncoder: Full graph encoder (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class LocalGCN(nn.Module):
    """
    Local Graph Convolutional Network.
    
    Processes current node and its neighbors using attention-based aggregation.
    This is more efficient than full-graph GCN for routing decisions.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for neighbor aggregation
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                current_node_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            current_node_features: [batch, node_feature_dim]
            neighbor_features: [batch, max_neighbors, node_feature_dim]
            neighbor_mask: [batch, max_neighbors]
            
        Returns:
            Local graph features [batch, output_dim]
        """
        batch_size = current_node_features.size(0)
        
        # Encode current node
        current_encoded = self.node_encoder(current_node_features)  # [batch, hidden_dim]
        
        # Encode neighbors
        neighbor_encoded = self.node_encoder(
            neighbor_features.view(-1, self.node_feature_dim)
        ).view(batch_size, self.max_neighbors, self.hidden_dim)
        
        # Compute attention scores
        current_expanded = current_encoded.unsqueeze(1).expand(-1, self.max_neighbors, -1)
        attention_input = torch.cat([current_expanded, neighbor_encoded], dim=-1)
        attention_scores = self.attention(attention_input).squeeze(-1)  # [batch, max_neighbors]
        
        # Apply mask (set invalid neighbors to -inf)
        attention_scores = attention_scores.masked_fill(neighbor_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Handle case where all neighbors are masked
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        # Aggregate neighbor features
        aggregated = torch.bmm(
            attention_weights.unsqueeze(1),
            neighbor_encoded
        ).squeeze(1)  # [batch, hidden_dim]
        
        # Combine current and aggregated features
        combined = torch.cat([current_encoded, aggregated], dim=-1)
        output = self.output_proj(combined)
        
        return output


class GCNEncoder(nn.Module):
    """
    Full Graph Convolutional Network encoder.
    
    For encoding the entire network topology (optional, not used in local routing).
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # GCN layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(self.node_feature_dim, self.hidden_dim))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        # Output layer
        if self.num_layers > 1:
            self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adjacency matrix.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            adj: Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Normalize adjacency
        deg = adj.sum(dim=1, keepdim=True)
        deg = torch.where(deg > 0, deg, torch.ones_like(deg))
        adj_norm = adj / deg
        
        # Apply GCN layers
        h = x
        for i, layer in enumerate(self.layers):
            h = torch.matmul(adj_norm, h)  # Aggregate
            h = layer(h)  # Transform
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout_layer(h)
        
        return h
