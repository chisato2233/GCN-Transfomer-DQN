"""
Graph Convolutional Network (GCN) encoder for SAGIN topology.

Captures the structural relationships between nodes in the network graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from typing import Optional, List, Tuple


class GCNLayer(nn.Module):
    """Single GCN layer with optional normalization and dropout."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 activation: str = "relu"):
        """
        Initialize GCN layer.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function name
        """
        super().__init__()

        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'none': nn.Identity()
        }
        self.activation = activations.get(activation, nn.ReLU())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GCNEncoder(nn.Module):
    """
    Multi-layer GCN encoder for graph-structured data.

    Aggregates node features based on graph topology.
    """

    def __init__(self, config: dict):
        """
        Initialize GCN encoder.

        Args:
            config: Configuration dictionary with:
                - node_feature_dim: Input node feature dimension
                - hidden_dim: Hidden layer dimension
                - output_dim: Output embedding dimension
                - num_layers: Number of GCN layers
                - dropout: Dropout rate
                - activation: Activation function
                - use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.activation = config.get('activation', 'relu')
        self.use_batch_norm = config.get('use_batch_norm', True)

        # Build layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GCNLayer(
            self.node_feature_dim, self.hidden_dim,
            self.dropout, self.use_batch_norm, self.activation
        ))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(GCNLayer(
                self.hidden_dim, self.hidden_dim,
                self.dropout, self.use_batch_norm, self.activation
            ))

        # Output layer (no dropout/activation on last layer)
        if self.num_layers > 1:
            self.layers.append(GCNLayer(
                self.hidden_dim, self.output_dim,
                0.0, False, 'none'
            ))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [N, node_feature_dim]
            edge_index: Edge indices [2, E]
            batch: Batch assignment [N] (optional)

        Returns:
            Node embeddings [N, output_dim]
        """
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def get_graph_embedding(self,
                           x: torch.Tensor,
                           edge_index: torch.Tensor,
                           batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get graph-level embedding by pooling node embeddings.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment

        Returns:
            Graph embedding [batch_size, output_dim]
        """
        node_embeddings = self.forward(x, edge_index, batch)

        if batch is not None:
            return global_mean_pool(node_embeddings, batch)
        else:
            return node_embeddings.mean(dim=0, keepdim=True)

    def get_node_embedding(self,
                          x: torch.Tensor,
                          edge_index: torch.Tensor,
                          node_idx: int) -> torch.Tensor:
        """Get embedding for a specific node."""
        node_embeddings = self.forward(x, edge_index)
        return node_embeddings[node_idx]


class LocalGCN(nn.Module):
    """
    Local GCN for processing current node and its neighbors.

    More efficient than full graph GCN when only local information is needed.
    """

    def __init__(self, config: dict):
        """
        Initialize local GCN.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        # Node encoder with LayerNorm for stability
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        # Neighbor aggregation with attention
        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Layer norm after attention
        self.attn_norm = nn.LayerNorm(self.hidden_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
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
                current_node: torch.Tensor,
                neighbor_nodes: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for local graph.

        Args:
            current_node: Current node features [batch, node_feature_dim]
            neighbor_nodes: Neighbor features [batch, max_neighbors, node_feature_dim]
            neighbor_mask: Valid neighbor mask [batch, max_neighbors]

        Returns:
            Local graph embedding [batch, output_dim]
        """
        batch_size = current_node.size(0)

        # Encode current node
        current_emb = self.node_encoder(current_node)  # [batch, hidden_dim]

        # Encode neighbor nodes
        neighbor_flat = neighbor_nodes.view(-1, self.node_feature_dim)
        neighbor_emb = self.node_encoder(neighbor_flat)
        neighbor_emb = neighbor_emb.view(batch_size, self.max_neighbors, self.hidden_dim)

        # Check if any sample has at least one valid neighbor
        has_neighbors = neighbor_mask.sum(dim=1) > 0  # [batch]

        # Initialize neighbor_agg with zeros (fallback for samples with no neighbors)
        neighbor_agg = torch.zeros(batch_size, self.hidden_dim, device=current_node.device)

        if has_neighbors.any():
            # Process only samples with valid neighbors
            valid_indices = has_neighbors.nonzero(as_tuple=True)[0]

            # Extract valid samples
            valid_current = current_emb[valid_indices].unsqueeze(1)  # [valid_batch, 1, hidden_dim]
            valid_neighbors = neighbor_emb[valid_indices]  # [valid_batch, max_neighbors, hidden_dim]
            valid_mask = ~neighbor_mask[valid_indices].bool()  # [valid_batch, max_neighbors] True = masked

            # Attention for valid samples
            valid_agg, _ = self.neighbor_attention(
                valid_current, valid_neighbors, valid_neighbors,
                key_padding_mask=valid_mask
            )
            valid_agg = valid_agg.squeeze(1)  # [valid_batch, hidden_dim]
            valid_agg = self.attn_norm(valid_agg)

            # Place back into full tensor
            neighbor_agg[valid_indices] = valid_agg

        # For samples with no neighbors, use current node embedding as fallback
        no_neighbor_indices = (~has_neighbors).nonzero(as_tuple=True)[0]
        if len(no_neighbor_indices) > 0:
            neighbor_agg[no_neighbor_indices] = current_emb[no_neighbor_indices]

        # Combine and project
        combined = torch.cat([current_emb, neighbor_agg], dim=-1)
        output = self.output_proj(combined)

        return output

    def forward_with_attention(self,
                               current_node: torch.Tensor,
                               neighbor_nodes: torch.Tensor,
                               neighbor_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning attention weights for interpretability."""
        batch_size = current_node.size(0)

        current_emb = self.node_encoder(current_node)
        neighbor_flat = neighbor_nodes.view(-1, self.node_feature_dim)
        neighbor_emb = self.node_encoder(neighbor_flat)
        neighbor_emb = neighbor_emb.view(batch_size, self.max_neighbors, self.hidden_dim)

        # Check if any sample has at least one valid neighbor
        has_neighbors = neighbor_mask.sum(dim=1) > 0

        neighbor_agg = torch.zeros(batch_size, self.hidden_dim, device=current_node.device)
        attn_weights = torch.zeros(batch_size, self.max_neighbors, device=current_node.device)

        if has_neighbors.any():
            valid_indices = has_neighbors.nonzero(as_tuple=True)[0]
            valid_current = current_emb[valid_indices].unsqueeze(1)
            valid_neighbors = neighbor_emb[valid_indices]
            valid_mask = ~neighbor_mask[valid_indices].bool()

            valid_agg, valid_attn = self.neighbor_attention(
                valid_current, valid_neighbors, valid_neighbors,
                key_padding_mask=valid_mask
            )
            valid_agg = valid_agg.squeeze(1)
            valid_agg = self.attn_norm(valid_agg)

            neighbor_agg[valid_indices] = valid_agg
            attn_weights[valid_indices] = valid_attn.squeeze(1)

        no_neighbor_indices = (~has_neighbors).nonzero(as_tuple=True)[0]
        if len(no_neighbor_indices) > 0:
            neighbor_agg[no_neighbor_indices] = current_emb[no_neighbor_indices]

        combined = torch.cat([current_emb, neighbor_agg], dim=-1)
        output = self.output_proj(combined)

        return output, attn_weights


class GATEncoder(nn.Module):
    """
    Graph Attention Network encoder (alternative to GCN).

    Uses attention mechanism to weight neighbor contributions.
    """

    def __init__(self, config: dict):
        """Initialize GAT encoder."""
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.output_dim = config.get('output_dim', 64)
        self.num_heads = config.get('num_heads', 4)
        self.dropout = config.get('dropout', 0.1)

        # First GAT layer with multi-head attention
        self.conv1 = GATConv(
            self.node_feature_dim,
            self.hidden_dim // self.num_heads,
            heads=self.num_heads,
            dropout=self.dropout,
            concat=True
        )

        # Second GAT layer
        self.conv2 = GATConv(
            self.hidden_dim,
            self.output_dim,
            heads=1,
            concat=False,
            dropout=self.dropout
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass."""
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout_layer(x)
        x = self.conv2(x, edge_index)
        return x


class NodeEmbedding(nn.Module):
    """
    Simple node embedding without graph structure.

    Used as a baseline or when graph information is not available.
    """

    def __init__(self, config: dict):
        """Initialize node embedding."""
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.output_dim = config.get('output_dim', 64)

        self.encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.encoder(x)
