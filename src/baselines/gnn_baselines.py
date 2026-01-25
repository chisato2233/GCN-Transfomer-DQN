"""
GNN-based DRL Baselines for routing.

References:
1. GAT-DRL: Graph Attention Network + DRL
   - "Graph Neural Networks for Routing Optimization" (MDPI 2024)
   - "GNN Enhanced Dynamic Routing for UAV-Assisted DTN" (ACM 2024)

2. GraphSAGE-DQN: GraphSAGE + DQN
   - "LEO Satellite Network Routing Based on GNN and DQN" (MDPI 2024)
   - https://www.mdpi.com/2076-3417/14/9/3840

3. GCN-DQN: Standard GCN + DQN
   - "GCN and DRL for Intelligent Edge Routing" (Computer Communications 2025)

4. Dueling DQN: Without GNN
   - "Dueling DQN Routing in SDN" (Wireless Networks 2024)

这些是2024-2025年SAGIN/GNN路由领域的主流方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import random


# ============================================================
# GAT-DRL Baseline (Graph Attention Network + DQN)
# ============================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer.

    Reference: Velickovic et al. (2018) "Graph Attention Networks"
    Used in routing: "GNN for Routing Optimization" (2024)
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 4, dropout: float = 0.1,
                 concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        # Linear transformation for each head
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)

        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, num_nodes, in_features]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]

        Returns:
            Updated node features
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation
        h = self.W(x)  # [batch, num_nodes, out_features * n_heads]
        h = h.view(batch_size, num_nodes, self.n_heads, self.out_features)
        h = h.permute(0, 2, 1, 3)  # [batch, n_heads, num_nodes, out_features]

        # Compute attention scores
        # Self-attention: concat features and apply attention vector
        h_i = h.unsqueeze(3).repeat(1, 1, 1, num_nodes, 1)  # [batch, heads, N, N, F]
        h_j = h.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1)  # [batch, heads, N, N, F]
        concat_features = torch.cat([h_i, h_j], dim=-1)  # [batch, heads, N, N, 2F]

        # Attention coefficients
        e = torch.einsum('bhijf,hf->bhij', concat_features, self.a)
        e = self.leaky_relu(e)

        # Mask with adjacency matrix
        adj_expanded = adj.unsqueeze(1)  # [batch, 1, N, N]
        e = e.masked_fill(adj_expanded == 0, float('-inf'))

        # Softmax over neighbors
        attention = F.softmax(e, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.dropout(attention)

        # Apply attention
        out = torch.einsum('bhij,bhjf->bhif', attention, h)  # [batch, heads, N, F]

        if self.concat:
            out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        else:
            out = out.mean(dim=1)

        return out


class GATDQNNetwork(nn.Module):
    """
    GAT-DQN Network for routing.

    Architecture:
    1. GAT layers to process graph structure
    2. Global pooling to get graph representation
    3. DQN head for Q-values
    """

    def __init__(self, config: dict):
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 8)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_actions = config.get('num_actions', 8)
        self.n_heads = config.get('n_heads', 4)

        # GAT layers
        self.gat1 = GraphAttentionLayer(
            self.node_feature_dim, self.hidden_dim // self.n_heads,
            n_heads=self.n_heads, concat=True
        )
        self.gat2 = GraphAttentionLayer(
            self.hidden_dim, self.hidden_dim // self.n_heads,
            n_heads=self.n_heads, concat=True
        )

        # Current node embedding + global context
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor,
                current_node_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [batch, num_nodes, node_feature_dim]
            adj: [batch, num_nodes, num_nodes]
            current_node_idx: [batch] - index of current node

        Returns:
            Q-values [batch, num_actions]
        """
        # GAT forward
        h = F.elu(self.gat1(node_features, adj))
        h = F.elu(self.gat2(h, adj))  # [batch, num_nodes, hidden_dim]

        # Get current node embedding
        batch_size = h.shape[0]
        current_embedding = h[torch.arange(batch_size), current_node_idx]  # [batch, hidden_dim]

        # Global pooling
        global_embedding = h.mean(dim=1)  # [batch, hidden_dim]

        # Concatenate and compute Q-values
        combined = torch.cat([current_embedding, global_embedding], dim=-1)
        q_values = self.fc(combined)

        return q_values


class GATDQNAgent:
    """GAT-DQN Agent for routing."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.num_actions = config.get('num_actions', 8)

        # Networks
        self.q_network = GATDQNNetwork(config).to(self.device)
        self.target_network = GATDQNNetwork(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('lr', 1e-3))

        # Replay buffer
        self.buffer = deque(maxlen=config.get('buffer_capacity', 10000))
        self.batch_size = config.get('batch_size', 64)

        # Exploration
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # DQN parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)

    def select_action(self, node_features: np.ndarray, adj: np.ndarray,
                      current_node_idx: int, action_mask: np.ndarray,
                      training: bool = True) -> int:
        """Select action using epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            valid_indices = np.where(action_mask > 0)[0]
            return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0

        with torch.no_grad():
            node_features_t = torch.FloatTensor(node_features).unsqueeze(0).to(self.device)
            adj_t = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
            current_idx_t = torch.LongTensor([current_node_idx]).to(self.device)

            q_values = self.q_network(node_features_t, adj_t, current_idx_t)
            q_values = q_values.squeeze(0).cpu().numpy()

            # Mask invalid actions
            q_values[action_mask == 0] = float('-inf')
            return int(np.argmax(q_values))

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def train(self):
        """Train the network."""
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        # Training logic similar to standard DQN
        # Omitted for brevity - follows standard DQN update

    def update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_network.parameters(),
                                        self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================
# GraphSAGE-DQN Baseline
# ============================================================

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer with mean aggregation.

    Reference:
    - Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"
    - "LEO Satellite Network Routing Based on GNN and DQN" (2024)
    """

    def __init__(self, in_features: int, out_features: int, aggregator: str = 'mean'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator

        # Self and neighbor transformations
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.W_neigh = nn.Linear(in_features, out_features, bias=False)

        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [batch, num_nodes, in_features]
            adj: Adjacency matrix [batch, num_nodes, num_nodes]

        Returns:
            Updated features [batch, num_nodes, out_features]
        """
        # Normalize adjacency for mean aggregation
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree

        # Aggregate neighbor features
        neigh_features = torch.bmm(adj_norm, x)  # [batch, num_nodes, in_features]

        # Transform and combine
        h_self = self.W_self(x)
        h_neigh = self.W_neigh(neigh_features)

        out = self.norm(h_self + h_neigh)
        return out


class GraphSAGEDQNNetwork(nn.Module):
    """
    GraphSAGE-DQN Network.

    Uses GraphSAGE for inductive learning on dynamic graphs.
    Suitable for LEO satellite networks with changing topology.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 8)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_actions = config.get('num_actions', 8)

        # GraphSAGE layers
        self.sage1 = GraphSAGELayer(self.node_feature_dim, self.hidden_dim)
        self.sage2 = GraphSAGELayer(self.hidden_dim, self.hidden_dim)

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor,
                current_node_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # GraphSAGE layers
        h = F.relu(self.sage1(node_features, adj))
        h = F.relu(self.sage2(h, adj))

        # Get current node and global representations
        batch_size = h.shape[0]
        current_embedding = h[torch.arange(batch_size), current_node_idx]
        global_embedding = h.mean(dim=1)

        # Q-values
        combined = torch.cat([current_embedding, global_embedding], dim=-1)
        return self.q_head(combined)


# ============================================================
# Dueling DQN Baseline (No GNN)
# ============================================================

class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network without GNN.

    Reference: "Dueling DQN Routing in SDN" (Wireless Networks 2024)

    Architecture separates value and advantage streams.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.state_dim = config.get('state_dim', 32)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_actions = config.get('num_actions', 8)

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_dim]

        Returns:
            Q-values [batch, num_actions]
        """
        features = self.feature(state)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q_values


class DuelingDQNAgent:
    """Dueling DQN Agent without graph structure."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.num_actions = config.get('num_actions', 8)

        # Networks
        self.q_network = DuelingDQNNetwork(config).to(self.device)
        self.target_network = DuelingDQNNetwork(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('lr', 1e-3))

        # Replay buffer
        self.buffer = deque(maxlen=config.get('buffer_capacity', 10000))
        self.batch_size = config.get('batch_size', 64)

        # Exploration
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # DQN parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray,
                      training: bool = True) -> int:
        """Select action using epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            valid_indices = np.where(action_mask > 0)[0]
            return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).squeeze(0).cpu().numpy()
            q_values[action_mask == 0] = float('-inf')
            return int(np.argmax(q_values))

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================
# Simple GCN-DQN Baseline (Standard GCN)
# ============================================================

class GCNLayer(nn.Module):
    """Standard Graph Convolutional Layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_nodes, in_features]
            adj: [batch, num_nodes, num_nodes]
        """
        # Normalize adjacency (add self-loops and degree normalization)
        adj = adj + torch.eye(adj.shape[1], device=adj.device).unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / degree.sqrt() / degree.sqrt().transpose(-1, -2)

        # GCN operation: A * X * W
        support = self.linear(x)
        out = torch.bmm(adj_norm, support)
        return out


class GCNDQNNetwork(nn.Module):
    """Standard GCN-DQN Network."""

    def __init__(self, config: dict):
        super().__init__()

        self.node_feature_dim = config.get('node_feature_dim', 8)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_actions = config.get('num_actions', 8)

        # GCN layers
        self.gcn1 = GCNLayer(self.node_feature_dim, self.hidden_dim)
        self.gcn2 = GCNLayer(self.hidden_dim, self.hidden_dim)

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor,
                current_node_idx: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(node_features, adj))
        h = F.relu(self.gcn2(h, adj))

        batch_size = h.shape[0]
        current_embedding = h[torch.arange(batch_size), current_node_idx]
        global_embedding = h.mean(dim=1)

        combined = torch.cat([current_embedding, global_embedding], dim=-1)
        return self.q_head(combined)
