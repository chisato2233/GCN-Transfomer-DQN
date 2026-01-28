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

SAGIN-Compatible Version:
- 适配 SAGIN 环境的 observation 格式
- 使用 neighbor_topology_features [max_neighbors, 14]
- 使用 neighbor_mask [max_neighbors]
- 使用 action_mask [max_neighbors]
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
# SAGIN-Compatible Replay Buffer
# ============================================================

class GNNReplayBuffer:
    """Replay buffer for GNN-based agents in SAGIN environment."""

    def __init__(self,
                 capacity: int,
                 feature_dim: int,
                 max_neighbors: int,
                 device: str = 'cuda'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Allocate buffers
        self.neighbor_features = np.zeros(
            (capacity, max_neighbors, feature_dim), dtype=np.float32
        )
        self.neighbor_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_neighbor_features = np.zeros(
            (capacity, max_neighbors, feature_dim), dtype=np.float32
        )
        self.next_neighbor_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)

    def push(self, neighbor_features, neighbor_mask, action, reward,
             next_neighbor_features, next_neighbor_mask, done,
             action_mask, next_action_mask):
        idx = self.position

        self.neighbor_features[idx] = neighbor_features
        self.neighbor_masks[idx] = neighbor_mask
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_neighbor_features[idx] = next_neighbor_features
        self.next_neighbor_masks[idx] = next_neighbor_mask
        self.dones[idx] = float(done)
        self.action_masks[idx] = action_mask
        self.next_action_masks[idx] = next_action_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'neighbor_features': torch.tensor(
                self.neighbor_features[indices], device=self.device
            ),
            'neighbor_masks': torch.tensor(
                self.neighbor_masks[indices], device=self.device
            ),
            'actions': torch.tensor(
                self.actions[indices], device=self.device
            ).unsqueeze(1),
            'rewards': torch.tensor(
                self.rewards[indices], device=self.device
            ).unsqueeze(1),
            'next_neighbor_features': torch.tensor(
                self.next_neighbor_features[indices], device=self.device
            ),
            'next_neighbor_masks': torch.tensor(
                self.next_neighbor_masks[indices], device=self.device
            ),
            'dones': torch.tensor(
                self.dones[indices], device=self.device
            ).unsqueeze(1),
            'action_masks': torch.tensor(
                self.action_masks[indices], device=self.device
            ),
            'next_action_masks': torch.tensor(
                self.next_action_masks[indices], device=self.device
            ),
        }

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def __len__(self) -> int:
        return self.size


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


# ============================================================
# SAGIN-Compatible GNN Agents
# ============================================================
# 以下是适配 SAGIN 环境的 GNN Agent 实现
# 使用 per-neighbor 特征格式，与 V3 Agent 观测格式兼容

class SAGINGATNetwork(nn.Module):
    """
    GAT Network for SAGIN environment.

    使用邻居特征构建局部图，应用 Graph Attention。
    关键区别：使用 neighbor_features 而非全局 node_features。
    """

    def __init__(self, config: dict):
        super().__init__()

        self.feature_dim = config.get('feature_dim', 14)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)
        self.n_heads = config.get('n_heads', 4)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Self-attention over neighbors (GAT-style)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=0.1,
            batch_first=True
        )

        # Global context aggregation
        self.global_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # Per-neighbor Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, neighbor_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbor_features: [batch, max_neighbors, feature_dim]
            neighbor_mask: [batch, max_neighbors]

        Returns:
            Q-values: [batch, max_neighbors]
        """
        batch_size = neighbor_features.size(0)

        # Encode neighbor features
        h = self.feature_encoder(neighbor_features)  # [batch, max_neighbors, hidden_dim]

        # Create attention mask (True = masked)
        attn_mask = (neighbor_mask == 0)

        # Apply self-attention (GAT-style)
        # Query: each neighbor, Key/Value: all neighbors
        h_attn, _ = self.attention(
            h, h, h,
            key_padding_mask=attn_mask
        )

        # Residual connection
        h = h + h_attn

        # Global context (mean-pooling over valid neighbors)
        masked_h = h * neighbor_mask.unsqueeze(-1)
        global_ctx = masked_h.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)
        global_ctx = self.global_fc(global_ctx)  # [batch, hidden_dim]

        # Expand global context to each neighbor
        global_ctx_expanded = global_ctx.unsqueeze(1).expand(-1, self.max_neighbors, -1)

        # Concatenate and compute Q-values for each neighbor
        combined = torch.cat([h, global_ctx_expanded], dim=-1)
        q_values = self.q_head(combined).squeeze(-1)  # [batch, max_neighbors]

        return q_values


class SAGINGraphSAGENetwork(nn.Module):
    """
    GraphSAGE Network for SAGIN environment.

    使用 mean aggregation 聚合邻居特征。
    """

    def __init__(self, config: dict):
        super().__init__()

        self.feature_dim = config.get('feature_dim', 14)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        # Feature encoder (self transformation)
        self.self_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Neighbor aggregation (SAGE-style mean)
        self.neigh_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Combine self and neighbor representations
        self.combine = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Per-neighbor Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, neighbor_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbor_features: [batch, max_neighbors, feature_dim]
            neighbor_mask: [batch, max_neighbors]

        Returns:
            Q-values: [batch, max_neighbors]
        """
        # Encode each neighbor's features (self representation)
        h_self = self.self_encoder(neighbor_features)  # [batch, max_neighbors, hidden_dim]

        # Mean aggregation over all valid neighbors (neighbor representation)
        h_neigh_encoded = self.neigh_encoder(neighbor_features)
        masked_h = h_neigh_encoded * neighbor_mask.unsqueeze(-1)
        h_agg = masked_h.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)
        # [batch, hidden_dim]

        # Expand aggregated features to each neighbor position
        h_agg_expanded = h_agg.unsqueeze(1).expand(-1, self.max_neighbors, -1)

        # Combine (SAGE: concat self + aggregated neighbors)
        h_combined = torch.cat([h_self, h_agg_expanded], dim=-1)
        h = self.combine(h_combined)  # [batch, max_neighbors, hidden_dim]

        # Global context
        masked_h = h * neighbor_mask.unsqueeze(-1)
        global_ctx = masked_h.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)
        global_ctx_expanded = global_ctx.unsqueeze(1).expand(-1, self.max_neighbors, -1)

        # Q-values
        q_input = torch.cat([h, global_ctx_expanded], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)

        return q_values


class SAGINGCNNetwork(nn.Module):
    """
    GCN Network for SAGIN environment.

    标准 GCN 变体，使用邻居特征构建局部图。
    """

    def __init__(self, config: dict):
        super().__init__()

        self.feature_dim = config.get('feature_dim', 14)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        # GCN-style feature transformation
        self.gcn1 = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        self.gcn2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Per-neighbor Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, neighbor_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbor_features: [batch, max_neighbors, feature_dim]
            neighbor_mask: [batch, max_neighbors]

        Returns:
            Q-values: [batch, max_neighbors]
        """
        # First GCN layer (transform features)
        h = self.gcn1(neighbor_features)  # [batch, max_neighbors, hidden_dim]

        # Aggregation (GCN-style: mean of neighbors including self)
        masked_h = h * neighbor_mask.unsqueeze(-1)
        h_agg = masked_h.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)
        h_agg_expanded = h_agg.unsqueeze(1).expand(-1, self.max_neighbors, -1)

        # Combine with aggregated (GCN message passing approximation)
        h = h + h_agg_expanded * 0.5

        # Second GCN layer
        h = self.gcn2(h)

        # Global context
        masked_h = h * neighbor_mask.unsqueeze(-1)
        global_ctx = masked_h.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)
        global_ctx_expanded = global_ctx.unsqueeze(1).expand(-1, self.max_neighbors, -1)

        # Q-values
        q_input = torch.cat([h, global_ctx_expanded], dim=-1)
        q_values = self.q_head(q_input).squeeze(-1)

        return q_values


class SAGINDuelingNetwork(nn.Module):
    """
    Dueling DQN Network for SAGIN environment (No GNN).

    直接处理邻居特征，使用 Dueling 架构。
    """

    def __init__(self, config: dict):
        super().__init__()

        self.feature_dim = config.get('feature_dim', 14)
        self.hidden_dim = config.get('hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        # Feature encoder for each neighbor
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        # Value stream (from global context)
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        # Advantage stream (per neighbor)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, neighbor_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neighbor_features: [batch, max_neighbors, feature_dim]
            neighbor_mask: [batch, max_neighbors]

        Returns:
            Q-values: [batch, max_neighbors]
        """
        # Encode each neighbor
        h = self.feature_encoder(neighbor_features)  # [batch, max_neighbors, hidden_dim]

        # Global context for value
        masked_h = h * neighbor_mask.unsqueeze(-1)
        global_ctx = masked_h.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)

        # Value (state value)
        value = self.value_stream(global_ctx)  # [batch, 1]

        # Advantage (per action/neighbor)
        advantages = self.advantage_stream(h).squeeze(-1)  # [batch, max_neighbors]

        # Combine: Q = V + (A - mean(A))
        masked_advantages = advantages.masked_fill(neighbor_mask == 0, 0.0)
        adv_mean = masked_advantages.sum(dim=1, keepdim=True) / (
            neighbor_mask.sum(dim=1, keepdim=True) + 1e-8
        )

        q_values = value + (advantages - adv_mean)

        return q_values


class SAGINGNNAgent:
    """
    Base GNN Agent for SAGIN environment.

    支持 GAT, GraphSAGE, GCN, Dueling 等变体。
    """

    NETWORK_CLASSES = {
        'gat': SAGINGATNetwork,
        'graphsage': SAGINGraphSAGENetwork,
        'gcn': SAGINGCNNetwork,
        'dueling': SAGINDuelingNetwork,
    }

    def __init__(self, config: dict, network_type: str = 'gat'):
        self.config = config
        self.device = config.get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        self.network_type = network_type
        self.max_neighbors = config.get('max_neighbors', 8)
        self.feature_dim = config.get('feature_dim', 14)

        # Build networks
        NetworkClass = self.NETWORK_CLASSES[network_type]
        self.q_network = NetworkClass(config).to(self.device)
        self.target_network = NetworkClass(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        lr = config.get('lr', 1e-4)
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        # Replay buffer
        buffer_capacity = config.get('buffer_capacity', 100000)
        self.replay_buffer = GNNReplayBuffer(
            capacity=buffer_capacity,
            feature_dim=self.feature_dim,
            max_neighbors=self.max_neighbors,
            device=self.device
        )

        # DQN parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.batch_size = config.get('batch_size', 128)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Exploration
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # Training state
        self.train_step = 0
        self.episode_count = 0

    def select_action(self,
                      neighbor_features: np.ndarray,
                      neighbor_mask: np.ndarray,
                      action_mask: np.ndarray,
                      training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0

        # Greedy action selection
        with torch.no_grad():
            feat_t = torch.tensor(
                neighbor_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask_t = torch.tensor(
                neighbor_mask, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            q_values = self.q_network(feat_t, mask_t).squeeze(0)

            # Mask invalid actions
            action_mask_t = torch.tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            q_values = q_values.masked_fill(action_mask_t == 0, float('-inf'))

            return int(q_values.argmax().item())

    def store_transition(self,
                         neighbor_features: np.ndarray,
                         neighbor_mask: np.ndarray,
                         action: int,
                         reward: float,
                         next_neighbor_features: np.ndarray,
                         next_neighbor_mask: np.ndarray,
                         done: bool,
                         action_mask: np.ndarray,
                         next_action_mask: np.ndarray):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            neighbor_features, neighbor_mask, action, reward,
            next_neighbor_features, next_neighbor_mask, done,
            action_mask, next_action_mask
        )

    def train(self) -> Dict[str, float]:
        """Perform one training step."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return {'loss': 0.0, 'q_value': 0.0}

        batch = self.replay_buffer.sample(self.batch_size)

        # Current Q-values
        current_q = self.q_network(
            batch['neighbor_features'],
            batch['neighbor_masks']
        )
        current_q = current_q.gather(1, batch['actions'])

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Online network selects action
            next_q_online = self.q_network(
                batch['next_neighbor_features'],
                batch['next_neighbor_masks']
            )
            next_q_online = next_q_online.masked_fill(
                batch['next_action_masks'] == 0, -1e8
            )
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Target network evaluates
            next_q_target = self.target_network(
                batch['next_neighbor_features'],
                batch['next_neighbor_masks']
            )
            next_q = next_q_target.gather(1, next_actions)

            target_q = batch['rewards'] + self.gamma * next_q * (1 - batch['dones'])

        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # Soft update target network
        self.train_step += 1
        self._soft_update()

        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'target_q': target_q.mean().item()
        }

    def _soft_update(self):
        """Soft update target network."""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def end_episode(self):
        """Called at end of each episode."""
        self.episode_count += 1
        self.decay_epsilon()

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'network_type': self.network_type,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step = checkpoint.get('train_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'network_type': self.network_type,
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'buffer_size': len(self.replay_buffer),
            'lr': self.optimizer.param_groups[0]['lr']
        }
