"""
DQN Agent with Per-Neighbor Q-Value Architecture (Version 3).

核心改进：修复 V2 架构中 neighbor-action 对应关系丢失的问题

V2 问题分析：
- GCN 将所有邻居特征聚合成单一向量 [64 dim]
- DuelingHead 从这个向量"猜测" 8 个 Q 值
- 丢失了 neighbor_i → action_i 的对应关系
- 模型难以学习 "选择邻居 i 是因为邻居 i 的特征好"

V3 解决方案：Per-Neighbor Q-Value
- 每个动作 i 的 Q 值直接从对应邻居 i 的特征计算
- Q_i = f(neighbor_i_features, global_context)
- 保持 neighbor-action 的直接映射

V3 Enhanced: 空天地三层网络联合优化
- 邻居特征从 8 维扩展到 14 维
- 新增特征：
  * 节点类型 one-hot [satellite, uav, ground] (3 dims)
  * 能量水平 (1 dim) - 对 UAV 选择至关重要
  * 队列拥塞度 (1 dim)
  * 层间切换指示 (1 dim) - 跨层跳转标记

这使模型能够学习：
1. 何时选择卫星链路（高带宽、高延迟）
2. 何时选择无人机（灵活、能量受限）
3. 何时选择地面站（稳定、覆盖有限）
4. 如何平衡层间切换成本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from copy import deepcopy


class TemporalContextEncoder(nn.Module):
    """
    Transformer encoder for extracting global temporal context.

    输出用于增强每个邻居的 Q 值计算，而不是直接融合。
    """

    def __init__(self, config: dict):
        super().__init__()

        self.input_dim = config.get('simplified_history_dim', 6)
        self.d_model = config.get('temporal_d_model', 32)
        self.num_heads = config.get('temporal_num_heads', 2)
        self.history_length = config.get('history_length', 10)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.history_length, self.d_model) * 0.1
        )

        # Single transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, simplified_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            simplified_history: [batch, history_length, 6]

        Returns:
            Temporal context [batch, d_model]
        """
        batch_size = simplified_history.size(0)

        # Project input
        x = self.input_proj(simplified_history)

        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # Padding mask
        padding_mask = (simplified_history.abs().sum(dim=-1) == 0)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Use last valid position
        valid_mask = ~padding_mask
        last_valid_idx = valid_mask.long().cumsum(dim=1).argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=x.device)
        x_last = x[batch_indices, last_valid_idx]

        return x_last


class PerNeighborQNetwork(nn.Module):
    """
    Per-Neighbor Q-Value Network.

    核心思想：每个动作的 Q 值直接从对应邻居的特征计算

    Q(s, a_i) = f(neighbor_i_features, temporal_context)

    这保持了 neighbor-action 的直接对应关系，类似于 Greedy 的工作原理，
    但通过神经网络学习更复杂的评估函数。

    架构：
    1. 每个邻居的特征 [14 dim] → NeighborEncoder → [hidden_dim]
       - 8 dim 路由特征 + 6 dim 三层网络特征
    2. 历史信息 → TemporalContext → [context_dim]
    3. 对每个邻居: [neighbor_hidden, context] → QHead → Q_i
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.num_actions = config.get('num_actions', 8)
        self.max_neighbors = config.get('max_neighbors', 8)
        self.history_length = config.get('history_length', 10)

        # Feature dimensions
        # V3 Enhanced: 14 dims = 8 routing + 6 three-layer features
        self.topology_feature_dim = config.get('topology_feature_dim', 14)
        self.simplified_history_dim = config.get('simplified_history_dim', 6)
        self.neighbor_hidden_dim = config.get('neighbor_hidden_dim', 64)
        self.context_dim = config.get('temporal_d_model', 32)

        # Ablation flags
        self.use_temporal = config.get('use_transformer', True)

        # === 1. Per-Neighbor Feature Encoder ===
        # 将每个邻居的 14 维特征编码为 hidden_dim
        # 包含: 路由特征(8) + 三层网络特征(6)
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.topology_feature_dim, self.neighbor_hidden_dim),
            nn.LayerNorm(self.neighbor_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.neighbor_hidden_dim, self.neighbor_hidden_dim),
            nn.LayerNorm(self.neighbor_hidden_dim),
            nn.ReLU()
        )

        # === 2. Temporal Context Encoder (optional) ===
        if self.use_temporal:
            temporal_config = {
                'simplified_history_dim': self.simplified_history_dim,
                'temporal_d_model': self.context_dim,
                'temporal_num_heads': 2,
                'history_length': self.history_length
            }
            self.temporal_encoder = TemporalContextEncoder(temporal_config)
        else:
            self.temporal_encoder = None
            self.context_dim = 0

        # === 3. Per-Neighbor Q-Value Head ===
        # 对每个邻居：[neighbor_hidden + context] → Q_i
        q_input_dim = self.neighbor_hidden_dim + self.context_dim

        self.q_head = nn.Sequential(
            nn.Linear(q_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出单个 Q 值
        )

        # === 4. Value Baseline (Dueling-like) ===
        # 全局状态值，用于稳定训练
        if self.use_temporal:
            self.value_head = nn.Sequential(
                nn.Linear(self.context_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        else:
            # 如果没有 temporal context，使用邻居特征的聚合
            self.value_head = nn.Sequential(
                nn.Linear(self.neighbor_hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self,
                neighbor_topology_features: torch.Tensor,
                neighbor_mask: torch.Tensor,
                simplified_history: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-neighbor Q-value computation.

        Args:
            neighbor_topology_features: [batch, max_neighbors, 8]
            neighbor_mask: [batch, max_neighbors]
            simplified_history: [batch, history_length, 6]

        Returns:
            Q-values [batch, num_actions] where Q_i corresponds to neighbor_i
        """
        batch_size = neighbor_topology_features.size(0)

        # 1. Encode each neighbor's features independently
        # [batch, max_neighbors, 8] → [batch, max_neighbors, hidden_dim]
        neighbor_hidden = self.neighbor_encoder(neighbor_topology_features)

        # 2. Get temporal context (if enabled)
        if self.use_temporal:
            temporal_context = self.temporal_encoder(simplified_history)
            # [batch, context_dim]

            # Expand context to match neighbors
            # [batch, context_dim] → [batch, max_neighbors, context_dim]
            context_expanded = temporal_context.unsqueeze(1).expand(
                -1, self.max_neighbors, -1
            )

            # Concatenate neighbor features with context
            # [batch, max_neighbors, hidden_dim + context_dim]
            q_input = torch.cat([neighbor_hidden, context_expanded], dim=-1)

            # Compute state value from temporal context
            state_value = self.value_head(temporal_context)  # [batch, 1]
        else:
            q_input = neighbor_hidden
            # Use mean-pooled neighbor features for value
            masked_hidden = neighbor_hidden * neighbor_mask.unsqueeze(-1)
            mean_hidden = masked_hidden.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-8)
            state_value = self.value_head(mean_hidden)

        # 3. Compute Q-value for each neighbor independently
        # [batch, max_neighbors, q_input_dim] → [batch, max_neighbors, 1] → [batch, max_neighbors]
        advantages = self.q_head(q_input).squeeze(-1)

        # 4. Dueling: Q = V + (A - mean(A))
        # Only compute mean over valid neighbors
        masked_advantages = advantages.masked_fill(neighbor_mask == 0, 0.0)
        advantage_mean = masked_advantages.sum(dim=1, keepdim=True) / (
            neighbor_mask.sum(dim=1, keepdim=True) + 1e-8
        )

        q_values = state_value + (advantages - advantage_mean)

        return q_values


class ImprovedReplayBufferV3:
    """Replay buffer for V3 agent."""

    def __init__(self,
                 capacity: int,
                 topology_feature_dim: int,
                 simplified_history_dim: int,
                 max_neighbors: int,
                 history_length: int,
                 device: str = 'cuda'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Allocate buffers
        self.neighbor_topology_features = np.zeros(
            (capacity, max_neighbors, topology_feature_dim), dtype=np.float32
        )
        self.neighbor_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.simplified_histories = np.zeros(
            (capacity, history_length, simplified_history_dim), dtype=np.float32
        )
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_neighbor_topology_features = np.zeros(
            (capacity, max_neighbors, topology_feature_dim), dtype=np.float32
        )
        self.next_neighbor_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.next_simplified_histories = np.zeros(
            (capacity, history_length, simplified_history_dim), dtype=np.float32
        )
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)

    def push(self,
             neighbor_topology_features, neighbor_mask, simplified_history,
             action, reward,
             next_neighbor_topology_features, next_neighbor_mask, next_simplified_history,
             done, action_mask, next_action_mask):
        """Add a transition to the buffer."""
        idx = self.position

        self.neighbor_topology_features[idx] = neighbor_topology_features
        self.neighbor_masks[idx] = neighbor_mask
        self.simplified_histories[idx] = simplified_history
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_neighbor_topology_features[idx] = next_neighbor_topology_features
        self.next_neighbor_masks[idx] = next_neighbor_mask
        self.next_simplified_histories[idx] = next_simplified_history
        self.dones[idx] = float(done)
        self.action_masks[idx] = action_mask
        self.next_action_masks[idx] = next_action_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'neighbor_topology_features': torch.tensor(
                self.neighbor_topology_features[indices], device=self.device
            ),
            'neighbor_masks': torch.tensor(
                self.neighbor_masks[indices], device=self.device
            ),
            'simplified_histories': torch.tensor(
                self.simplified_histories[indices], device=self.device
            ),
            'actions': torch.tensor(
                self.actions[indices], device=self.device
            ).unsqueeze(1),
            'rewards': torch.tensor(
                self.rewards[indices], device=self.device
            ).unsqueeze(1),
            'next_neighbor_topology_features': torch.tensor(
                self.next_neighbor_topology_features[indices], device=self.device
            ),
            'next_neighbor_masks': torch.tensor(
                self.next_neighbor_masks[indices], device=self.device
            ),
            'next_simplified_histories': torch.tensor(
                self.next_simplified_histories[indices], device=self.device
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


class DQNGCNTransformerAgentV3:
    """
    DQN Agent with Per-Neighbor Q-Value Architecture.

    核心改进：
    1. 每个动作的 Q 值直接从对应邻居的特征计算
    2. 保持 neighbor-action 的直接映射
    3. Transformer 提供全局时间上下文，而非融合

    这解决了 V2 中 GCN 聚合丢失对应关系的问题。
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        # Network dimensions
        self.num_actions = config.get('num_actions', 8)
        self.max_neighbors = config.get('max_neighbors', 8)
        self.history_length = config.get('history_length', 10)
        self.topology_feature_dim = config.get('topology_feature_dim', 8)
        self.simplified_history_dim = config.get('simplified_history_dim', 6)

        # DQN parameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)

        # Exploration
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # Training
        lr = config.get('lr', 1e-4)
        self.batch_size = config.get('batch_size', 128)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Build networks
        self.q_network = PerNeighborQNetwork(config).to(self.device)
        self.target_network = PerNeighborQNetwork(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.9
        )

        # Replay buffer
        buffer_capacity = config.get('buffer_capacity', 100000)
        self.replay_buffer = ImprovedReplayBufferV3(
            capacity=buffer_capacity,
            topology_feature_dim=self.topology_feature_dim,
            simplified_history_dim=self.simplified_history_dim,
            max_neighbors=self.max_neighbors,
            history_length=self.history_length,
            device=self.device
        )

        # Training state
        self.train_step = 0
        self.episode_count = 0

    def select_action(self,
                      neighbor_topology_features: np.ndarray,
                      neighbor_mask: np.ndarray,
                      simplified_history: np.ndarray,
                      action_mask: np.ndarray,
                      training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            neighbor_topology_features: [max_neighbors, 8]
            neighbor_mask: [max_neighbors]
            simplified_history: [history_length, 6]
            action_mask: [max_neighbors]
            training: Whether in training mode

        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0

        # Greedy action selection
        with torch.no_grad():
            topo_t = torch.tensor(
                neighbor_topology_features, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask_t = torch.tensor(
                neighbor_mask, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            hist_t = torch.tensor(
                simplified_history, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            q_values = self.q_network(topo_t, mask_t, hist_t).squeeze(0)

            # Mask invalid actions
            action_mask_t = torch.tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            q_values = q_values.masked_fill(action_mask_t == 0, float('-inf'))

            return int(q_values.argmax().item())

    def store_transition(self,
                         neighbor_topology_features: np.ndarray,
                         neighbor_mask: np.ndarray,
                         simplified_history: np.ndarray,
                         action: int,
                         reward: float,
                         next_neighbor_topology_features: np.ndarray,
                         next_neighbor_mask: np.ndarray,
                         next_simplified_history: np.ndarray,
                         done: bool,
                         action_mask: np.ndarray,
                         next_action_mask: np.ndarray):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            neighbor_topology_features, neighbor_mask, simplified_history,
            action, reward,
            next_neighbor_topology_features, next_neighbor_mask, next_simplified_history,
            done, action_mask, next_action_mask
        )

    def train(self) -> Dict[str, float]:
        """Perform one training step."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return {'loss': 0.0, 'q_value': 0.0}

        batch = self.replay_buffer.sample(self.batch_size)

        # Current Q-values
        current_q = self.q_network(
            batch['neighbor_topology_features'],
            batch['neighbor_masks'],
            batch['simplified_histories']
        )
        current_q = current_q.gather(1, batch['actions'])

        # Target Q-values (Double DQN)
        with torch.no_grad():
            # Online network selects action
            next_q_online = self.q_network(
                batch['next_neighbor_topology_features'],
                batch['next_neighbor_masks'],
                batch['next_simplified_histories']
            )
            next_q_online = next_q_online.masked_fill(
                batch['next_action_masks'] == 0, -1e8
            )
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # Target network evaluates
            next_q_target = self.target_network(
                batch['next_neighbor_topology_features'],
                batch['next_neighbor_masks'],
                batch['next_simplified_histories']
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
        self.scheduler.step()

    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step = checkpoint.get('train_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'buffer_size': len(self.replay_buffer),
            'lr': self.optimizer.param_groups[0]['lr']
        }
