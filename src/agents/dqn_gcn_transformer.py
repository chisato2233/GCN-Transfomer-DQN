"""
DQN Agent with full GCN + Transformer integration.

This module implements intelligent routing using:
- GCN: Captures spatial graph topology features
- Transformer: Captures temporal state sequence patterns
- Feature Fusion: Combines spatial, temporal, and state features

[FIXED VERSION] Key changes:
- Increased tau for soft update (0.001 -> 0.005)
- Added gradient clipping improvements
- Better initialization
- Added training stability features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from copy import deepcopy

from ..models.gcn import LocalGCN, GCNEncoder
from ..models.transformer import TemporalEncoder
from ..models.fusion import FeatureFusion, GatedFusion


class DuelingHead(nn.Module):
    """Dueling DQN head: separates value and advantage streams."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()

        # Value stream V(s) with LayerNorm for stability
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream A(s,a) with LayerNorm for stability
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Q = V + A - mean(A)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class GCNTransformerNetwork(nn.Module):
    """
    Neural network combining GCN, Transformer, and state encoding.

    Architecture:
    1. LocalGCN: Processes current node + neighbors (spatial features)
    2. TemporalEncoder: Processes state history (temporal features)
    3. StateEncoder: Processes current state vector
    4. GatedFusion: Fuses all three feature types
    5. DuelingHead: Outputs Q-values
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.state_dim = config.get('state_dim', 132)
        self.num_actions = config.get('num_actions', 8)
        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.max_neighbors = config.get('max_neighbors', 8)
        self.history_length = config.get('history_length', 10)

        # Feature dimensions
        gcn_output_dim = config.get('gcn_output_dim', 64)
        transformer_output_dim = config.get('transformer_output_dim', 64)
        state_encoded_dim = config.get('state_encoded_dim', 64)
        fused_dim = config.get('fused_dim', 128)

        # === 1. Local GCN for spatial features ===
        gcn_config = {
            'node_feature_dim': self.node_feature_dim,
            'hidden_dim': 64,
            'output_dim': gcn_output_dim,
            'max_neighbors': self.max_neighbors
        }
        self.local_gcn = LocalGCN(gcn_config)

        # === 2. Temporal Transformer for history ===
        transformer_config = {
            'd_model': transformer_output_dim,
            'num_heads': 4,
            'num_layers': 2,
            'd_feedforward': 128,
            'dropout': 0.1,
            'history_length': self.history_length,
            'max_seq_length': self.history_length
        }
        self.temporal_encoder = TemporalEncoder(self.state_dim, transformer_config)

        # === 3. State encoder (MLP) with LayerNorm for stability ===
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, state_encoded_dim),
            nn.LayerNorm(state_encoded_dim),
            nn.ReLU()
        )

        # === 4. Feature Fusion (Gated) ===
        self.fusion = GatedFusion(
            gcn_dim=gcn_output_dim,
            temporal_dim=transformer_output_dim,
            state_dim=state_encoded_dim,
            output_dim=fused_dim,
            dropout=0.1
        )

        # === 5. Q-Network Head ===
        self.use_dueling = config.get('use_dueling', True)
        if self.use_dueling:
            self.q_head = DuelingHead(fused_dim, self.num_actions, hidden_dim=128)
        else:
            self.q_head = nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )

        # Initialize state encoder weights
        self._init_state_encoder()

    def _init_state_encoder(self):
        """Initialize state encoder weights with Xavier/Glorot for stability."""
        for module in self.state_encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Also initialize non-dueling q_head if present
        if not self.use_dueling:
            for module in self.q_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self,
                state: torch.Tensor,
                state_history: torch.Tensor,
                current_node_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values.

        Args:
            state: Current state vector [batch, state_dim]
            state_history: Historical states [batch, history_length, state_dim]
            current_node_features: Current node features [batch, node_feature_dim]
            neighbor_features: Neighbor node features [batch, max_neighbors, node_feature_dim]
            neighbor_mask: Valid neighbor mask [batch, max_neighbors]

        Returns:
            Q-values for each action [batch, num_actions]
        """
        batch_size = state.size(0)

        # 1. GCN spatial features
        gcn_features = self.local_gcn(
            current_node_features,
            neighbor_features,
            neighbor_mask
        )  # [batch, gcn_output_dim]

        # 2. Transformer temporal features
        temporal_features = self.temporal_encoder(state_history)  # [batch, transformer_output_dim]

        # 3. State encoding
        state_features = self.state_encoder(state)  # [batch, state_encoded_dim]

        # 4. Feature fusion
        fused_features = self.fusion(
            gcn_features,
            temporal_features,
            state_features
        )  # [batch, fused_dim]

        # 5. Q-values
        q_values = self.q_head(fused_features)

        return q_values


class GraphReplayBuffer:
    """
    Replay buffer storing transitions with graph and temporal data.
    """

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 node_feature_dim: int,
                 max_neighbors: int,
                 history_length: int,
                 device: str = 'cuda'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Allocate buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.state_histories = np.zeros((capacity, history_length, state_dim), dtype=np.float32)
        self.current_node_features = np.zeros((capacity, node_feature_dim), dtype=np.float32)
        self.neighbor_features = np.zeros((capacity, max_neighbors, node_feature_dim), dtype=np.float32)
        self.neighbor_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state_histories = np.zeros((capacity, history_length, state_dim), dtype=np.float32)
        self.next_current_node_features = np.zeros((capacity, node_feature_dim), dtype=np.float32)
        self.next_neighbor_features = np.zeros((capacity, max_neighbors, node_feature_dim), dtype=np.float32)
        self.next_neighbor_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)

    def push(self,
             state, state_history,
             current_node_features, neighbor_features, neighbor_mask,
             action, reward,
             next_state, next_state_history,
             next_current_node_features, next_neighbor_features, next_neighbor_mask,
             done, action_mask, next_action_mask):
        """Add a transition to the buffer."""
        idx = self.position

        self.states[idx] = state
        self.state_histories[idx] = state_history
        self.current_node_features[idx] = current_node_features
        self.neighbor_features[idx] = neighbor_features
        self.neighbor_masks[idx] = neighbor_mask
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.next_state_histories[idx] = next_state_history
        self.next_current_node_features[idx] = next_current_node_features
        self.next_neighbor_features[idx] = next_neighbor_features
        self.next_neighbor_masks[idx] = next_neighbor_mask
        self.dones[idx] = float(done)
        self.action_masks[idx] = action_mask
        self.next_action_masks[idx] = next_action_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.tensor(self.states[indices], device=self.device),
            'state_histories': torch.tensor(self.state_histories[indices], device=self.device),
            'current_node_features': torch.tensor(self.current_node_features[indices], device=self.device),
            'neighbor_features': torch.tensor(self.neighbor_features[indices], device=self.device),
            'neighbor_masks': torch.tensor(self.neighbor_masks[indices], device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device).unsqueeze(1),
            'rewards': torch.tensor(self.rewards[indices], device=self.device).unsqueeze(1),
            'next_states': torch.tensor(self.next_states[indices], device=self.device),
            'next_state_histories': torch.tensor(self.next_state_histories[indices], device=self.device),
            'next_current_node_features': torch.tensor(self.next_current_node_features[indices], device=self.device),
            'next_neighbor_features': torch.tensor(self.next_neighbor_features[indices], device=self.device),
            'next_neighbor_masks': torch.tensor(self.next_neighbor_masks[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device).unsqueeze(1),
            'action_masks': torch.tensor(self.action_masks[indices], device=self.device),
            'next_action_masks': torch.tensor(self.next_action_masks[indices], device=self.device),
        }

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size

    def __len__(self) -> int:
        return self.size


class PrioritizedGraphReplayBuffer(GraphReplayBuffer):
    """Prioritized Experience Replay buffer with graph data."""

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 node_feature_dim: int,
                 max_neighbors: int,
                 history_length: int,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 device: str = 'cuda'):
        super().__init__(capacity, state_dim, node_feature_dim, max_neighbors, history_length, device)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, *args, **kwargs):
        """Add transition with max priority."""
        idx = self.position
        super().push(*args, **kwargs)
        self.priorities[idx] = self.max_priority

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample based on priorities."""
        self.frame += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)

        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = {
            'states': torch.tensor(self.states[indices], device=self.device),
            'state_histories': torch.tensor(self.state_histories[indices], device=self.device),
            'current_node_features': torch.tensor(self.current_node_features[indices], device=self.device),
            'neighbor_features': torch.tensor(self.neighbor_features[indices], device=self.device),
            'neighbor_masks': torch.tensor(self.neighbor_masks[indices], device=self.device),
            'actions': torch.tensor(self.actions[indices], device=self.device).unsqueeze(1),
            'rewards': torch.tensor(self.rewards[indices], device=self.device).unsqueeze(1),
            'next_states': torch.tensor(self.next_states[indices], device=self.device),
            'next_state_histories': torch.tensor(self.next_state_histories[indices], device=self.device),
            'next_current_node_features': torch.tensor(self.next_current_node_features[indices], device=self.device),
            'next_neighbor_features': torch.tensor(self.next_neighbor_features[indices], device=self.device),
            'next_neighbor_masks': torch.tensor(self.next_neighbor_masks[indices], device=self.device),
            'dones': torch.tensor(self.dones[indices], device=self.device).unsqueeze(1),
            'action_masks': torch.tensor(self.action_masks[indices], device=self.device),
            'next_action_masks': torch.tensor(self.next_action_masks[indices], device=self.device),
            'weights': torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1),
            'indices': indices,
        }
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority + 1e-6)


class DQNGCNTransformerAgent:
    """
    DQN Agent with GCN + Transformer integration.

    [FIXED VERSION] Key changes:
    - Increased tau for soft update (0.001 -> 0.005)
    - Better gradient clipping
    - Improved exploration decay handling
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = config.get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'

        # Network dimensions
        self.state_dim = config.get('state_dim', 132)
        self.num_actions = config.get('num_actions', 8)
        self.node_feature_dim = config.get('node_feature_dim', 9)
        self.max_neighbors = config.get('max_neighbors', 8)
        self.history_length = config.get('history_length', 10)

        # DQN parameters
        self.gamma = config.get('gamma', 0.99)
        self.use_double_dqn = config.get('use_double_dqn', True)
        self.use_dueling = config.get('use_dueling', True)
        self.use_per = config.get('use_per', False)

        # [FIX] Increased tau for faster target network updates
        self.tau = config.get('tau', 0.005)  # Was 0.001

        # Exploration parameters
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)  # Faster decay

        # Training parameters
        lr = config.get('lr', 3e-4)
        self.batch_size = config.get('batch_size', 64)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.target_update_freq = config.get('target_update_freq', 100)

        # Build networks
        self.q_network = GCNTransformerNetwork(config).to(self.device)
        self.target_network = GCNTransformerNetwork(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=lr,
            weight_decay=1e-5  # Small L2 regularization
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=500,
            gamma=0.95
        )

        # Replay buffer
        buffer_capacity = config.get('buffer_capacity', 100000)

        if self.use_per:
            self.replay_buffer = PrioritizedGraphReplayBuffer(
                capacity=buffer_capacity,
                state_dim=self.state_dim,
                node_feature_dim=self.node_feature_dim,
                max_neighbors=self.max_neighbors,
                history_length=self.history_length,
                alpha=config.get('per_alpha', 0.6),
                beta_start=config.get('per_beta_start', 0.4),
                beta_frames=config.get('per_beta_frames', 100000),
                device=self.device
            )
        else:
            self.replay_buffer = GraphReplayBuffer(
                capacity=buffer_capacity,
                state_dim=self.state_dim,
                node_feature_dim=self.node_feature_dim,
                max_neighbors=self.max_neighbors,
                history_length=self.history_length,
                device=self.device
            )

        # Training state
        self.train_step = 0
        self.episode_count = 0

    def select_action(self,
                      state: np.ndarray,
                      state_history: np.ndarray,
                      current_node_features: np.ndarray,
                      neighbor_features: np.ndarray,
                      neighbor_mask: np.ndarray,
                      action_mask: np.ndarray,
                      training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with full feature input.

        Args:
            state: Current state vector
            state_history: Historical states [history_length, state_dim]
            current_node_features: Current node features
            neighbor_features: Neighbor node features
            neighbor_mask: Valid neighbor mask
            action_mask: Valid action mask
            training: Whether in training mode

        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration (only during training)
        if training and np.random.random() < self.epsilon:
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0

        # Greedy action selection
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            history_t = torch.tensor(state_history, dtype=torch.float32, device=self.device).unsqueeze(0)
            node_t = torch.tensor(current_node_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            neighbor_t = torch.tensor(neighbor_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(neighbor_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            q_values = self.q_network(state_t, history_t, node_t, neighbor_t, mask_t).squeeze(0)

            # Mask invalid actions
            action_mask_t = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            q_values = q_values.masked_fill(action_mask_t == 0, float('-inf'))

            return int(q_values.argmax().item())

    def store_transition(self,
                        state: np.ndarray,
                        state_history: np.ndarray,
                        current_node_features: np.ndarray,
                        neighbor_features: np.ndarray,
                        neighbor_mask: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        next_state_history: np.ndarray,
                        next_current_node_features: np.ndarray,
                        next_neighbor_features: np.ndarray,
                        next_neighbor_mask: np.ndarray,
                        done: bool,
                        action_mask: np.ndarray,
                        next_action_mask: np.ndarray):
        """Store transition with all graph and temporal data."""
        self.replay_buffer.push(
            state, state_history,
            current_node_features, neighbor_features, neighbor_mask,
            action, reward,
            next_state, next_state_history,
            next_current_node_features, next_neighbor_features, next_neighbor_mask,
            done, action_mask, next_action_mask
        )

    def train(self) -> Dict[str, float]:
        """Perform one training step with GCN + Transformer features."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return {'loss': 0.0, 'q_value': 0.0}

        batch = self.replay_buffer.sample(self.batch_size)

        # Compute current Q-values
        current_q = self.q_network(
            batch['states'],
            batch['state_histories'],
            batch['current_node_features'],
            batch['neighbor_features'],
            batch['neighbor_masks']
        )
        current_q = current_q.gather(1, batch['actions'])

        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: online network selects action
                next_q_online = self.q_network(
                    batch['next_states'],
                    batch['next_state_histories'],
                    batch['next_current_node_features'],
                    batch['next_neighbor_features'],
                    batch['next_neighbor_masks']
                )
                # Use large negative value instead of -inf to avoid NaN
                next_q_online = next_q_online.masked_fill(
                    batch['next_action_masks'] == 0, -1e8
                )
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # Target network evaluates action
                next_q_target = self.target_network(
                    batch['next_states'],
                    batch['next_state_histories'],
                    batch['next_current_node_features'],
                    batch['next_neighbor_features'],
                    batch['next_neighbor_masks']
                )
                next_q = next_q_target.gather(1, next_actions)
            else:
                next_q = self.target_network(
                    batch['next_states'],
                    batch['next_state_histories'],
                    batch['next_current_node_features'],
                    batch['next_neighbor_features'],
                    batch['next_neighbor_masks']
                )
                # Use large negative value instead of -inf to avoid NaN
                next_q = next_q.masked_fill(batch['next_action_masks'] == 0, -1e8)
                next_q = next_q.max(dim=1, keepdim=True)[0]

            target_q = batch['rewards'] + self.gamma * next_q * (1 - batch['dones'])

        # Compute loss
        if self.use_per:
            weights = batch['weights']
            td_errors = (current_q - target_q).abs().detach()
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

            # Update priorities
            self.replay_buffer.update_priorities(
                batch['indices'],
                td_errors.cpu().numpy().squeeze()
            )
        else:
            loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # [FIX] Update target network with increased tau
        self.train_step += 1
        self.update_target_network(soft=True, tau=self.tau)

        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'target_q': target_q.mean().item()
        }

    def update_target_network(self, soft: bool = True, tau: float = None):
        """
        Update target network.

        Args:
            soft: If True, use soft update (Polyak averaging)
            tau: Soft update coefficient (1.0 = hard update)
        """
        if tau is None:
            tau = self.tau

        if soft:
            # Soft update: θ_target = τ*θ_online + (1-τ)*θ_target
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )
        else:
            # Hard update
            self.target_network.load_state_dict(self.q_network.state_dict())

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
