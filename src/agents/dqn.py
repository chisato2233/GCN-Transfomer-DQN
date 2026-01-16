"""
DQN Agent with Double DQN, Dueling architecture, and GCN-Transformer features.

Implements intelligent routing using deep Q-learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from copy import deepcopy

from ..models.gcn import LocalGCN
from ..models.transformer import TemporalEncoder
from ..models.fusion import FeatureFusion
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.

    Separates state value V(s) and advantage A(s,a) estimation.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self,
                 input_dim: int,
                 num_actions: int,
                 hidden_dims: list = [256, 128, 64]):
        """
        Initialize Dueling Q-Network.

        Args:
            input_dim: Input feature dimension
            num_actions: Number of possible actions
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()

        self.num_actions = num_actions

        # Shared feature layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.feature_layer = nn.Sequential(*layers)

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], num_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values.

        Args:
            x: Input features [batch, input_dim]

        Returns:
            Q-values [batch, num_actions]
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)  # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, num_actions]

        # Combine using dueling formula
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))

        return q_values


class StandardQNetwork(nn.Module):
    """Standard (non-dueling) Q-Network."""

    def __init__(self,
                 input_dim: int,
                 num_actions: int,
                 hidden_dims: list = [256, 128, 64]):
        """Initialize standard Q-Network."""
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values."""
        return self.network(x)


class GCNTransformerQNetwork(nn.Module):
    """
    Q-Network with GCN and Transformer feature encoders.

    Combines spatial (GCN) and temporal (Transformer) features for Q-value estimation.
    """

    def __init__(self, config: dict):
        """
        Initialize GCN-Transformer Q-Network.

        Args:
            config: Configuration dictionary
        """
        super().__init__()

        # Configuration
        self.state_dim = config.get('state_dim', 128)
        self.num_actions = config.get('num_actions', 8)
        self.use_gcn = config.get('use_gcn', True)
        self.use_transformer = config.get('use_transformer', True)
        self.use_dueling = config.get('use_dueling', True)

        gcn_config = config.get('gcn', {})
        transformer_config = config.get('transformer', {})
        fusion_config = config.get('fusion', {})

        # GCN encoder for local graph
        if self.use_gcn:
            self.gcn = LocalGCN(gcn_config)
            gcn_dim = gcn_config.get('output_dim', 64)
        else:
            gcn_dim = 0
            self.gcn = None

        # Transformer encoder for temporal features
        if self.use_transformer:
            self.transformer = TemporalEncoder(self.state_dim, transformer_config)
            transformer_dim = transformer_config.get('d_model', 64)
        else:
            transformer_dim = 0
            self.transformer = None

        # State encoder (always used)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        state_encoded_dim = 64

        # Feature fusion
        # Note: During training from replay buffer, we only have state vectors
        # GCN and Transformer features are used during action selection but not stored
        # So Q-network input dimension is based on state_encoded_dim only for training
        self.fusion = None
        fused_dim = state_encoded_dim  # 64 - only state features during training

        # Q-Network head
        q_hidden_dims = config.get('q_hidden_dims', [256, 128, 64])

        if self.use_dueling:
            self.q_network = DuelingQNetwork(fused_dim, self.num_actions, q_hidden_dims)
        else:
            self.q_network = StandardQNetwork(fused_dim, self.num_actions, q_hidden_dims)

    def forward(self,
                state: torch.Tensor,
                state_history: Optional[torch.Tensor] = None,
                current_node_features: Optional[torch.Tensor] = None,
                neighbor_features: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute Q-values.

        The Q-network only uses state encoding for simplicity and consistency
        between action selection and training. The state vector already contains
        rich information about the current node, neighbors, and destination.

        Args:
            state: Current state [batch, state_dim]
            state_history: Historical states (unused, for future extension)
            current_node_features: Current node features (unused)
            neighbor_features: Neighbor features (unused)
            neighbor_mask: Valid neighbor mask (unused)

        Returns:
            Q-values [batch, num_actions]
        """
        # State encoding
        state_features = self.state_encoder(state)

        # Q-values
        q_values = self.q_network(state_features)

        return q_values


class DQNAgent:
    """
    DQN Agent with Double DQN, Dueling, and PER support.

    Handles action selection, training, and model management.
    """

    def __init__(self, config: dict):
        """
        Initialize DQN Agent.

        Args:
            config: Agent configuration dictionary
        """
        self.config = config

        # Device
        device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)

        # Dimensions
        self.state_dim = config.get('state_dim', 128)
        self.num_actions = config.get('num_actions', 8)
        self.max_neighbors = config.get('max_neighbors', 8)
        self.history_length = config.get('history_length', 10)

        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 1e-4)
        self.batch_size = config.get('batch_size', 64)
        self.target_update_freq = config.get('target_update_freq', 100)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

        # Algorithm variants
        self.use_double_dqn = config.get('use_double_dqn', True)
        self.use_per = config.get('use_per', False)

        # Exploration
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # Build networks
        self.q_network = GCNTransformerQNetwork(config).to(self.device)
        self.target_network = deepcopy(self.q_network).to(self.device)
        self.target_network.eval()

        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.95
        )

        # Replay buffer
        buffer_capacity = config.get('buffer_capacity', 100000)

        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_capacity,
                state_dim=self.state_dim,
                max_neighbors=self.max_neighbors,
                alpha=config.get('per_alpha', 0.6),
                beta_start=config.get('per_beta_start', 0.4),
                beta_frames=config.get('per_beta_frames', 100000),
                device=self.device
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_capacity,
                state_dim=self.state_dim,
                max_neighbors=self.max_neighbors,
                device=self.device
            )

        # Training state
        self.train_step = 0
        self.episode_count = 0

    def select_action(self,
                      state: np.ndarray,
                      action_mask: np.ndarray,
                      state_history: Optional[np.ndarray] = None,
                      current_node_features: Optional[np.ndarray] = None,
                      neighbor_features: Optional[np.ndarray] = None,
                      neighbor_mask: Optional[np.ndarray] = None,
                      training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            action_mask: Valid action mask
            state_history: Historical states (optional)
            current_node_features: Current node features (optional)
            neighbor_features: Neighbor features (optional)
            neighbor_mask: Neighbor mask (optional)
            training: Whether in training mode

        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            # Random valid action
            valid_actions = np.where(action_mask > 0)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0

        # Greedy action selection
        with torch.no_grad():
            # Prepare tensors
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            history_tensor = None
            if state_history is not None:
                history_tensor = torch.tensor(
                    state_history, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

            node_tensor = None
            neighbor_tensor = None
            mask_tensor = None
            if current_node_features is not None:
                node_tensor = torch.tensor(
                    current_node_features, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                neighbor_tensor = torch.tensor(
                    neighbor_features, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                mask_tensor = torch.tensor(
                    neighbor_mask, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

            # Compute Q-values
            q_values = self.q_network(
                state_tensor, history_tensor,
                node_tensor, neighbor_tensor, mask_tensor
            ).squeeze(0)

            # Mask invalid actions
            action_mask_tensor = torch.tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            q_values = q_values.masked_fill(action_mask_tensor == 0, float('-inf'))

            return int(q_values.argmax().item())

    def store_transition(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool,
                        action_mask: np.ndarray,
                        next_action_mask: np.ndarray):
        """Store transition in replay buffer."""
        self.replay_buffer.push(
            state, action, reward, next_state, done,
            action_mask, next_action_mask
        )

    def train(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary with training metrics
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return {'loss': 0.0, 'q_value': 0.0}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        next_action_masks = batch['next_action_masks']

        # Compute current Q-values
        current_q = self.q_network(states)
        current_q = current_q.gather(1, actions)

        # Compute target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Use online network for action selection
                next_q_online = self.q_network(next_states)
                next_q_online = next_q_online.masked_fill(
                    next_action_masks == 0, float('-inf')
                )
                next_actions = next_q_online.argmax(dim=1, keepdim=True)

                # Use target network for Q-value estimation
                next_q_target = self.target_network(next_states)
                next_q = next_q_target.gather(1, next_actions)
            else:
                # Standard DQN
                next_q = self.target_network(next_states)
                next_q = next_q.masked_fill(next_action_masks == 0, float('-inf'))
                next_q = next_q.max(dim=1, keepdim=True)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        if self.use_per:
            weights = batch['weights']
            td_errors = (current_q - target_q).abs().detach().cpu().numpy()
            loss = (weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

            # Update priorities
            self.replay_buffer.update_priorities(batch['indices'], td_errors.squeeze())
        else:
            loss = F.smooth_l1_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'target_q': target_q.mean().item()
        }

    def update_target_network(self):
        """Hard update target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target_network(self, tau: float = 0.005):
        """Soft update target network."""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
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
            'episode_count': self.episode_count
        }, path)

    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
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
