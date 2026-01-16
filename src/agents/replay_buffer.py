"""
Experience replay buffers for DRL training.

Includes standard replay buffer and Prioritized Experience Replay (PER).
"""

import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from collections import deque
import random


class ReplayBuffer:
    """
    Standard experience replay buffer.

    Stores transitions and samples uniformly at random.
    """

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 max_neighbors: int = 8,
                 device: str = 'cpu'):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: State vector dimension
            max_neighbors: Maximum number of neighbors (action mask size)
            device: Device for tensors
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.max_neighbors = max_neighbors
        self.device = device

        self.position = 0
        self.size = 0

        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             action_mask: np.ndarray,
             next_action_mask: np.ndarray):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            action_mask: Valid action mask for current state
            next_action_mask: Valid action mask for next state
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.action_masks[self.position] = action_mask
        self.next_action_masks[self.position] = next_action_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions uniformly.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of tensors for training
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.tensor(
                self.states[indices], dtype=torch.float32, device=self.device
            ),
            'actions': torch.tensor(
                self.actions[indices], dtype=torch.long, device=self.device
            ),
            'rewards': torch.tensor(
                self.rewards[indices], dtype=torch.float32, device=self.device
            ),
            'next_states': torch.tensor(
                self.next_states[indices], dtype=torch.float32, device=self.device
            ),
            'dones': torch.tensor(
                self.dones[indices], dtype=torch.float32, device=self.device
            ),
            'action_masks': torch.tensor(
                self.action_masks[indices], dtype=torch.float32, device=self.device
            ),
            'next_action_masks': torch.tensor(
                self.next_action_masks[indices], dtype=torch.float32, device=self.device
            )
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


class SumTree:
    """
    Sum tree data structure for efficient priority-based sampling.

    Supports O(log n) priority updates and sampling.
    """

    def __init__(self, capacity: int):
        """Initialize sum tree."""
        self.capacity = capacity
        # Tree has 2 * capacity - 1 nodes
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for a given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        """Get total priority sum."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Get maximum priority in the tree."""
        return np.max(self.tree[-self.capacity:])

    @property
    def min_priority(self) -> float:
        """Get minimum non-zero priority."""
        leaves = self.tree[-self.capacity:]
        non_zero = leaves[leaves > 0]
        return np.min(non_zero) if len(non_zero) > 0 else 1.0

    def add(self, priority: float, data_idx: int):
        """Add or update priority for a data index."""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Get sample based on cumulative priority.

        Args:
            s: Random value in [0, total_priority)

        Returns:
            Tuple of (tree_idx, priority, data_idx)
        """
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer.

    Samples transitions based on their TD-error priority.
    """

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 max_neighbors: int = 8,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 device: str = 'cpu'):
        """
        Initialize PER buffer.

        Args:
            capacity: Maximum number of transitions
            state_dim: State dimension
            max_neighbors: Action mask size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_frames: Frames to anneal beta to 1.0
            device: Device for tensors
        """
        super().__init__(capacity, state_dim, max_neighbors, device)

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

        # Sum tree for priority-based sampling
        self.tree = SumTree(capacity)

        # Priority bounds
        self.min_priority = 1e-6
        self.max_priority = 1.0

    @property
    def beta(self) -> float:
        """Get current beta value (annealed from beta_start to 1.0)."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def push(self,
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             action_mask: np.ndarray,
             next_action_mask: np.ndarray):
        """Add transition with maximum priority."""
        # Store transition
        super().push(state, action, reward, next_state, done,
                    action_mask, next_action_mask)

        # New transitions get maximum priority
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (self.position - 1) % self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch based on priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary with transitions and importance weights
        """
        indices = []
        priorities = []

        # Divide priority range into segments
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.get(s)
            indices.append(data_idx)
            priorities.append(priority)

        indices = np.array(indices)
        priorities = np.array(priorities)

        # Compute importance sampling weights
        probs = priorities / self.tree.total
        weights = (self.size * probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        self.frame += 1

        return {
            'states': torch.tensor(
                self.states[indices], dtype=torch.float32, device=self.device
            ),
            'actions': torch.tensor(
                self.actions[indices], dtype=torch.long, device=self.device
            ),
            'rewards': torch.tensor(
                self.rewards[indices], dtype=torch.float32, device=self.device
            ),
            'next_states': torch.tensor(
                self.next_states[indices], dtype=torch.float32, device=self.device
            ),
            'dones': torch.tensor(
                self.dones[indices], dtype=torch.float32, device=self.device
            ),
            'action_masks': torch.tensor(
                self.action_masks[indices], dtype=torch.float32, device=self.device
            ),
            'next_action_masks': torch.tensor(
                self.next_action_masks[indices], dtype=torch.float32, device=self.device
            ),
            'weights': torch.tensor(
                weights, dtype=torch.float32, device=self.device
            ).unsqueeze(1),
            'indices': indices
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Data indices to update
            td_errors: Corresponding TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.add(priority, idx)


class SequenceReplayBuffer:
    """
    Replay buffer that stores sequences for Transformer training.

    Stores complete state history sequences.
    """

    def __init__(self,
                 capacity: int,
                 state_dim: int,
                 history_length: int,
                 max_neighbors: int = 8,
                 device: str = 'cpu'):
        """
        Initialize sequence replay buffer.

        Args:
            capacity: Maximum number of transitions
            state_dim: State dimension
            history_length: Length of state history
            max_neighbors: Action mask size
            device: Device for tensors
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.history_length = history_length
        self.max_neighbors = max_neighbors
        self.device = device

        self.position = 0
        self.size = 0

        # Store current state and history
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.state_histories = np.zeros(
            (capacity, history_length, state_dim), dtype=np.float32
        )
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state_histories = np.zeros(
            (capacity, history_length, state_dim), dtype=np.float32
        )
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, max_neighbors), dtype=np.float32)

    def push(self,
             state: np.ndarray,
             state_history: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             next_state_history: np.ndarray,
             done: bool,
             action_mask: np.ndarray,
             next_action_mask: np.ndarray):
        """Add transition with state history."""
        self.states[self.position] = state
        self.state_histories[self.position] = state_history
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.next_state_histories[self.position] = next_state_history
        self.dones[self.position] = float(done)
        self.action_masks[self.position] = action_mask
        self.next_action_masks[self.position] = next_action_mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with state histories."""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.tensor(
                self.states[indices], dtype=torch.float32, device=self.device
            ),
            'state_histories': torch.tensor(
                self.state_histories[indices], dtype=torch.float32, device=self.device
            ),
            'actions': torch.tensor(
                self.actions[indices], dtype=torch.long, device=self.device
            ),
            'rewards': torch.tensor(
                self.rewards[indices], dtype=torch.float32, device=self.device
            ),
            'next_states': torch.tensor(
                self.next_states[indices], dtype=torch.float32, device=self.device
            ),
            'next_state_histories': torch.tensor(
                self.next_state_histories[indices], dtype=torch.float32, device=self.device
            ),
            'dones': torch.tensor(
                self.dones[indices], dtype=torch.float32, device=self.device
            ),
            'action_masks': torch.tensor(
                self.action_masks[indices], dtype=torch.float32, device=self.device
            ),
            'next_action_masks': torch.tensor(
                self.next_action_masks[indices], dtype=torch.float32, device=self.device
            )
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size
