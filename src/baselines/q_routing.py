"""
Q-Routing Baseline Implementation.

Reference:
- Boyan & Littman (1993): "Packet Routing in Dynamically Changing Networks"
- Q-adaptive (2024): "Q-adaptive: A Multi-Agent Reinforcement Learning Based Routing"
  https://arxiv.org/abs/2403.16301

Q-Routing是最早将强化学习应用于网络路由的方法之一。
每个节点作为独立的agent，维护Q表进行路由决策。

特点：
- 分布式：每个节点独立决策
- 表格型Q-learning：使用Q表而非神经网络
- 局部信息：仅使用邻居信息
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class QRoutingAgent:
    """
    Q-Routing Agent for packet routing.

    Q(x, d, y) = estimate of time for packet at node x destined for d
                 to reach d if sent to neighbor y

    Update rule:
    Q(x, d, y) <- Q(x, d, y) + α * (q + s + min_z Q(y, d, z) - Q(x, d, y))

    where:
    - q: queuing delay at y
    - s: transmission time from x to y
    - min_z Q(y, d, z): best estimated time from y to d
    """

    def __init__(self,
                 num_nodes: int,
                 learning_rate: float = 0.5,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.9995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Routing agent.

        Args:
            num_nodes: Total number of nodes in network
            learning_rate: Learning rate for Q-table updates
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
        """
        self.num_nodes = num_nodes
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: Q[node][destination][neighbor] -> estimated time
        # Initialize with small random values
        self.q_table: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: np.random.uniform(0, 1)))
        )

    def select_action(self,
                      current_node: int,
                      destination: int,
                      neighbors: List[int],
                      action_mask: np.ndarray,
                      training: bool = True) -> int:
        """
        Select next hop using epsilon-greedy policy.

        Args:
            current_node: Current node ID
            destination: Destination node ID
            neighbors: List of neighbor node IDs
            action_mask: Valid action mask
            training: Whether in training mode

        Returns:
            Action index (index into neighbors list)
        """
        valid_indices = np.where(action_mask > 0)[0]
        if len(valid_indices) == 0:
            return 0

        # Direct delivery
        for i, n in enumerate(neighbors):
            if n == destination and i in valid_indices:
                return i

        # Epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))

        # Greedy selection based on Q-values
        best_action = valid_indices[0]
        best_q = float('inf')  # Lower is better (estimated time)

        for i in valid_indices:
            if i < len(neighbors):
                neighbor = neighbors[i]
                q_value = self.q_table[current_node][destination][neighbor]
                if q_value < best_q:
                    best_q = q_value
                    best_action = i

        return int(best_action)

    def update(self,
               current_node: int,
               destination: int,
               next_node: int,
               transmission_time: float,
               next_neighbors: List[int]):
        """
        Update Q-table based on observed transition.

        Q(x, d, y) <- Q(x, d, y) + α * (t + min_z Q(y, d, z) - Q(x, d, y))

        Args:
            current_node: Node where decision was made
            destination: Final destination
            next_node: Chosen next hop
            transmission_time: Time to reach next_node (delay + queue)
            next_neighbors: Neighbors of next_node
        """
        # Current Q-value
        current_q = self.q_table[current_node][destination][next_node]

        # Estimate from next node
        if next_node == destination:
            next_q = 0
        else:
            # Find minimum Q-value from next_node
            next_q = float('inf')
            for neighbor in next_neighbors:
                q = self.q_table[next_node][destination][neighbor]
                next_q = min(next_q, q)
            if next_q == float('inf'):
                next_q = 10.0  # Default estimate

        # Q-learning update
        target = transmission_time + next_q
        self.q_table[current_node][destination][next_node] = (
            current_q + self.learning_rate * (target - current_q)
        )

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, node: int, destination: int, neighbor: int) -> float:
        """Get Q-value for debugging."""
        return self.q_table[node][destination][neighbor]


class PredictiveQRouting(QRoutingAgent):
    """
    Predictive Q-Routing (PQ-Routing) variant.

    Reference: Choi & Yeung (1996)

    Improvement: Uses recovery mode to handle congestion.
    When network load decreases, quickly recover to better routes.
    """

    def __init__(self,
                 num_nodes: int,
                 learning_rate: float = 0.5,
                 epsilon: float = 0.1,
                 recovery_rate: float = 0.95):
        super().__init__(num_nodes, learning_rate, epsilon)
        self.recovery_rate = recovery_rate

        # Store best known Q-values (used in recovery)
        self.best_q: Dict[int, Dict[int, Dict[int, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: float('inf')))
        )

    def update(self,
               current_node: int,
               destination: int,
               next_node: int,
               transmission_time: float,
               next_neighbors: List[int]):
        """Update with predictive recovery."""
        # Current Q-value
        current_q = self.q_table[current_node][destination][next_node]

        # Estimate from next node
        if next_node == destination:
            next_q = 0
        else:
            next_q = float('inf')
            for neighbor in next_neighbors:
                q = self.q_table[next_node][destination][neighbor]
                next_q = min(next_q, q)
            if next_q == float('inf'):
                next_q = 10.0

        # Compute new Q-value
        new_q = transmission_time + next_q

        # Update best known Q-value
        if new_q < self.best_q[current_node][destination][next_node]:
            self.best_q[current_node][destination][next_node] = new_q

        # Recovery: if current estimate is worse than best, try to recover
        best = self.best_q[current_node][destination][next_node]
        if new_q > best:
            # Interpolate towards best
            new_q = self.recovery_rate * new_q + (1 - self.recovery_rate) * best

        # Update Q-table
        self.q_table[current_node][destination][next_node] = (
            current_q + self.learning_rate * (new_q - current_q)
        )


class DualReinforcementQRouting(QRoutingAgent):
    """
    Dual Reinforcement Q-Routing (DRQ-Routing).

    Reference: Kumar & Miikkulainen (1998)

    Improvement: Also learns from packets going in the opposite direction.
    """

    def __init__(self,
                 num_nodes: int,
                 learning_rate: float = 0.5,
                 epsilon: float = 0.1,
                 backward_learning_rate: float = 0.3):
        super().__init__(num_nodes, learning_rate, epsilon)
        self.backward_learning_rate = backward_learning_rate

    def backward_update(self,
                        source: int,
                        current_node: int,
                        path_time: float):
        """
        Update Q-values using backward information.

        When a packet arrives at destination, update Q-values
        for nodes along the reverse path.
        """
        # The packet came from source to current_node
        # Update Q-value for routing from current_node back to source
        for neighbor in self.q_table[current_node][source]:
            current_q = self.q_table[current_node][source][neighbor]
            # If this path was good, update all routes to source
            self.q_table[current_node][source][neighbor] = (
                current_q - self.backward_learning_rate * (current_q - path_time)
            )
