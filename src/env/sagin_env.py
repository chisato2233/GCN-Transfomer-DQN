"""
SAGIN Intelligent Routing Environment.

A Gymnasium environment for training DRL agents to make routing decisions
in Space-Air-Ground Integrated Networks.

[FIXED VERSION] Key changes:
- Fixed loop penalty multiplier (0.5 -> 1.0)
- Improved reward shaping for better learning signal
- Added success proximity bonus
- Better progress reward scaling
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from .network_topology import SAGINTopology, NodeType


class SAGINRoutingEnv(gym.Env):
    """
    SAGIN Intelligent Routing Environment.

    The agent must route a packet from source to destination by selecting
    the next-hop neighbor at each step.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Dict, render_mode: Optional[str] = None):
        """
        Initialize the routing environment.

        Args:
            config: Environment configuration dictionary
            render_mode: Rendering mode (optional)
        """
        super().__init__()

        self.config = config
        self.render_mode = render_mode

        # Initialize network topology
        network_config = config.get('network', {})
        self.topology = SAGINTopology(network_config)

        # Environment parameters
        env_config = config.get('environment', {})
        self.max_hops = env_config.get('max_hops', 50)
        self.max_neighbors = env_config.get('max_neighbors', 8)
        self.history_length = env_config.get('history_length', 10)
        self.topology_update_freq = env_config.get('topology_update_freq', 10)
        self.link_failure_prob = env_config.get('link_failure_prob', 0.05)

        # Reward parameters
        reward_config = config.get('reward', {})
        self.alpha = reward_config.get('delay_weight', 1.0)
        self.beta = reward_config.get('loss_weight', 10.0)
        self.gamma_congestion = reward_config.get('congestion_weight', 0.5)
        self.success_bonus = reward_config.get('success_bonus', 20.0)  # Increased from 10
        self.loop_penalty = reward_config.get('loop_penalty', 2.0)
        self.timeout_penalty = reward_config.get('timeout_penalty', 5.0)  # Increased
        self.invalid_action_penalty = reward_config.get('invalid_action_penalty', 1.0)
        self.progress_reward = reward_config.get('progress_reward', 0.5)  # Increased

        # Feature dimensions
        self.node_feature_dim = 9  # From Node.to_feature_vector()
        self.link_feature_dim = 5  # From Link.to_feature_vector()

        # State dimension calculation
        # Current node (9) + neighbor links (5*8) + neighbor nodes (9*8) + target info (9+2)
        self.state_dim = (
            self.node_feature_dim +                           # Current node
            self.link_feature_dim * self.max_neighbors +      # Neighbor links
            self.node_feature_dim * self.max_neighbors +      # Neighbor nodes
            self.node_feature_dim + 2                         # Target node + distance info
        )

        # Action space: Discrete selection of next-hop neighbor
        self.action_space = spaces.Discrete(self.max_neighbors)

        # Observation space
        self.observation_space = spaces.Dict({
            'state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim,), dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.max_neighbors,), dtype=np.float32
            ),
            'node_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.topology.total_nodes, self.node_feature_dim),
                dtype=np.float32
            ),
            'adjacency': spaces.Box(
                low=0, high=1,
                shape=(self.topology.total_nodes, self.topology.total_nodes),
                dtype=np.float32
            )
        })

        # Episode state
        self.current_node: int = 0
        self.destination: int = 0
        self.source: int = 0
        self.path_history: List[int] = []
        self.visited_nodes: set = set()
        self.hop_count: int = 0
        self.total_delay: float = 0.0
        self.packet_dropped: bool = False
        self.time_step: int = 0
        
        # [FIX] Track initial distance for consistent normalization
        self.initial_distance: float = 1.0

        # State history for Transformer
        self.state_history: deque = deque(maxlen=self.history_length)

        # Statistics
        self.episode_count = 0

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset topology periodically
        if self.episode_count % 100 == 0:
            self.topology.reset()

        self.episode_count += 1

        # Sample source and destination
        self.source = self._sample_source_node()
        self.destination = self._sample_destination_node(self.source)
        self.current_node = self.source

        # [FIX] Calculate and store initial distance
        source_pos = self.topology.nodes[self.source].position
        dest_pos = self.topology.nodes[self.destination].position
        self.initial_distance = np.linalg.norm(source_pos - dest_pos) + 1e-6

        # Reset episode state
        self.path_history = [self.current_node]
        self.visited_nodes = {self.current_node}
        self.hop_count = 0
        self.total_delay = 0.0
        self.packet_dropped = False
        self.time_step = 0

        # Clear state history
        self.state_history.clear()

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute a routing action.

        Args:
            action: Index of the selected neighbor (0 to max_neighbors-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.time_step += 1

        # Get current neighbors and action mask
        neighbors = self.topology.get_neighbors(self.current_node)
        action_mask = self._get_action_mask()

        # Check if action is valid
        if action >= len(neighbors) or action_mask[action] == 0:
            # Invalid action
            reward = -self.invalid_action_penalty
            obs = self._get_observation()
            return obs, reward, False, False, self._get_info()

        # Get next hop
        next_node = neighbors[action]

        # Get link properties
        link = self.topology.get_link(self.current_node, next_node)

        if link is None or not link.is_active:
            # Link failed during transit
            reward = -self.invalid_action_penalty
            obs = self._get_observation()
            return obs, reward, False, False, self._get_info()

        # Check for packet loss
        if np.random.random() < link.loss_rate:
            self.packet_dropped = True
            reward = -self.beta
            obs = self._get_observation()
            info = self._get_info()
            return obs, reward, True, False, info

        # Update state
        prev_node = self.current_node
        self.current_node = next_node
        self.total_delay += link.delay
        self.hop_count += 1
        self.path_history.append(next_node)

        # Calculate reward BEFORE updating visited_nodes
        is_revisit = next_node in self.visited_nodes
        reward = self._calculate_reward(link, next_node, prev_node, is_revisit)
        
        # Now update visited nodes
        self.visited_nodes.add(next_node)

        # Check termination conditions
        terminated = (next_node == self.destination)
        if terminated:
            reward += self.success_bonus
            # [FIX] Bonus for efficient paths
            optimal_path, _ = self.topology.shortest_path(self.source, self.destination)
            if optimal_path:
                optimal_hops = len(optimal_path) - 1
                if self.hop_count <= optimal_hops * 1.5:
                    reward += 5.0  # Efficiency bonus

        # Check truncation (max hops exceeded)
        truncated = (self.hop_count >= self.max_hops)
        if truncated and not terminated:
            reward -= self.timeout_penalty

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _sample_source_node(self) -> int:
        """Sample a source node (prefer ground stations)."""
        ground_nodes = [
            nid for nid, node in self.topology.nodes.items()
            if node.node_type == NodeType.GROUND
        ]
        if ground_nodes:
            return np.random.choice(ground_nodes)
        return np.random.randint(0, self.topology.total_nodes)

    def _sample_destination_node(self, source: int) -> int:
        """Sample a destination different from source."""
        candidates = [nid for nid in self.topology.nodes if nid != source]
        
        # [FIX] Ensure destination is reachable
        reachable = []
        for c in candidates:
            path, _ = self.topology.shortest_path(source, c)
            if path:
                reachable.append(c)
        
        if reachable:
            return np.random.choice(reachable)
        return np.random.choice(candidates) if candidates else (source + 1) % self.topology.total_nodes

    def _get_observation(self) -> Dict:
        """
        Construct the observation dictionary.

        Returns:
            Dictionary containing state vector, action mask, and graph data
        """
        state = self._build_state_vector()

        # Update state history
        self.state_history.append(state.copy())

        return {
            'state': state,
            'action_mask': self._get_action_mask(),
            'node_features': self.topology.get_node_features(),
            'adjacency': self.topology.get_adjacency_matrix()
        }

    def _build_state_vector(self) -> np.ndarray:
        """Build the flat state vector."""
        components = []

        # Current node features (9 dim)
        current_features = self.topology.nodes[self.current_node].to_feature_vector()
        components.append(current_features)

        # Neighbor information
        neighbors = self.topology.get_neighbors(self.current_node)

        # Neighbor link features (5 * max_neighbors)
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                link = self.topology.get_link(self.current_node, neighbors[i])
                if link:
                    components.append(link.to_feature_vector())
                else:
                    components.append(np.zeros(self.link_feature_dim, dtype=np.float32))
            else:
                components.append(np.zeros(self.link_feature_dim, dtype=np.float32))

        # Neighbor node features (9 * max_neighbors)
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                components.append(self.topology.nodes[neighbors[i]].to_feature_vector())
            else:
                components.append(np.zeros(self.node_feature_dim, dtype=np.float32))

        # Target information (9 + 2 = 11 dim)
        dest_features = self.topology.nodes[self.destination].to_feature_vector()
        components.append(dest_features)

        # Distance to destination (normalized)
        current_pos = self.topology.nodes[self.current_node].position
        dest_pos = self.topology.nodes[self.destination].position
        distance = np.linalg.norm(current_pos - dest_pos)
        
        # [FIX] Use stored initial distance for consistent normalization
        normalized_distance = distance / self.initial_distance

        # Progress indicator (how close compared to start)
        progress = 1.0 - normalized_distance  # 0 at start, 1 at destination

        components.append(np.array([normalized_distance, progress], dtype=np.float32))

        return np.concatenate(components).astype(np.float32)

    def _get_action_mask(self) -> np.ndarray:
        """
        Get valid action mask.

        Actions are invalid if:
        - Neighbor doesn't exist
        - Would create an immediate loop (revisit recent node)
        - Link is inactive
        """
        neighbors = self.topology.get_neighbors(self.current_node)
        mask = np.zeros(self.max_neighbors, dtype=np.float32)

        for i, neighbor_id in enumerate(neighbors):
            if i >= self.max_neighbors:
                break

            # Always allow going to destination
            if neighbor_id == self.destination:
                mask[i] = 1.0
                continue

            # [FIX] Check if this would create a loop (visited in last 3 hops)
            # But be less restrictive to avoid getting stuck
            recent_path = self.path_history[-2:] if len(self.path_history) >= 2 else self.path_history
            if neighbor_id in recent_path:
                continue  # Mask out this action

            # Check link is active
            link = self.topology.get_link(self.current_node, neighbor_id)
            if link and link.is_active:
                mask[i] = 1.0

        # If no valid actions, allow all neighbors (fallback)
        if mask.sum() == 0 and len(neighbors) > 0:
            for i in range(min(len(neighbors), self.max_neighbors)):
                link = self.topology.get_link(self.current_node, neighbors[i])
                if link and link.is_active:
                    mask[i] = 1.0

        return mask

    def _calculate_reward(self, link, next_node: int, prev_node: int, is_revisit: bool) -> float:
        """
        Calculate the step reward with improved shaping.

        [FIXED VERSION] Key changes:
        - Full loop penalty (removed 0.5 multiplier)
        - Better progress reward scaling
        - Added hop efficiency consideration
        - Smoother reward signal

        Args:
            link: The link traversed
            next_node: The node we're moving to
            prev_node: The node we came from
            is_revisit: Whether next_node was already visited
        """
        reward = 0.0

        # Get positions
        prev_pos = self.topology.nodes[prev_node].position
        next_pos = self.topology.nodes[next_node].position
        dest_pos = self.topology.nodes[self.destination].position

        prev_dist = np.linalg.norm(prev_pos - dest_pos)
        next_dist = np.linalg.norm(next_pos - dest_pos)

        # === 1. Progress reward (main learning signal) ===
        # Proportional to distance improvement, normalized by initial distance
        distance_improvement = (prev_dist - next_dist) / self.initial_distance
        reward += distance_improvement * self.progress_reward * 10.0  # Scaled up

        # === 2. Proximity bonus (encourages getting closer) ===
        proximity_ratio = 1.0 - (next_dist / self.initial_distance)
        proximity_ratio = max(0, min(1, proximity_ratio))  # Clamp to [0,1]
        reward += proximity_ratio * 0.3

        # === 3. Destination neighborhood bonus ===
        if next_dist < self.initial_distance * 0.15:  # Within 15% of initial distance
            reward += 1.5
        elif next_dist < self.initial_distance * 0.3:  # Within 30%
            reward += 0.5

        # === 4. Delay penalty (small, normalized) ===
        reward -= self.alpha * 0.1 * link.delay / 100.0

        # === 5. Loop penalty [FIXED: removed 0.5 multiplier] ===
        if is_revisit:
            reward -= self.loop_penalty  # Full penalty now

        # === 6. Hop efficiency penalty (BALANCED) ===
        optimal_path, _ = self.topology.shortest_path(self.source, self.destination)
        expected_hops = len(optimal_path) - 1 if optimal_path else 5

        # 温和的超跳惩罚
        if self.hop_count > expected_hops:
            excess_hops = self.hop_count - expected_hops
            reward -= 0.2 * excess_hops  # 温和惩罚

        # === 7. Moving towards destination check ===
        if next_dist >= prev_dist and next_node != self.destination:
            reward -= 0.5  # 适中惩罚

        # === 8. [NEW] 高效路径奖励 ===
        if self.hop_count <= expected_hops + 2:
            reward += 0.5  # 奖励走得快的

        return reward

    def _get_info(self) -> Dict:
        """Get episode information."""
        # Compute optimal path for comparison
        optimal_path, optimal_delay = self.topology.shortest_path(
            self.source, self.destination
        )

        return {
            'source': self.source,
            'destination': self.destination,
            'current_node': self.current_node,
            'hop_count': self.hop_count,
            'total_delay': self.total_delay,
            'path_history': self.path_history.copy(),
            'packet_dropped': self.packet_dropped,
            'optimal_hops': len(optimal_path) - 1 if optimal_path else -1,
            'optimal_delay': optimal_delay,
            'success': self.current_node == self.destination,
            'initial_distance': self.initial_distance
        }

    def get_state_history(self) -> np.ndarray:
        """
        Get state history for Transformer input.

        Returns:
            Array of shape (history_length, state_dim)
        """
        if len(self.state_history) < self.history_length:
            # Pad with zeros
            padding = [
                np.zeros(self.state_dim, dtype=np.float32)
                for _ in range(self.history_length - len(self.state_history))
            ]
            return np.array(padding + list(self.state_history), dtype=np.float32)

        return np.array(list(self.state_history), dtype=np.float32)

    def get_graph_data(self) -> Dict:
        """
        Get graph data for GCN input.

        Returns:
            Dictionary with node features, edge index, etc.
        """
        return {
            'x': torch.tensor(self.topology.get_node_features(), dtype=torch.float32),
            'edge_index': self.topology.get_edge_index(),
            'edge_attr': torch.tensor(self.topology.get_edge_attr(), dtype=torch.float32),
            'current_node': self.current_node,
            'destination': self.destination
        }

    def get_local_graph_data(self) -> Dict:
        """
        Get local subgraph data (current node and neighbors).

        Returns:
            Dictionary with local node features and neighbor info
        """
        neighbors = self.topology.get_neighbors(self.current_node)

        # Current node features
        current_features = self.topology.nodes[self.current_node].to_feature_vector()

        # Neighbor features
        neighbor_features = []
        neighbor_mask = np.zeros(self.max_neighbors, dtype=np.float32)

        for i in range(self.max_neighbors):
            if i < len(neighbors):
                neighbor_features.append(
                    self.topology.nodes[neighbors[i]].to_feature_vector()
                )
                neighbor_mask[i] = 1.0
            else:
                neighbor_features.append(np.zeros(self.node_feature_dim, dtype=np.float32))

        return {
            'current_node_features': current_features,
            'neighbor_features': np.array(neighbor_features, dtype=np.float32),
            'neighbor_mask': neighbor_mask,
            'neighbors': neighbors
        }

    def render(self):
        """Render the environment (optional visualization)."""
        if self.render_mode == "human":
            self._render_text()

    def _render_text(self):
        """Text-based rendering."""
        print(f"\n{'='*50}")
        print(f"Step: {self.time_step}, Hop: {self.hop_count}")
        print(f"Current: {self.current_node} -> Destination: {self.destination}")
        print(f"Path: {' -> '.join(map(str, self.path_history))}")
        print(f"Total Delay: {self.total_delay:.2f} ms")
        print(f"Neighbors: {self.topology.get_neighbors(self.current_node)}")
        print(f"{'='*50}")

    def close(self):
        """Clean up resources."""
        pass


# Utility function for creating environment
def make_sagin_env(config_path: str = None, **kwargs) -> SAGINRoutingEnv:
    """
    Factory function to create SAGIN environment.

    Args:
        config_path: Path to YAML config file
        **kwargs: Override config values

    Returns:
        SAGINRoutingEnv instance
    """
    if config_path:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Apply overrides
    config.update(kwargs)

    return SAGINRoutingEnv(config)
