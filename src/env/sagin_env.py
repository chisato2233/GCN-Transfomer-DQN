"""
SAGIN Intelligent Routing Environment.

A Gymnasium environment for training DRL agents to make routing decisions
in Space-Air-Ground Integrated Networks.
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
        self.success_bonus = reward_config.get('success_bonus', 10.0)
        self.loop_penalty = reward_config.get('loop_penalty', 5.0)
        self.timeout_penalty = reward_config.get('timeout_penalty', 1.0)
        self.invalid_action_penalty = reward_config.get('invalid_action_penalty', 1.0)
        self.progress_reward = reward_config.get('progress_reward', 0.1)

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
        self.total_delay += link.delay
        self.hop_count += 1
        self.path_history.append(next_node)
        self.visited_nodes.add(next_node)

        # Calculate reward
        reward = self._calculate_reward(link, next_node)

        # Check termination conditions
        terminated = (next_node == self.destination)
        if terminated:
            reward += self.success_bonus

        # Check truncation (max hops exceeded)
        truncated = (self.hop_count >= self.max_hops)
        if truncated and not terminated:
            reward -= self.timeout_penalty

        # Update current node
        self.current_node = next_node

        # Update topology dynamics periodically
        if self.time_step % self.topology_update_freq == 0:
            self.topology.update_topology(self.time_step)

        # Occasional random link failure
        if np.random.random() < self.link_failure_prob / 10:
            self.topology.apply_link_failure(self.link_failure_prob)

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _sample_source_node(self) -> int:
        """Sample a source node (prefer ground nodes)."""
        ground_nodes = [
            nid for nid, node in self.topology.nodes.items()
            if node.node_type == NodeType.GROUND
        ]

        # 70% chance to start from ground node
        if ground_nodes and np.random.random() < 0.7:
            return int(np.random.choice(ground_nodes))

        return int(np.random.randint(self.topology.total_nodes))

    def _sample_destination_node(self, source: int) -> int:
        """Sample a destination node (different from source, prefer ground)."""
        ground_nodes = [
            nid for nid, node in self.topology.nodes.items()
            if node.node_type == NodeType.GROUND and nid != source
        ]

        # 70% chance for ground destination
        if ground_nodes and np.random.random() < 0.7:
            return int(np.random.choice(ground_nodes))

        # Otherwise random (excluding source)
        candidates = [i for i in range(self.topology.total_nodes) if i != source]
        return int(np.random.choice(candidates))

    def _get_observation(self) -> Dict:
        """Construct the observation dictionary."""
        # Current node features
        current_features = self.topology.nodes[self.current_node].to_feature_vector()

        # Get neighbors
        neighbors = self.topology.get_neighbors(self.current_node)

        # Neighbor link and node features
        neighbor_link_features = []
        neighbor_node_features = []

        for i in range(self.max_neighbors):
            if i < len(neighbors):
                neighbor_id = neighbors[i]
                link = self.topology.get_link(self.current_node, neighbor_id)

                if link:
                    neighbor_link_features.append(link.to_feature_vector())
                else:
                    neighbor_link_features.append(np.zeros(self.link_feature_dim, dtype=np.float32))

                neighbor_node_features.append(
                    self.topology.nodes[neighbor_id].to_feature_vector()
                )
            else:
                # Padding for non-existent neighbors
                neighbor_link_features.append(np.zeros(self.link_feature_dim, dtype=np.float32))
                neighbor_node_features.append(np.zeros(self.node_feature_dim, dtype=np.float32))

        # Destination node features
        dest_features = self.topology.nodes[self.destination].to_feature_vector()

        # Distance and progress info
        current_pos = self.topology.nodes[self.current_node].position
        dest_pos = self.topology.nodes[self.destination].position
        distance = np.linalg.norm(current_pos - dest_pos)
        hop_progress = self.hop_count / self.max_hops

        # Combine into state vector
        state = np.concatenate([
            current_features,
            np.concatenate(neighbor_link_features),
            np.concatenate(neighbor_node_features),
            dest_features,
            np.array([distance / 1e6, hop_progress], dtype=np.float32)
        ]).astype(np.float32)

        # Update history
        self.state_history.append(state.copy())

        return {
            'state': state,
            'action_mask': self._get_action_mask(),
            'node_features': self.topology.get_node_features(),
            'adjacency': self.topology.get_adjacency_matrix()
        }

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

            # Check if this would create a loop (visited in last 3 hops)
            recent_path = self.path_history[-3:] if len(self.path_history) >= 3 else self.path_history
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

    def _calculate_reward(self, link, next_node: int) -> float:
        """
        Calculate the step reward with improved shaping.

        Key improvements:
        1. Stronger distance-based progress reward (proportional to improvement)
        2. Hop efficiency bonus (prefer shorter paths)
        3. Reduced penalties to avoid overly negative rewards
        """
        reward = 0.0

        # Get positions
        current_pos = self.topology.nodes[self.current_node].position
        next_pos = self.topology.nodes[next_node].position
        dest_pos = self.topology.nodes[self.destination].position

        current_dist = np.linalg.norm(current_pos - dest_pos)
        next_dist = np.linalg.norm(next_pos - dest_pos)

        # Normalize distance by initial source-destination distance
        source_pos = self.topology.nodes[self.source].position
        initial_dist = np.linalg.norm(source_pos - dest_pos) + 1e-6

        # 1. Progress reward (proportional to distance improvement)
        distance_improvement = (current_dist - next_dist) / initial_dist
        reward += distance_improvement * 5.0  # Scaled progress reward

        # 2. Proximity bonus (closer to destination = higher bonus)
        proximity_ratio = 1.0 - (next_dist / initial_dist)
        reward += proximity_ratio * 0.5  # Small proximity bonus

        # 3. Reaching destination neighborhood bonus
        if next_dist < initial_dist * 0.1:  # Within 10% of initial distance
            reward += 1.0

        # 4. Light delay penalty (normalized, reduced weight)
        reward -= self.alpha * 0.3 * link.delay / 100.0

        # 5. Loop penalty (revisiting nodes) - reduced
        if next_node in self.visited_nodes:
            reward -= self.loop_penalty * 0.5

        # 6. Hop efficiency: penalize long paths lightly
        if self.hop_count > 10:
            reward -= 0.1 * (self.hop_count - 10) / self.max_hops

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
            'success': self.current_node == self.destination
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
