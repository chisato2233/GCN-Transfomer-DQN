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
        info = self._get_info(compute_optimal=False)

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
            # Invalid action - increment hop count to eventually trigger truncation
            self.hop_count += 1
            reward = -self.invalid_action_penalty

            # [FIX] Truncate if too many invalid actions to prevent infinite loops
            truncated = (self.hop_count >= self.max_hops)
            obs = self._get_observation()
            return obs, reward, False, truncated, self._get_info(compute_optimal=False)

        # Get next hop
        next_node = neighbors[action]

        # Get link properties
        link = self.topology.get_link(self.current_node, next_node)

        if link is None or not link.is_active:
            # Link failed during transit - increment hop count
            self.hop_count += 1
            reward = -self.invalid_action_penalty

            # [FIX] Truncate if too many failed attempts
            truncated = (self.hop_count >= self.max_hops)
            obs = self._get_observation()
            return obs, reward, False, truncated, self._get_info(compute_optimal=False)

        # Check for packet loss - episode terminates
        if np.random.random() < link.loss_rate:
            self.packet_dropped = True
            reward = -self.beta
            obs = self._get_observation()
            info = self._get_info(compute_optimal=False)  # Skip for training speed
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
            # [FIXED v2] Efficiency bonus based on LOCAL information only
            # Reward for reaching destination quickly (without knowing optimal)
            # Use hop count relative to initial distance as proxy
            expected_hops_estimate = max(1, self.initial_distance / 500)  # Rough estimate
            if self.hop_count <= expected_hops_estimate * 2:
                reward += 3.0  # Good efficiency bonus
            elif self.hop_count <= expected_hops_estimate * 3:
                reward += 1.0  # Moderate efficiency

        # Check truncation (max hops exceeded)
        truncated = (self.hop_count >= self.max_hops)
        if truncated and not terminated:
            reward -= self.timeout_penalty

        # Get observation
        obs = self._get_observation()
        # [PERF FIX] Skip optimal path computation during training for speed
        # Only compute when episode ends (terminated or truncated) if needed for eval
        info = self._get_info(compute_optimal=False)

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

        # [PERF FIX] Use BFS for reachability check instead of shortest_path
        # This is much faster than calling Dijkstra for every candidate
        reachable = self._get_reachable_nodes(source)
        reachable_candidates = [c for c in candidates if c in reachable]

        if reachable_candidates:
            return np.random.choice(reachable_candidates)
        return np.random.choice(candidates) if candidates else (source + 1) % self.topology.total_nodes

    def _get_reachable_nodes(self, source: int) -> set:
        """Get all nodes reachable from source using BFS (fast)."""
        visited = {source}
        queue = [source]

        while queue:
            node = queue.pop(0)
            for neighbor in self.topology.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

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
        Calculate the step reward using ONLY LOCAL information.

        [V3 Enhanced] 空天地三层网络联合优化奖励函数

        Key features:
        - 基于局部可观测信息（无全局最短路径）
        - 考虑三层网络特性：
          * 能量感知（避免低电量 UAV）
          * 拥塞感知（避免高队列节点）
          * 层间切换成本

        Args:
            link: The link traversed
            next_node: The node we're moving to
            prev_node: The node we came from
            is_revisit: Whether next_node was already visited
        """
        reward = 0.0

        # Get node info
        prev_node_obj = self.topology.nodes[prev_node]
        next_node_obj = self.topology.nodes[next_node]

        # Get positions (LOCAL: coordinates are observable)
        prev_pos = prev_node_obj.position
        next_pos = next_node_obj.position
        dest_pos = self.topology.nodes[self.destination].position

        prev_dist = np.linalg.norm(prev_pos - dest_pos)
        next_dist = np.linalg.norm(next_pos - dest_pos)

        # === 1. Progress reward (main learning signal) ===
        distance_improvement = (prev_dist - next_dist) / self.initial_distance
        reward += distance_improvement * self.progress_reward * 10.0

        # === 2. Proximity bonus (encourages getting closer) ===
        proximity_ratio = 1.0 - (next_dist / self.initial_distance)
        proximity_ratio = max(0, min(1, proximity_ratio))
        reward += proximity_ratio * 0.3

        # === 3. Destination neighborhood bonus ===
        if next_dist < self.initial_distance * 0.15:
            reward += 1.5
        elif next_dist < self.initial_distance * 0.3:
            reward += 0.5

        # === 4. Delay penalty (LOCAL: link delay is observable) ===
        reward -= self.alpha * 0.1 * link.delay / 100.0

        # === 5. Loop penalty (LOCAL: agent knows its own path history) ===
        if is_revisit:
            reward -= self.loop_penalty

        # === 6. Hop count penalty ===
        if self.hop_count > 3:
            reward -= 0.1 * (self.hop_count - 3)

        # === 7. Moving away from destination penalty ===
        if next_dist >= prev_dist and next_node != self.destination:
            reward -= 0.5

        # ========== 三层网络联合优化奖励 ==========

        # === 8. UAV 能量感知 ===
        # 惩罚选择低电量的 UAV，鼓励保护 UAV 电量
        if next_node_obj.node_type == NodeType.UAV:
            if next_node_obj.energy < 0.3:
                reward -= 1.0  # 严重低电量惩罚
            elif next_node_obj.energy < 0.5:
                reward -= 0.3  # 中等低电量惩罚

        # === 9. 拥塞感知 ===
        # 惩罚选择高队列长度的节点
        queue_ratio = next_node_obj.queue_length / 100.0
        if queue_ratio > 0.7:
            reward -= 0.5 * queue_ratio

        # === 10. 层间切换考虑 ===
        # 跨层切换有额外开销，但有时是必要的
        if prev_node_obj.node_type != next_node_obj.node_type:
            # 小惩罚鼓励同层路由（减少切换开销）
            reward -= 0.1

            # 但如果是上升到卫星层且距离很远，给予奖励（卫星覆盖广）
            if next_node_obj.node_type == NodeType.SATELLITE and next_dist > self.initial_distance * 0.5:
                reward += 0.3  # 长距离时卫星路由更高效

        # === 11. 带宽奖励 ===
        # 高带宽链路更好
        if link.bandwidth > 50e6:  # > 50 Mbps
            reward += 0.2
        elif link.bandwidth > 20e6:  # > 20 Mbps
            reward += 0.1

        return reward

    def _get_info(self, compute_optimal: bool = False) -> Dict:
        """
        Get episode information.

        Args:
            compute_optimal: If True, compute optimal path (expensive).
                           Only set True at episode end for evaluation.
        """
        info = {
            'source': self.source,
            'destination': self.destination,
            'current_node': self.current_node,
            'hop_count': self.hop_count,
            'total_delay': self.total_delay,
            'path_history': self.path_history.copy(),
            'packet_dropped': self.packet_dropped,
            'success': self.current_node == self.destination,
            'initial_distance': self.initial_distance
        }

        # [PERF FIX] Only compute optimal path when requested (expensive!)
        # Training doesn't need optimal path info on every step
        if compute_optimal:
            optimal_path, optimal_delay = self.topology.shortest_path(
                self.source, self.destination
            )
            info['optimal_hops'] = len(optimal_path) - 1 if optimal_path else -1
            info['optimal_delay'] = optimal_delay
        else:
            info['optimal_hops'] = -1
            info['optimal_delay'] = -1

        return info

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

    def get_topology_aware_features(self) -> Dict:
        """
        Get topology-aware features for SAGIN three-layer joint optimization.

        [V3 ENHANCED] Features for Space-Air-Ground Integrated Network routing:

        Routing features (8 dims):
        - Distance to destination (normalized)
        - Distance improvement vs current node
        - Neighbor degree (connectivity)
        - 2-hop lookahead (min distance through neighbor's neighbors)
        - Is destination flag
        - Link delay (normalized)
        - Link bandwidth (normalized)
        - Was visited flag (loop detection)

        Three-layer features (6 dims):
        - Node type one-hot: [is_satellite, is_uav, is_ground]
        - Energy level (important for UAV)
        - Queue length / congestion
        - Layer transition indicator (cross-layer hop)

        Returns:
            Dictionary with:
            - neighbor_topology_features: [max_neighbors, 14] routing + layer features
            - neighbor_mask: [max_neighbors] valid neighbor mask
            - neighbors: list of neighbor IDs
        """
        neighbors = self.topology.get_neighbors(self.current_node)
        dest_pos = self.topology.nodes[self.destination].position
        current_pos = self.topology.nodes[self.current_node].position
        current_dist = np.linalg.norm(current_pos - dest_pos)
        current_node_type = self.topology.nodes[self.current_node].node_type

        # Topology feature dimension = 14 (8 routing + 6 layer)
        topology_features = []
        neighbor_mask = np.zeros(self.max_neighbors, dtype=np.float32)

        # Get max degree for normalization
        max_degree = max(len(self.topology.get_neighbors(n)) for n in self.topology.nodes)
        max_degree = max(max_degree, 1)  # Avoid division by zero

        for i in range(self.max_neighbors):
            if i < len(neighbors):
                neighbor_id = neighbors[i]
                neighbor_node = self.topology.nodes[neighbor_id]
                neighbor_pos = neighbor_node.position
                neighbor_dist = np.linalg.norm(neighbor_pos - dest_pos)

                # Get link properties
                link = self.topology.get_link(self.current_node, neighbor_id)
                link_delay = link.delay if link else 0.0
                link_bandwidth = link.bandwidth if link else 0.0

                # Get neighbor's neighbors for 2-hop lookahead
                neighbor_neighbors = self.topology.get_neighbors(neighbor_id)
                neighbor_degree = len(neighbor_neighbors)

                # 2-hop lookahead: find min distance through this neighbor
                min_2hop_dist = neighbor_dist  # Default to 1-hop distance
                for nn in neighbor_neighbors:
                    if nn != self.current_node:  # Don't go back
                        nn_pos = self.topology.nodes[nn].position
                        nn_dist = np.linalg.norm(nn_pos - dest_pos)
                        min_2hop_dist = min(min_2hop_dist, nn_dist)

                # === Three-layer network features ===
                # Node type one-hot [satellite, uav, ground]
                type_onehot = [0.0, 0.0, 0.0]
                type_onehot[neighbor_node.node_type.value] = 1.0

                # Energy level (critical for UAV selection)
                energy = neighbor_node.energy

                # Queue length / congestion indicator
                queue_congestion = min(neighbor_node.queue_length / 100.0, 1.0)

                # Layer transition: 1 if crossing layers, 0 if same layer
                is_layer_transition = 1.0 if neighbor_node.node_type != current_node_type else 0.0

                # Build feature vector (14 dimensions)
                features = np.array([
                    # === Routing features (8 dims) ===
                    # 1. Distance to target (normalized by initial distance)
                    neighbor_dist / self.initial_distance,
                    # 2. Distance improvement (positive = getting closer)
                    (current_dist - neighbor_dist) / self.initial_distance,
                    # 3. Neighbor degree (normalized)
                    neighbor_degree / max_degree,
                    # 4. Min 2-hop distance (normalized)
                    min_2hop_dist / self.initial_distance,
                    # 5. Is destination (binary)
                    1.0 if neighbor_id == self.destination else 0.0,
                    # 6. Link delay (normalized, typical range 0-10ms)
                    min(link_delay / 10.0, 1.0),
                    # 7. Link bandwidth (normalized, log scale)
                    np.log10(link_bandwidth + 1) / 9.0 if link_bandwidth > 0 else 0.0,
                    # 8. Was visited (loop indicator)
                    1.0 if neighbor_id in self.visited_nodes else 0.0,
                    # === Three-layer features (6 dims) ===
                    # 9-11. Node type one-hot [satellite, uav, ground]
                    type_onehot[0],  # is_satellite
                    type_onehot[1],  # is_uav
                    type_onehot[2],  # is_ground
                    # 12. Energy level (0-1, important for UAV)
                    energy,
                    # 13. Queue congestion (0-1)
                    queue_congestion,
                    # 14. Layer transition indicator
                    is_layer_transition,
                ], dtype=np.float32)

                topology_features.append(features)
                neighbor_mask[i] = 1.0
            else:
                # Padding for non-existent neighbors
                topology_features.append(np.zeros(14, dtype=np.float32))

        return {
            'neighbor_topology_features': np.array(topology_features, dtype=np.float32),
            'neighbor_mask': neighbor_mask,
            'neighbors': neighbors
        }

    def get_simplified_history(self) -> np.ndarray:
        """
        Get simplified state history for Transformer.

        Instead of full 132-dim state, uses key routing features only:
        - distance_to_target (normalized)
        - distance_improvement (from previous step)
        - is_loop_step (revisited a node)
        - node_type (one-hot: satellite, UAV, ground)

        Returns:
            Array of shape (history_length, 6) with simplified features
        """
        simplified_dim = 6  # distance, improvement, is_loop, type_sat, type_uav, type_ground

        if len(self.path_history) == 0:
            return np.zeros((self.history_length, simplified_dim), dtype=np.float32)

        dest_pos = self.topology.nodes[self.destination].position
        history = []

        # Build history from path
        prev_dist = self.initial_distance
        visited_so_far = set()

        for step_idx, node_id in enumerate(self.path_history):
            node = self.topology.nodes[node_id]
            node_pos = node.position
            current_dist = np.linalg.norm(node_pos - dest_pos)

            # Distance improvement
            improvement = (prev_dist - current_dist) / self.initial_distance
            prev_dist = current_dist

            # Loop detection
            is_loop = 1.0 if node_id in visited_so_far else 0.0
            visited_so_far.add(node_id)

            # Node type one-hot
            type_onehot = [0.0, 0.0, 0.0]
            type_onehot[node.node_type.value] = 1.0

            features = np.array([
                current_dist / self.initial_distance,  # Normalized distance
                improvement,                            # Distance improvement
                is_loop,                               # Loop indicator
                type_onehot[0],                        # Is satellite
                type_onehot[1],                        # Is UAV
                type_onehot[2],                        # Is ground
            ], dtype=np.float32)

            history.append(features)

        # Pad or truncate to history_length
        if len(history) < self.history_length:
            padding = [np.zeros(simplified_dim, dtype=np.float32)
                      for _ in range(self.history_length - len(history))]
            history = padding + history
        else:
            history = history[-self.history_length:]

        return np.array(history, dtype=np.float32)

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
