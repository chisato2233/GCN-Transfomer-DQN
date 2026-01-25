"""
Traditional Routing Protocol Baselines.

References:
- OSPF: RFC 2328 (Dijkstra-based shortest path)
- AODV: RFC 3561 (Ad-hoc On-demand Distance Vector)
- AI-Enhanced Routing Protocols (2024):
  https://ictactjournals.in/paper/IJCT_Vol_15_Iss_3_Paper_10_3300_3306.pdf

这些是传统路由协议的模拟实现，用于与DRL方法对比。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import heapq
from collections import defaultdict


class OSPFRouter:
    """
    OSPF-like Shortest Path First Router.

    使用Dijkstra算法计算最短路径。
    这是"全局信息"基线，代表理论最优（如果链路权重固定）。

    注意：这个方法使用全局拓扑信息，在实际分布式场景中
    需要通过LSA泛洪获取，有延迟和开销。
    """

    def __init__(self, env):
        """
        Initialize OSPF router.

        Args:
            env: SAGINRoutingEnv environment
        """
        self.env = env
        self.path_cache = {}

    def compute_shortest_path(self,
                               source: int,
                               destination: int,
                               metric: str = 'delay') -> Tuple[List[int], float]:
        """
        Compute shortest path using Dijkstra's algorithm.

        Args:
            source: Source node
            destination: Destination node
            metric: 'delay', 'hops', or 'bandwidth'

        Returns:
            (path, cost) tuple
        """
        cache_key = (source, destination, metric)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]

        # Dijkstra's algorithm
        distances = {source: 0}
        predecessors = {source: None}
        pq = [(0, source)]
        visited = set()

        while pq:
            dist, node = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)

            if node == destination:
                break

            neighbors = self.env.topology.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                link = self.env.topology.get_link(node, neighbor)
                if not link or not link.is_active:
                    continue

                # Compute edge weight based on metric
                if metric == 'delay':
                    weight = link.delay
                elif metric == 'hops':
                    weight = 1
                elif metric == 'bandwidth':
                    # Inverse bandwidth (higher bandwidth = lower cost)
                    weight = 1.0 / (link.bandwidth + 1e-6)
                else:
                    weight = link.delay

                new_dist = dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if destination not in predecessors:
            return [], float('inf')

        path = []
        node = destination
        while node is not None:
            path.append(node)
            node = predecessors[node]
        path.reverse()

        result = (path, distances.get(destination, float('inf')))
        self.path_cache[cache_key] = result
        return result

    def get_next_hop(self,
                     current: int,
                     destination: int,
                     neighbors: List[int],
                     action_mask: np.ndarray) -> int:
        """
        Get next hop action using shortest path.

        Args:
            current: Current node
            destination: Destination node
            neighbors: Available neighbors
            action_mask: Valid action mask

        Returns:
            Action index
        """
        path, _ = self.compute_shortest_path(current, destination)

        if len(path) >= 2:
            next_node = path[1]
            for i, neighbor in enumerate(neighbors):
                if neighbor == next_node and i < len(action_mask) and action_mask[i] > 0:
                    return i

        # Fallback to random valid action
        valid_indices = np.where(action_mask > 0)[0]
        return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0

    def clear_cache(self):
        """Clear path cache (call when topology changes)."""
        self.path_cache = {}


class ECMPRouter(OSPFRouter):
    """
    Equal-Cost Multi-Path (ECMP) Router.

    When multiple shortest paths exist, randomly select among them.
    This provides basic load balancing.
    """

    def compute_all_shortest_paths(self,
                                    source: int,
                                    destination: int) -> List[List[int]]:
        """
        Find all shortest paths using modified Dijkstra.

        Returns:
            List of all shortest paths
        """
        # Standard Dijkstra to find shortest distance
        distances = {source: 0}
        pq = [(0, source)]
        visited = set()

        while pq:
            dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)

            neighbors = self.env.topology.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                link = self.env.topology.get_link(node, neighbor)
                if not link or not link.is_active:
                    continue
                new_dist = dist + link.delay
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

        if destination not in distances:
            return []

        # BFS to find all shortest paths
        all_paths = []
        queue = [[source]]

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node == destination:
                all_paths.append(path)
                continue

            if len(path) > len(distances):  # Prevent infinite loops
                continue

            neighbors = self.env.topology.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor in path:  # Avoid cycles
                    continue
                link = self.env.topology.get_link(node, neighbor)
                if not link or not link.is_active:
                    continue
                # Check if this edge is on a shortest path
                if neighbor in distances:
                    expected_dist = distances[node] + link.delay
                    if abs(expected_dist - distances[neighbor]) < 1e-6:
                        queue.append(path + [neighbor])

        return all_paths

    def get_next_hop(self,
                     current: int,
                     destination: int,
                     neighbors: List[int],
                     action_mask: np.ndarray) -> int:
        """Get next hop with ECMP load balancing."""
        paths = self.compute_all_shortest_paths(current, destination)

        if paths:
            # Randomly select one of the shortest paths
            path = paths[np.random.randint(len(paths))]
            if len(path) >= 2:
                next_node = path[1]
                for i, neighbor in enumerate(neighbors):
                    if neighbor == next_node and i < len(action_mask) and action_mask[i] > 0:
                        return i

        return super().get_next_hop(current, destination, neighbors, action_mask)


class AODVRouter:
    """
    AODV-like Reactive Router.

    Ad-hoc On-demand Distance Vector protocol simulation.
    Routes are discovered on-demand through route request flooding.

    This simulates the behavior without actual message passing.
    """

    def __init__(self, env, route_timeout: int = 100):
        """
        Initialize AODV router.

        Args:
            env: SAGINRoutingEnv environment
            route_timeout: Steps before route expires
        """
        self.env = env
        self.route_timeout = route_timeout

        # Routing table: {destination: (next_hop, hop_count, timestamp)}
        self.routing_table: Dict[int, Dict[int, Tuple[int, int, int]]] = defaultdict(dict)
        self.current_step = 0

    def route_discovery(self, source: int, destination: int) -> Optional[int]:
        """
        Simulate AODV route discovery (RREQ/RREP).

        In real AODV, this would involve flooding RREQ packets.
        Here we simulate by running BFS from source.

        Args:
            source: Source node
            destination: Destination node

        Returns:
            Next hop from source, or None if no route
        """
        # BFS to find path
        queue = [(source, [source])]
        visited = {source}

        while queue:
            node, path = queue.pop(0)

            if node == destination:
                # Route found, update routing tables along path
                for i in range(len(path) - 1):
                    n = path[i]
                    next_hop = path[i + 1]
                    hop_count = len(path) - 1 - i
                    self.routing_table[n][destination] = (next_hop, hop_count, self.current_step)

                return path[1] if len(path) > 1 else None

            neighbors = self.env.topology.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    link = self.env.topology.get_link(node, neighbor)
                    if link and link.is_active:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))

        return None

    def get_next_hop(self,
                     current: int,
                     destination: int,
                     neighbors: List[int],
                     action_mask: np.ndarray) -> int:
        """
        Get next hop using AODV routing table or route discovery.
        """
        self.current_step += 1

        # Check if we have a valid route
        if destination in self.routing_table[current]:
            next_hop, hop_count, timestamp = self.routing_table[current][destination]
            if self.current_step - timestamp < self.route_timeout:
                # Route is still valid
                for i, neighbor in enumerate(neighbors):
                    if neighbor == next_hop and i < len(action_mask) and action_mask[i] > 0:
                        return i

        # Route discovery needed
        next_hop = self.route_discovery(current, destination)

        if next_hop is not None:
            for i, neighbor in enumerate(neighbors):
                if neighbor == next_hop and i < len(action_mask) and action_mask[i] > 0:
                    return i

        # Fallback
        valid_indices = np.where(action_mask > 0)[0]
        return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0


class LoadAwareRouter(OSPFRouter):
    """
    Load-Aware Shortest Path Router.

    Extension of OSPF that considers link load/congestion.
    Similar to PARP (Predictive Adaptive Routing Protocol) from 2024.

    Reference: AI-Enhanced Routing Protocols (2024)
    """

    def __init__(self, env, load_weight: float = 0.3):
        """
        Args:
            env: Environment
            load_weight: Weight for load factor in cost computation
        """
        super().__init__(env)
        self.load_weight = load_weight

    def compute_shortest_path(self,
                               source: int,
                               destination: int,
                               metric: str = 'combined') -> Tuple[List[int], float]:
        """
        Compute path considering both delay and load.
        """
        if metric != 'combined':
            return super().compute_shortest_path(source, destination, metric)

        # Dijkstra with combined metric
        distances = {source: 0}
        predecessors = {source: None}
        pq = [(0, source)]
        visited = set()

        while pq:
            dist, node = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)

            if node == destination:
                break

            neighbors = self.env.topology.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                link = self.env.topology.get_link(node, neighbor)
                if not link or not link.is_active:
                    continue

                # Combined cost: delay + load_weight * congestion
                neighbor_node = self.env.topology.nodes[neighbor]
                congestion = neighbor_node.queue_length / 100.0  # Normalized
                weight = link.delay * (1 + self.load_weight * congestion)

                new_dist = dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        if destination not in predecessors:
            return [], float('inf')

        path = []
        node = destination
        while node is not None:
            path.append(node)
            node = predecessors[node]
        path.reverse()

        return (path, distances.get(destination, float('inf')))
