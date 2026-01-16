"""
SAGIN (Space-Air-Ground Integrated Network) topology modeling.

This module implements a three-layer network topology with:
- Space layer: LEO satellites
- Air layer: UAVs (Unmanned Aerial Vehicles)
- Ground layer: Ground stations and users
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx


class NodeType(Enum):
    """Network node types."""
    SATELLITE = 0  # LEO satellite
    UAV = 1        # Unmanned Aerial Vehicle
    GROUND = 2     # Ground station/user


@dataclass
class Node:
    """Network node representation."""
    id: int
    node_type: NodeType
    position: np.ndarray  # [x, y, z] coordinates in meters

    # Node state
    queue_length: float = 0.0      # Packets in queue
    energy: float = 1.0            # Remaining energy (0-1)
    cpu_frequency: float = 1e9     # Hz
    memory: float = 4e9            # Bytes

    # For UAV mobility
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_feature_vector(self) -> np.ndarray:
        """Convert node to feature vector for neural network input."""
        # One-hot encoding of node type (3 dims)
        type_onehot = np.zeros(3)
        type_onehot[self.node_type.value] = 1.0

        return np.concatenate([
            type_onehot,                          # 3 dims: node type
            self.position / 1e6,                  # 3 dims: normalized position
            [self.queue_length / 100],            # 1 dim: normalized queue
            [self.energy],                        # 1 dim: energy level
            [np.log10(self.cpu_frequency + 1) / 12],  # 1 dim: normalized CPU
        ]).astype(np.float32)  # Total: 9 dims

    def update_position(self, dt: float = 1.0):
        """Update position based on velocity."""
        self.position = self.position + self.velocity * dt


@dataclass
class Link:
    """Network link between two nodes."""
    source_id: int
    target_id: int
    bandwidth: float      # Hz
    delay: float          # milliseconds
    loss_rate: float      # Packet loss probability [0, 1]
    distance: float       # meters
    is_active: bool = True

    def to_feature_vector(self) -> np.ndarray:
        """Convert link to feature vector."""
        return np.array([
            self.bandwidth / 1e8,              # Normalized bandwidth
            self.delay / 100,                  # Normalized delay
            self.loss_rate,                    # Loss rate
            np.log10(self.distance + 1) / 6,   # Normalized distance
            float(self.is_active)              # Active status
        ], dtype=np.float32)  # Total: 5 dims


class SAGINTopology:
    """
    SAGIN three-layer network topology.

    Manages the network graph including nodes, links, and their dynamics.
    """

    def __init__(self, config: Dict):
        """
        Initialize the SAGIN topology.

        Args:
            config: Network configuration dictionary
        """
        self.config = config

        # Node counts
        self.num_satellites = config.get('num_satellites', 3)
        self.num_uavs = config.get('num_uavs', 6)
        self.num_ground = config.get('num_ground', 10)
        self.total_nodes = self.num_satellites + self.num_uavs + self.num_ground

        # Spatial parameters
        self.area_size = config.get('area_size', 10000.0)
        self.satellite_altitude = config.get('satellite_altitude', 550000.0)
        self.uav_altitude_range = config.get('uav_altitude_range', [100.0, 500.0])

        # Coverage parameters
        self.max_neighbors = config.get('max_neighbors', 8)
        self.satellite_coverage = config.get('satellite_coverage', 500000.0)
        self.uav_coverage = config.get('uav_coverage', 5000.0)

        # Bandwidth parameters
        self.g2u_bandwidth = config.get('g2u_bandwidth', 20e6)
        self.u2u_bandwidth = config.get('u2u_bandwidth', 10e6)
        self.u2s_bandwidth = config.get('u2s_bandwidth', 50e6)
        self.s2s_bandwidth = config.get('s2s_bandwidth', 100e6)

        # Mobility parameters
        self.uav_max_speed = config.get('uav_max_speed', 10.0)
        self.orbital_period = config.get('satellite_orbital_period', 5400)

        # Data structures
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[Tuple[int, int], Link] = {}
        self.adjacency_list: Dict[int, List[int]] = {}

        # Initialize topology
        self._initialize_nodes()
        self._establish_links()

    def _initialize_nodes(self):
        """Initialize all network nodes."""
        node_id = 0

        # Create satellite nodes (Space layer)
        for i in range(self.num_satellites):
            # Distribute satellites evenly in orbit
            angle = 2 * np.pi * i / self.num_satellites
            x = self.area_size / 2 + self.satellite_coverage * 0.3 * np.cos(angle)
            y = self.area_size / 2 + self.satellite_coverage * 0.3 * np.sin(angle)
            z = self.satellite_altitude

            self.nodes[node_id] = Node(
                id=node_id,
                node_type=NodeType.SATELLITE,
                position=np.array([x, y, z]),
                cpu_frequency=10e9,
                memory=16e9,
                energy=1.0
            )
            node_id += 1

        # Create UAV nodes (Air layer)
        for i in range(self.num_uavs):
            x = np.random.uniform(0, self.area_size)
            y = np.random.uniform(0, self.area_size)
            z = np.random.uniform(*self.uav_altitude_range)

            # Random initial velocity
            vx = np.random.uniform(-self.uav_max_speed, self.uav_max_speed)
            vy = np.random.uniform(-self.uav_max_speed, self.uav_max_speed)

            self.nodes[node_id] = Node(
                id=node_id,
                node_type=NodeType.UAV,
                position=np.array([x, y, z]),
                velocity=np.array([vx, vy, 0.0]),
                cpu_frequency=2e9,
                memory=4e9,
                energy=np.random.uniform(0.5, 1.0)
            )
            node_id += 1

        # Create ground nodes (Ground layer)
        for i in range(self.num_ground):
            x = np.random.uniform(0, self.area_size)
            y = np.random.uniform(0, self.area_size)
            z = 0.0

            self.nodes[node_id] = Node(
                id=node_id,
                node_type=NodeType.GROUND,
                position=np.array([x, y, z]),
                cpu_frequency=1e9,
                memory=2e9,
                energy=1.0  # Ground nodes have stable power
            )
            node_id += 1

        # Initialize adjacency list
        for nid in self.nodes:
            self.adjacency_list[nid] = []

    def _establish_links(self):
        """Establish links between nodes based on connectivity rules."""
        self.links.clear()

        for nid in self.nodes:
            self.adjacency_list[nid] = []

        # Check all node pairs
        for i in range(self.total_nodes):
            for j in range(i + 1, self.total_nodes):
                if self._can_establish_link(i, j):
                    self._create_link(i, j)

    def _can_establish_link(self, i: int, j: int) -> bool:
        """
        Check if a link can be established between two nodes.

        Args:
            i: First node ID
            j: Second node ID

        Returns:
            True if link is possible
        """
        node_i = self.nodes[i]
        node_j = self.nodes[j]
        distance = np.linalg.norm(node_i.position - node_j.position)

        type_i = node_i.node_type
        type_j = node_j.node_type

        # Satellite-Satellite: Always connected (inter-satellite links)
        if type_i == NodeType.SATELLITE and type_j == NodeType.SATELLITE:
            return True

        # Satellite-UAV: Within satellite coverage
        if {type_i, type_j} == {NodeType.SATELLITE, NodeType.UAV}:
            return distance < self.satellite_coverage

        # UAV-UAV: Within mutual coverage
        if type_i == NodeType.UAV and type_j == NodeType.UAV:
            return distance < self.uav_coverage * 2

        # UAV-Ground: Within UAV coverage
        if {type_i, type_j} == {NodeType.UAV, NodeType.GROUND}:
            return distance < self.uav_coverage

        # Ground-Ground: Adjacent nodes only
        if type_i == NodeType.GROUND and type_j == NodeType.GROUND:
            return distance < self.area_size / 5

        # Satellite-Ground: Not directly connected (must go through UAV)
        return False

    def _create_link(self, i: int, j: int):
        """Create a bidirectional link between two nodes."""
        node_i = self.nodes[i]
        node_j = self.nodes[j]
        distance = np.linalg.norm(node_i.position - node_j.position)

        # Determine bandwidth based on link type
        bandwidth = self._get_link_bandwidth(node_i.node_type, node_j.node_type)

        # Calculate propagation delay (speed of light)
        propagation_delay = distance / 3e8 * 1000  # ms

        # Add processing delay
        processing_delay = np.random.uniform(0.1, 0.5)  # ms
        total_delay = propagation_delay + processing_delay

        # Calculate loss rate (increases with distance)
        base_loss = 0.001
        distance_factor = distance / 1e6
        loss_rate = min(base_loss + distance_factor * 0.01, 0.1)

        # Create link
        link = Link(
            source_id=i,
            target_id=j,
            bandwidth=bandwidth,
            delay=total_delay,
            loss_rate=loss_rate,
            distance=distance,
            is_active=True
        )

        # Store bidirectional
        self.links[(i, j)] = link
        self.links[(j, i)] = link

        # Update adjacency list (respect max_neighbors)
        if len(self.adjacency_list[i]) < self.max_neighbors:
            self.adjacency_list[i].append(j)
        if len(self.adjacency_list[j]) < self.max_neighbors:
            self.adjacency_list[j].append(i)

    def _get_link_bandwidth(self, type_i: NodeType, type_j: NodeType) -> float:
        """Get bandwidth for a link based on node types."""
        if type_i == NodeType.SATELLITE and type_j == NodeType.SATELLITE:
            return self.s2s_bandwidth
        if {type_i, type_j} == {NodeType.SATELLITE, NodeType.UAV}:
            return self.u2s_bandwidth
        if type_i == NodeType.UAV and type_j == NodeType.UAV:
            return self.u2u_bandwidth
        return self.g2u_bandwidth

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Get active neighbors of a node.

        Args:
            node_id: Node ID

        Returns:
            List of neighbor node IDs
        """
        neighbors = []
        for n in self.adjacency_list.get(node_id, []):
            link = self.links.get((node_id, n))
            if link and link.is_active:
                neighbors.append(n)
        return neighbors

    def get_link(self, source: int, target: int) -> Optional[Link]:
        """Get link between two nodes."""
        return self.links.get((source, target))

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix."""
        adj = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for (i, j), link in self.links.items():
            if link.is_active:
                adj[i, j] = 1.0
        return adj

    def get_node_features(self) -> np.ndarray:
        """Get feature matrix for all nodes."""
        features = []
        for nid in range(self.total_nodes):
            features.append(self.nodes[nid].to_feature_vector())
        return np.array(features, dtype=np.float32)

    def get_edge_index(self) -> torch.LongTensor:
        """Get edge index in PyTorch Geometric format."""
        edges = []
        for (i, j), link in self.links.items():
            if link.is_active:
                edges.append([i, j])

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def get_edge_attr(self) -> np.ndarray:
        """Get edge attributes."""
        edge_features = []
        seen = set()

        for (i, j), link in self.links.items():
            if link.is_active and (i, j) not in seen and (j, i) not in seen:
                edge_features.append(link.to_feature_vector())
                seen.add((i, j))

        if not edge_features:
            return np.zeros((0, 5), dtype=np.float32)

        return np.array(edge_features, dtype=np.float32)

    def update_topology(self, time_step: int):
        """
        Update topology for dynamic simulation.

        Args:
            time_step: Current simulation time step
        """
        # Update UAV positions
        for nid, node in self.nodes.items():
            if node.node_type == NodeType.UAV:
                # Update position with boundary reflection
                node.update_position(dt=1.0)

                # Reflect at boundaries
                for dim in [0, 1]:  # x, y dimensions
                    if node.position[dim] < 0:
                        node.position[dim] = -node.position[dim]
                        node.velocity[dim] = -node.velocity[dim]
                    elif node.position[dim] > self.area_size:
                        node.position[dim] = 2 * self.area_size - node.position[dim]
                        node.velocity[dim] = -node.velocity[dim]

                # Random velocity perturbation
                node.velocity[:2] += np.random.uniform(-1, 1, 2)
                speed = np.linalg.norm(node.velocity[:2])
                if speed > self.uav_max_speed:
                    node.velocity[:2] *= self.uav_max_speed / speed

        # Update satellite positions (orbital motion)
        angular_speed = 2 * np.pi / self.orbital_period
        for nid, node in self.nodes.items():
            if node.node_type == NodeType.SATELLITE:
                sat_idx = nid  # Satellite IDs start from 0
                angle = angular_speed * time_step + 2 * np.pi * sat_idx / self.num_satellites
                r = self.satellite_coverage * 0.3
                cx, cy = self.area_size / 2, self.area_size / 2
                node.position[0] = cx + r * np.cos(angle)
                node.position[1] = cy + r * np.sin(angle)

        # Update queue lengths (random simulation)
        for nid, node in self.nodes.items():
            node.queue_length = max(0, node.queue_length + np.random.uniform(-5, 5))
            node.queue_length = min(100, node.queue_length)

        # Rebuild links based on new positions
        self._establish_links()

    def apply_link_failure(self, failure_prob: float = 0.05):
        """
        Apply random link failures.

        Args:
            failure_prob: Probability of each link failing
        """
        for key, link in self.links.items():
            if link.is_active and np.random.random() < failure_prob:
                link.is_active = False

    def shortest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        """
        Compute shortest path using Dijkstra's algorithm.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Tuple of (path as list of node IDs, total delay)
        """
        G = nx.Graph()

        for (i, j), link in self.links.items():
            if link.is_active and i < j:
                G.add_edge(i, j, weight=link.delay)

        try:
            path = nx.shortest_path(G, source, target, weight='weight')
            length = nx.shortest_path_length(G, source, target, weight='weight')
            return path, length
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], float('inf')

    def get_all_paths(self, source: int, target: int, cutoff: int = 10) -> List[List[int]]:
        """Get all simple paths up to a certain length."""
        G = nx.Graph()

        for (i, j), link in self.links.items():
            if link.is_active and i < j:
                G.add_edge(i, j)

        try:
            return list(nx.all_simple_paths(G, source, target, cutoff=cutoff))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        G = nx.Graph()

        for nid, node in self.nodes.items():
            G.add_node(
                nid,
                node_type=node.node_type.name,
                position=node.position.tolist(),
                queue_length=node.queue_length,
                energy=node.energy
            )

        for (i, j), link in self.links.items():
            if link.is_active and i < j:
                G.add_edge(
                    i, j,
                    bandwidth=link.bandwidth,
                    delay=link.delay,
                    loss_rate=link.loss_rate,
                    distance=link.distance
                )

        return G

    def get_node_type_indices(self) -> Dict[NodeType, List[int]]:
        """Get node indices grouped by type."""
        indices = {t: [] for t in NodeType}
        for nid, node in self.nodes.items():
            indices[node.node_type].append(nid)
        return indices

    def get_statistics(self) -> Dict:
        """Get topology statistics."""
        active_links = sum(1 for l in self.links.values() if l.is_active) // 2
        avg_degree = sum(len(self.get_neighbors(i)) for i in range(self.total_nodes)) / self.total_nodes

        delays = [l.delay for l in self.links.values() if l.is_active]
        avg_delay = np.mean(delays) if delays else 0

        return {
            'total_nodes': self.total_nodes,
            'num_satellites': self.num_satellites,
            'num_uavs': self.num_uavs,
            'num_ground': self.num_ground,
            'active_links': active_links,
            'avg_degree': avg_degree,
            'avg_delay': avg_delay
        }

    def reset(self):
        """Reset topology to initial state."""
        self._initialize_nodes()
        self._establish_links()
