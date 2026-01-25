"""
Real-World Topology Builder for SAGIN Networks.

Builds realistic Space-Air-Ground Integrated Network topologies using:
- Real satellite TLE data (Starlink, Iridium, OneWeb)
- Ground station locations
- Simulated UAV relay networks

Data sources:
- Satellites: CelesTrak TLE data
- Ground stations: ground_stations.json
- UAV trajectories: uav_trajectories.json
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .tle_parser import TLEParser


@dataclass
class NetworkNode:
    """Network node with real-world parameters."""
    id: int
    name: str
    node_type: str  # 'satellite', 'uav', 'ground'
    lat: float
    lon: float
    altitude_m: float
    x: float  # ECEF x (meters)
    y: float  # ECEF y (meters)
    z: float  # ECEF z (meters)

    # Type-specific parameters
    energy: float = 1.0
    bandwidth_hz: float = 100e6
    queue_capacity: int = 1000
    coverage_radius_m: float = 500000

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.node_type,
            'position': {'lat': self.lat, 'lon': self.lon, 'alt': self.altitude_m},
            'ecef': {'x': self.x, 'y': self.y, 'z': self.z},
            'energy': self.energy,
            'bandwidth': self.bandwidth_hz,
            'coverage_radius': self.coverage_radius_m
        }


class RealWorldTopologyBuilder:
    """
    Builds SAGIN topologies from real-world data.

    Usage:
        builder = RealWorldTopologyBuilder('data')
        topology = builder.build_topology(
            num_satellites=20,
            num_uavs=10,
            num_ground=15,
            region={'lat_range': (20, 50), 'lon_range': (100, 130)}
        )
    """

    # Physical constants
    EARTH_RADIUS_M = 6371000
    SPEED_OF_LIGHT = 299792458  # m/s

    # Link parameters by type
    LINK_PARAMS = {
        ('satellite', 'satellite'): {
            'bandwidth': 100e6,  # 100 MHz ISL
            'base_delay_ms': 5,
            'loss_rate': 0.001
        },
        ('satellite', 'uav'): {
            'bandwidth': 50e6,
            'base_delay_ms': 10,
            'loss_rate': 0.01
        },
        ('satellite', 'ground'): {
            'bandwidth': 50e6,
            'base_delay_ms': 20,
            'loss_rate': 0.005
        },
        ('uav', 'uav'): {
            'bandwidth': 20e6,
            'base_delay_ms': 2,
            'loss_rate': 0.02
        },
        ('uav', 'ground'): {
            'bandwidth': 20e6,
            'base_delay_ms': 1,
            'loss_rate': 0.01
        },
        ('ground', 'ground'): {
            'bandwidth': 1e9,  # 1 Gbps fiber
            'base_delay_ms': 0.5,
            'loss_rate': 0.0001
        }
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.tle_parser = TLEParser(data_dir)
        self.tle_parser.load_all_constellations()

        # Load ground stations
        self.ground_stations = self._load_ground_stations()

        # Load UAV trajectories
        self.uav_config = self._load_uav_config()

    def _load_ground_stations(self) -> List[Dict]:
        """Load ground station data."""
        gs_file = self.data_dir / 'ground_stations.json'
        if gs_file.exists():
            with open(gs_file, 'r') as f:
                data = json.load(f)
                return data.get('stations', [])
        return []

    def _load_uav_config(self) -> Dict:
        """Load UAV trajectory configuration."""
        uav_file = self.data_dir / 'uav_trajectories.json'
        if uav_file.exists():
            with open(uav_file, 'r') as f:
                return json.load(f)
        return {}

    def _latlon_to_ecef(self, lat: float, lon: float, alt_m: float) -> Tuple[float, float, float]:
        """Convert lat/lon/altitude to ECEF coordinates."""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        r = self.EARTH_RADIUS_M + alt_m

        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)

        return x, y, z

    def _compute_distance(self, node1: NetworkNode, node2: NetworkNode) -> float:
        """Compute 3D distance between nodes in meters."""
        return np.sqrt(
            (node1.x - node2.x) ** 2 +
            (node1.y - node2.y) ** 2 +
            (node1.z - node2.z) ** 2
        )

    def _in_region(self, lat: float, lon: float, region: Dict) -> bool:
        """Check if coordinates are within specified region."""
        lat_range = region.get('lat_range', (-90, 90))
        lon_range = region.get('lon_range', (-180, 180))
        return lat_range[0] <= lat <= lat_range[1] and lon_range[0] <= lon <= lon_range[1]

    def build_topology(
        self,
        num_satellites: int = 20,
        num_uavs: int = 10,
        num_ground: int = 15,
        region: Optional[Dict] = None,
        time_offset_s: float = 0,
        constellation: str = 'starlink'
    ) -> Dict:
        """
        Build a realistic SAGIN topology.

        Args:
            num_satellites: Number of satellites to include
            num_uavs: Number of UAVs to include
            num_ground: Number of ground stations to include
            region: Geographic region filter {'lat_range': (min, max), 'lon_range': (min, max)}
            time_offset_s: Time offset for satellite positions
            constellation: Satellite constellation to use

        Returns:
            Dictionary with nodes, links, and network statistics
        """
        if region is None:
            region = {'lat_range': (20, 50), 'lon_range': (100, 135)}  # Default: China region

        nodes: List[NetworkNode] = []
        node_id = 0

        # 1. Add satellites
        sat_positions = self.tle_parser.get_satellite_positions(
            constellation, time_offset_s, num_satellites=num_satellites * 3
        )

        # Filter satellites in region (approximate - satellites move)
        sats_in_region = []
        for pos in sat_positions:
            # Satellites cover a wide area, so use relaxed region check
            if self._in_region(pos['lat'], pos['lon'], {
                'lat_range': (region['lat_range'][0] - 30, region['lat_range'][1] + 30),
                'lon_range': (region['lon_range'][0] - 30, region['lon_range'][1] + 30)
            }):
                sats_in_region.append(pos)

        for pos in sats_in_region[:num_satellites]:
            x, y, z = self._latlon_to_ecef(pos['lat'], pos['lon'], pos['altitude_km'] * 1000)
            nodes.append(NetworkNode(
                id=node_id,
                name=pos['name'],
                node_type='satellite',
                lat=pos['lat'],
                lon=pos['lon'],
                altitude_m=pos['altitude_km'] * 1000,
                x=x, y=y, z=z,
                energy=1.0,
                bandwidth_hz=100e6,
                coverage_radius_m=pos['altitude_km'] * 1000 * 0.8  # ~80% of altitude
            ))
            node_id += 1

        # 2. Add ground stations
        gs_in_region = [
            gs for gs in self.ground_stations
            if self._in_region(gs['lat'], gs['lon'], region)
        ]

        for gs in gs_in_region[:num_ground]:
            x, y, z = self._latlon_to_ecef(gs['lat'], gs['lon'], 0)
            nodes.append(NetworkNode(
                id=node_id,
                name=gs['name'],
                node_type='ground',
                lat=gs['lat'],
                lon=gs['lon'],
                altitude_m=0,
                x=x, y=y, z=z,
                energy=1.0,
                bandwidth_hz=1e9 if gs.get('type') == 'gateway' else 100e6,
                coverage_radius_m=50000  # 50km ground coverage
            ))
            node_id += 1

        # 3. Add UAVs (interpolated from trajectories or generated)
        uav_positions = self._generate_uav_positions(num_uavs, region, time_offset_s)
        for i, pos in enumerate(uav_positions):
            x, y, z = self._latlon_to_ecef(pos['lat'], pos['lon'], pos['alt'])
            nodes.append(NetworkNode(
                id=node_id,
                name=f"UAV-{i+1:03d}",
                node_type='uav',
                lat=pos['lat'],
                lon=pos['lon'],
                altitude_m=pos['alt'],
                x=x, y=y, z=z,
                energy=np.random.uniform(0.3, 1.0),  # Variable battery
                bandwidth_hz=20e6,
                coverage_radius_m=10000  # 10km UAV coverage
            ))
            node_id += 1

        # 4. Establish links
        links = self._establish_links(nodes)

        # 5. Compute statistics
        stats = self._compute_statistics(nodes, links)

        return {
            'nodes': [n.to_dict() for n in nodes],
            'links': links,
            'statistics': stats,
            'region': region,
            'time_offset_s': time_offset_s
        }

    def _generate_uav_positions(
        self,
        num_uavs: int,
        region: Dict,
        time_offset_s: float
    ) -> List[Dict]:
        """Generate UAV positions based on trajectories or random placement."""
        positions = []

        # Use trajectory waypoints if available
        trajectories = self.uav_config.get('trajectories', [])

        for traj in trajectories:
            if len(positions) >= num_uavs:
                break

            # Check if trajectory is in region
            center_lat = traj.get('center_lat', 0)
            center_lon = traj.get('center_lon', 0)

            if not self._in_region(center_lat, center_lon, region):
                continue

            # Interpolate position along trajectory
            waypoints = traj.get('waypoints', [])
            if not waypoints:
                continue

            # Simple linear interpolation
            total_time = waypoints[-1]['time_s'] if waypoints else 1
            t = time_offset_s % total_time

            # Find segment
            for i in range(len(waypoints) - 1):
                if waypoints[i]['time_s'] <= t < waypoints[i+1]['time_s']:
                    dt = waypoints[i+1]['time_s'] - waypoints[i]['time_s']
                    alpha = (t - waypoints[i]['time_s']) / dt if dt > 0 else 0
                    positions.append({
                        'lat': waypoints[i]['lat'] + alpha * (waypoints[i+1]['lat'] - waypoints[i]['lat']),
                        'lon': waypoints[i]['lon'] + alpha * (waypoints[i+1]['lon'] - waypoints[i]['lon']),
                        'alt': waypoints[i]['alt'] + alpha * (waypoints[i+1]['alt'] - waypoints[i]['alt'])
                    })
                    break

        # Fill remaining with random positions in region
        while len(positions) < num_uavs:
            lat = np.random.uniform(*region['lat_range'])
            lon = np.random.uniform(*region['lon_range'])
            alt = np.random.uniform(100, 500)  # 100-500m altitude
            positions.append({'lat': lat, 'lon': lon, 'alt': alt})

        return positions[:num_uavs]

    def _establish_links(self, nodes: List[NetworkNode]) -> List[Dict]:
        """Establish links between nodes based on coverage and type."""
        links = []

        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:
                    continue

                distance = self._compute_distance(node1, node2)

                # Check if link is possible
                can_link, link_type = self._can_establish_link(node1, node2, distance)
                if not can_link:
                    continue

                # Get link parameters
                params = self.LINK_PARAMS.get(link_type, self.LINK_PARAMS[('ground', 'ground')])

                # Compute propagation delay
                prop_delay_ms = (distance / self.SPEED_OF_LIGHT) * 1000
                total_delay = params['base_delay_ms'] + prop_delay_ms

                links.append({
                    'source': node1.id,
                    'target': node2.id,
                    'distance_m': distance,
                    'delay_ms': total_delay,
                    'bandwidth_hz': params['bandwidth'],
                    'loss_rate': params['loss_rate'],
                    'link_type': f"{node1.node_type}-{node2.node_type}"
                })

        return links

    def _can_establish_link(
        self,
        node1: NetworkNode,
        node2: NetworkNode,
        distance: float
    ) -> Tuple[bool, Tuple[str, str]]:
        """Check if a link can be established between two nodes."""
        type1, type2 = sorted([node1.node_type, node2.node_type])
        link_type = (type1, type2)

        # Coverage rules based on real-world parameters
        if type1 == 'satellite' and type2 == 'satellite':
            # ISL: up to ~5000km for LEO (laser link range)
            return distance < 5000000, link_type

        elif type1 == 'ground' and type2 == 'satellite':
            # Satellite-Ground: Use elevation angle based coverage
            # For LEO at 550km, max slant range at 25° elevation is ~1200km
            # At 40° elevation: ~800km
            sat_node = node2 if node2.node_type == 'satellite' else node1
            sat_alt = sat_node.altitude_m
            # Max slant range = altitude / sin(min_elevation)
            # Using 25° minimum elevation for Starlink
            min_elevation_rad = np.radians(25)
            max_slant_range = sat_alt / np.sin(min_elevation_rad) * 1.2  # 20% margin
            return distance < max_slant_range, link_type

        elif type1 == 'satellite' and type2 == 'uav':
            # Satellite-UAV: Similar to ground but UAV can be higher
            sat_node = node1 if node1.node_type == 'satellite' else node2
            sat_alt = sat_node.altitude_m
            min_elevation_rad = np.radians(20)  # Lower elevation for UAV
            max_slant_range = sat_alt / np.sin(min_elevation_rad) * 1.2
            return distance < max_slant_range, link_type

        elif type1 == 'uav' and type2 == 'uav':
            # UAV-UAV: Line of sight, ~30km range
            return distance < 30000, link_type

        elif type1 == 'ground' and type2 == 'uav':
            # Ground-UAV: UAV coverage radius
            uav_node = node2 if node2.node_type == 'uav' else node1
            return distance < uav_node.coverage_radius_m * 2, link_type

        elif type1 == 'ground' and type2 == 'ground':
            # Ground-Ground: Fiber/terrestrial links, ~200km
            return distance < 200000, link_type

        return False, link_type

    def _compute_statistics(self, nodes: List[NetworkNode], links: List[Dict]) -> Dict:
        """Compute network statistics."""
        node_types = {'satellite': 0, 'uav': 0, 'ground': 0}
        for node in nodes:
            node_types[node.node_type] += 1

        link_types = {}
        delays = []
        for link in links:
            lt = link['link_type']
            link_types[lt] = link_types.get(lt, 0) + 1
            delays.append(link['delay_ms'])

        # Compute average degree
        degree_count = {}
        for link in links:
            degree_count[link['source']] = degree_count.get(link['source'], 0) + 1
            degree_count[link['target']] = degree_count.get(link['target'], 0) + 1

        avg_degree = np.mean(list(degree_count.values())) if degree_count else 0

        return {
            'total_nodes': len(nodes),
            'node_types': node_types,
            'total_links': len(links),
            'link_types': link_types,
            'avg_degree': avg_degree,
            'avg_delay_ms': np.mean(delays) if delays else 0,
            'max_delay_ms': max(delays) if delays else 0
        }

    def export_config(self, topology: Dict, output_path: str):
        """Export topology as YAML config for training."""
        import yaml

        config = {
            'network': {
                'num_satellites': topology['statistics']['node_types']['satellite'],
                'num_uavs': topology['statistics']['node_types']['uav'],
                'num_ground': topology['statistics']['node_types']['ground'],
                'topology_source': 'real_world',
                'region': topology['region']
            },
            'nodes': topology['nodes'],
            'links': topology['links']
        }

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


if __name__ == '__main__':
    builder = RealWorldTopologyBuilder('data')

    # Build China region topology
    topology = builder.build_topology(
        num_satellites=30,
        num_uavs=15,
        num_ground=20,
        region={'lat_range': (20, 50), 'lon_range': (100, 135)}
    )

    print("Network Topology Statistics:")
    stats = topology['statistics']
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  - Satellites: {stats['node_types']['satellite']}")
    print(f"  - UAVs: {stats['node_types']['uav']}")
    print(f"  - Ground: {stats['node_types']['ground']}")
    print(f"  Total links: {stats['total_links']}")
    print(f"  Average degree: {stats['avg_degree']:.2f}")
    print(f"  Average delay: {stats['avg_delay_ms']:.2f} ms")
