"""
Visualization tools for GCN-Transformer routing model.

Includes:
- Training curves (loss, reward, success rate)
- Network topology visualization
- Routing path visualization
- Performance comparison charts
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
import json
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)


def load_training_logs(log_dir: str) -> Dict[str, List]:
    """Load training logs from TensorBoard events or log files."""
    logs = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'success_rates': [],
        'epsilons': [],
        'delays': [],
        'hops': [],
        'q_values': []
    }

    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Log directory not found: {log_dir}")
        return logs

    # Try to load from TensorBoard events first
    event_files = list(log_path.glob('events.out.tfevents.*'))
    if event_files:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(event_files[0]))
            ea.Reload()

            # Map TensorBoard tags to log keys
            tag_mapping = {
                'Train/Reward': 'rewards',
                'Train/Loss': 'losses',
                'Train/DeliveryRate': 'success_rates',
                'Train/Epsilon': 'epsilons',
                'Train/AvgDelay': 'delays',
                'Train/AvgHops': 'hops',
                'Train/QValue': 'q_values'
            }

            for tag, key in tag_mapping.items():
                if tag in ea.Tags().get('scalars', []):
                    events = ea.Scalars(tag)
                    logs[key] = [e.value for e in events]
                    if key == 'rewards' and not logs['episodes']:
                        logs['episodes'] = [e.step for e in events]

            if logs['rewards']:
                print(f"  Loaded {len(logs['rewards'])} data points from TensorBoard")
                return logs
        except ImportError:
            print("  TensorBoard not available, trying text logs...")
        except Exception as e:
            print(f"  Error loading TensorBoard: {e}")

    # Fallback to text log files
    log_files = list(log_path.glob('*.log')) + list(log_path.glob('training_*.json'))

    if not log_files:
        print(f"No log files found in {log_dir}")
        return logs

    # Parse log file
    for log_file in log_files:
        if log_file.suffix == '.json':
            with open(log_file, 'r') as f:
                data = json.load(f)
                for key in logs:
                    if key in data:
                        logs[key].extend(data[key])
        else:
            # Parse text log
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if 'Episode' in line and 'Loss' in line:
                        try:
                            parts = line.split('|')
                            for part in parts:
                                part = part.strip()
                                if part.startswith('Episode'):
                                    ep = int(part.split()[1])
                                    logs['episodes'].append(ep)
                                elif part.startswith('Reward:'):
                                    reward = float(part.split(':')[1])
                                    logs['rewards'].append(reward)
                                elif part.startswith('Loss:'):
                                    loss_str = part.split(':')[1].strip()
                                    if loss_str != 'nan':
                                        loss = float(loss_str)
                                        logs['losses'].append(loss)
                                elif part.startswith('Delivery:'):
                                    rate = float(part.split(':')[1].replace('%', ''))
                                    logs['success_rates'].append(rate)
                                elif part.startswith('Epsilon:'):
                                    eps = float(part.split(':')[1])
                                    logs['epsilons'].append(eps)
                                elif part.startswith('Delay:'):
                                    delay = float(part.split(':')[1].replace('ms', ''))
                                    logs['delays'].append(delay)
                                elif part.startswith('Hops:'):
                                    hops = float(part.split(':')[1])
                                    logs['hops'].append(hops)
                        except (ValueError, IndexError):
                            continue

    return logs


def plot_training_curves(logs: Dict[str, List], save_path: str = 'training_curves.png'):
    """Plot training curves: loss, reward, success rate, epsilon."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    episodes = logs.get('episodes', list(range(len(logs.get('rewards', [])))))

    # Plot 1: Reward
    ax = axes[0, 0]
    if logs['rewards']:
        ax.plot(episodes[:len(logs['rewards'])], logs['rewards'], 'b-', alpha=0.7, label='Episode Reward')
        # Moving average
        if len(logs['rewards']) > 50:
            ma = np.convolve(logs['rewards'], np.ones(50)/50, mode='valid')
            ax.plot(episodes[49:49+len(ma)], ma, 'r-', linewidth=2, label='Moving Avg (50)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Reward')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No reward data', ha='center', va='center', transform=ax.transAxes)

    # Plot 2: Loss
    ax = axes[0, 1]
    if logs['losses']:
        ax.plot(episodes[:len(logs['losses'])], logs['losses'], 'g-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
    else:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax.transAxes)

    # Plot 3: Success Rate
    ax = axes[0, 2]
    if logs['success_rates']:
        ax.plot(episodes[:len(logs['success_rates'])], logs['success_rates'], 'm-', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delivery Rate (%)')
        ax.set_title('Packet Delivery Rate')
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'No success rate data', ha='center', va='center', transform=ax.transAxes)

    # Plot 4: Epsilon
    ax = axes[1, 0]
    if logs['epsilons']:
        ax.plot(episodes[:len(logs['epsilons'])], logs['epsilons'], 'c-')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon)')
    else:
        ax.text(0.5, 0.5, 'No epsilon data', ha='center', va='center', transform=ax.transAxes)

    # Plot 5: Average Delay
    ax = axes[1, 1]
    if logs['delays']:
        ax.plot(episodes[:len(logs['delays'])], logs['delays'], 'orange', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delay (ms)')
        ax.set_title('Average Transmission Delay')
    else:
        ax.text(0.5, 0.5, 'No delay data', ha='center', va='center', transform=ax.transAxes)

    # Plot 6: Average Hops
    ax = axes[1, 2]
    if logs['hops']:
        ax.plot(episodes[:len(logs['hops'])], logs['hops'], 'purple', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Hops')
        ax.set_title('Average Path Length (Hops)')
    else:
        ax.text(0.5, 0.5, 'No hops data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_network_topology(env, save_path: str = 'network_topology.png'):
    """Visualize the SAGIN network topology."""
    fig, ax = plt.subplots(figsize=(12, 10))

    topology = env.topology
    num_nodes = topology.total_nodes

    # Get positions
    positions = np.array([topology.nodes[i].position for i in range(num_nodes)])

    # Normalize positions for display
    positions_2d = positions[:, :2]  # Use only x, y

    # Node colors by type
    colors = []
    sizes = []
    num_satellites = topology.num_satellites
    num_uavs = topology.num_uavs

    for i in range(num_nodes):
        if i < num_satellites:
            colors.append('#FF6B6B')  # Red for satellites
            sizes.append(300)
        elif i < num_satellites + num_uavs:
            colors.append('#4ECDC4')  # Teal for UAVs
            sizes.append(200)
        else:
            colors.append('#45B7D1')  # Blue for ground
            sizes.append(150)

    # Draw edges
    adj_matrix = topology.get_adjacency_matrix()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:
                # Determine edge type by node types
                if i < num_satellites or j < num_satellites:
                    edge_color = '#FFB6C1'
                    alpha = 0.5
                elif i < num_satellites + num_uavs or j < num_satellites + num_uavs:
                    edge_color = '#98D8C8'
                    alpha = 0.6
                else:
                    edge_color = '#B0C4DE'
                    alpha = 0.7

                ax.plot([positions_2d[i, 0], positions_2d[j, 0]],
                       [positions_2d[i, 1], positions_2d[j, 1]],
                       color=edge_color, alpha=alpha, linewidth=1, zorder=1)

    # Draw nodes
    ax.scatter(positions_2d[:, 0], positions_2d[:, 1],
              c=colors, s=sizes, zorder=2, edgecolors='white', linewidths=1.5)

    # Add node labels
    for i in range(num_nodes):
        ax.annotate(f'{i}', (positions_2d[i, 0], positions_2d[i, 1]),
                   fontsize=8, ha='center', va='center', color='white', fontweight='bold')

    # Legend
    num_ground = num_nodes - num_satellites - num_uavs
    legend_elements = [
        mpatches.Patch(color='#FF6B6B', label=f'Satellites ({num_satellites})'),
        mpatches.Patch(color='#4ECDC4', label=f'UAVs ({num_uavs})'),
        mpatches.Patch(color='#45B7D1', label=f'Ground ({num_ground})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title('SAGIN Network Topology\n(Space-Air-Ground Integrated Network)', fontsize=14)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Network topology saved to {save_path}")
    plt.close()


def plot_routing_path(env, path: List[int], source: int, destination: int,
                      save_path: str = 'routing_path.png', title: str = 'Routing Path'):
    """Visualize a routing path on the network."""
    fig, ax = plt.subplots(figsize=(12, 10))

    topology = env.topology
    num_nodes = topology.total_nodes
    num_satellites = topology.num_satellites
    num_uavs = topology.num_uavs

    # Get positions
    positions = np.array([topology.nodes[i].position for i in range(num_nodes)])
    positions_2d = positions[:, :2]

    adj_matrix = topology.get_adjacency_matrix()

    # Draw all edges (faded)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i, j] > 0:
                ax.plot([positions_2d[i, 0], positions_2d[j, 0]],
                       [positions_2d[i, 1], positions_2d[j, 1]],
                       color='#E0E0E0', alpha=0.3, linewidth=1, zorder=1)

    # Draw routing path (highlighted)
    for i in range(len(path) - 1):
        node_a, node_b = path[i], path[i + 1]
        ax.plot([positions_2d[node_a, 0], positions_2d[node_b, 0]],
               [positions_2d[node_a, 1], positions_2d[node_b, 1]],
               color='#FF4444', linewidth=3, zorder=3)

    # Draw nodes
    colors = []
    sizes = []
    for i in range(num_nodes):
        if i == source:
            colors.append('#00FF00')  # Green for source
            sizes.append(400)
        elif i == destination:
            colors.append('#FF0000')  # Red for destination
            sizes.append(400)
        elif i in path:
            colors.append('#FFA500')  # Orange for path nodes
            sizes.append(250)
        elif i < num_satellites:
            colors.append('#FFB6C1')
            sizes.append(200)
        elif i < num_satellites + num_uavs:
            colors.append('#98D8C8')
            sizes.append(150)
        else:
            colors.append('#B0C4DE')
            sizes.append(100)

    ax.scatter(positions_2d[:, 0], positions_2d[:, 1],
              c=colors, s=sizes, zorder=2, edgecolors='black', linewidths=1)

    # Add node labels
    for i in range(num_nodes):
        ax.annotate(f'{i}', (positions_2d[i, 0], positions_2d[i, 1]),
                   fontsize=8, ha='center', va='center', fontweight='bold')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#00FF00', markersize=15, label=f'Source ({source})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF0000', markersize=15, label=f'Destination ({destination})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=12, label='Path Node'),
        Line2D([0], [0], color='#FF4444', linewidth=3, label='Routing Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(f'{title}\nPath: {" -> ".join(map(str, path))} ({len(path)-1} hops)', fontsize=14)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Routing path saved to {save_path}")
    plt.close()


def plot_comparison_bar(results: Dict[str, Dict[str, float]], save_path: str = 'comparison.png'):
    """Plot bar chart comparing different methods."""
    methods = list(results.keys())
    metrics = ['success_rate', 'avg_reward', 'avg_hops', 'avg_delay']
    metric_names = ['Success Rate (%)', 'Avg Reward', 'Avg Hops', 'Avg Delay (ms)']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        values = [results[m].get(metric, 0) for m in methods]

        bars = ax.bar(methods, values, color=colors[:len(methods)], edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        ax.set_ylabel(name)
        ax.set_title(name)

        # Rotate x labels if needed
        if len(methods) > 3:
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=15, ha='right')

    plt.suptitle('Performance Comparison: GCN-Transformer vs Baselines', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison chart saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory containing training logs')
    parser.add_argument('--config', type=str, default='configs/routing_config.yaml',
                        help='Path to config file')
    parser.add_argument('--results', type=str, default='evaluation_results.yaml',
                        help='Path to evaluation results file')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Output directory for plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("GCN-Transformer Routing Visualization")
    print("=" * 60)

    # 1. Plot training curves
    print("\n1. Loading training logs...")

    # Find the latest log directory with TensorBoard events
    log_path = Path(args.log_dir)
    logs = {}
    if log_path.exists():
        # Search in subdirectories for TensorBoard events
        subdirs = sorted([d for d in log_path.iterdir() if d.is_dir()],
                        key=lambda x: x.stat().st_mtime, reverse=True)
        for subdir in subdirs:
            event_files = list(subdir.glob('events.out.tfevents.*'))
            if event_files:
                print(f"   Found TensorBoard logs in: {subdir.name}")
                logs = load_training_logs(str(subdir))
                if logs.get('rewards'):
                    break

    if not logs:
        logs = load_training_logs(args.log_dir)

    if any(len(v) > 0 for v in logs.values()):
        plot_training_curves(logs, os.path.join(args.output_dir, 'training_curves.png'))
    else:
        print("   No training data found, creating sample visualization...")
        # Create sample data for demonstration
        sample_logs = {
            'episodes': list(range(0, 1000, 10)),
            'rewards': [-150 + i*0.05 + np.random.randn()*10 for i in range(100)],
            'losses': [5 - i*0.02 + np.random.randn()*0.5 for i in range(100)],
            'success_rates': [30 + i*0.3 + np.random.randn()*5 for i in range(100)],
            'epsilons': [1.0 * (0.995 ** i) for i in range(100)],
            'delays': [5 - i*0.02 + np.random.randn()*0.3 for i in range(100)],
            'hops': [15 - i*0.03 + np.random.randn()*1 for i in range(100)]
        }
        plot_training_curves(sample_logs, os.path.join(args.output_dir, 'training_curves_sample.png'))

    # 2. Plot network topology
    print("\n2. Creating network topology visualization...")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        from src.env.sagin_env import SAGINRoutingEnv
        env = SAGINRoutingEnv(config)
        plot_network_topology(env, os.path.join(args.output_dir, 'network_topology.png'))

        # 3. Plot sample routing path
        print("\n3. Creating sample routing path visualization...")
        import random
        num_satellites = env.topology.num_satellites
        num_uavs = env.topology.num_uavs
        num_nodes = env.topology.total_nodes

        source = random.randint(num_satellites + num_uavs, num_nodes - 1)
        destination = random.randint(num_satellites + num_uavs, num_nodes - 1)
        while destination == source:
            destination = random.randint(num_satellites + num_uavs, num_nodes - 1)

        # Get shortest path as sample
        sample_path, _ = env.topology.shortest_path(source, destination)
        if sample_path:
            plot_routing_path(env, sample_path, source, destination,
                             os.path.join(args.output_dir, 'routing_path_sample.png'),
                             'Sample Routing Path (Shortest Path)')

    except Exception as e:
        print(f"   Error creating topology visualization: {e}")

    # 4. Plot comparison chart if results exist
    print("\n4. Creating comparison chart...")
    if os.path.exists(args.results):
        with open(args.results, 'r') as f:
            results = yaml.safe_load(f)
        if results:
            plot_comparison_bar(results, os.path.join(args.output_dir, 'comparison.png'))
    else:
        print(f"   Results file not found: {args.results}")
        print("   Run evaluate.py first to generate results.")
        # Create sample comparison
        sample_results = {
            'GCN-Transformer-DQN': {'success_rate': 65, 'avg_reward': -100, 'avg_hops': 12, 'avg_delay': 3.5},
            'Shortest Path': {'success_rate': 80, 'avg_reward': -80, 'avg_hops': 8, 'avg_delay': 2.5},
            'Greedy': {'success_rate': 55, 'avg_reward': -120, 'avg_hops': 15, 'avg_delay': 4.0},
            'Random': {'success_rate': 20, 'avg_reward': -180, 'avg_hops': 30, 'avg_delay': 8.0}
        }
        plot_comparison_bar(sample_results, os.path.join(args.output_dir, 'comparison_sample.png'))

    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
