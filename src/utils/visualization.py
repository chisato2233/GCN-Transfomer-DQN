"""
Visualization tools for SAGIN Intelligent Routing.

Provides:
1. Training curve plots (reward, loss, delivery rate)
2. Network topology visualization
3. Routing path visualization
4. Comparison plots for ablation studies
5. Performance distribution plots (CDF, histogram)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ============================================================================
# Training Curves
# ============================================================================

def plot_training_curves(
    episode_rewards: List[float],
    eval_rewards: Optional[List[float]] = None,
    losses: Optional[List[float]] = None,
    delivery_rates: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Progress",
    window_size: int = 50
):
    """
    Plot training curves with smoothing.

    Args:
        episode_rewards: Episode rewards
        eval_rewards: Evaluation rewards (optional)
        losses: Training losses (optional)
        delivery_rates: Delivery rates (optional)
        save_path: Path to save figure
        title: Plot title
        window_size: Smoothing window size
    """
    num_plots = 1
    if losses is not None:
        num_plots += 1
    if delivery_rates is not None:
        num_plots += 1

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    if num_plots == 1:
        axes = [axes]

    # Helper for smoothing
    def smooth(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # Plot 1: Rewards
    ax = axes[0]
    episodes = np.arange(len(episode_rewards))
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    smoothed = smooth(episode_rewards, window_size)
    ax.plot(np.arange(len(smoothed)) + window_size//2, smoothed,
            color='blue', linewidth=2, label=f'Smoothed (window={window_size})')

    if eval_rewards is not None:
        eval_episodes = np.linspace(0, len(episode_rewards)-1, len(eval_rewards))
        ax.scatter(eval_episodes, eval_rewards, color='red', s=50,
                  marker='*', label='Eval Reward', zorder=5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'{title} - Reward')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plot_idx = 1

    # Plot 2: Loss
    if losses is not None:
        ax = axes[plot_idx]
        ax.plot(losses, alpha=0.3, color='orange')
        smoothed_loss = smooth(losses, window_size)
        ax.plot(np.arange(len(smoothed_loss)) + window_size//2, smoothed_loss,
                color='orange', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{title} - Training Loss')
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot 3: Delivery Rate
    if delivery_rates is not None:
        ax = axes[plot_idx]
        eval_episodes = np.linspace(0, len(episode_rewards)-1, len(delivery_rates))
        ax.plot(eval_episodes, np.array(delivery_rates) * 100, color='green',
                marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delivery Rate (%)')
        ax.set_title(f'{title} - Packet Delivery Rate')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")

    return fig


def plot_comparison(
    results: Dict[str, Dict],
    metric: str = 'eval_rewards',
    save_path: Optional[str] = None,
    title: str = "Algorithm Comparison"
):
    """
    Plot comparison of multiple algorithms/configurations.

    Args:
        results: Dictionary of results from ablation study
        metric: Metric to compare
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning curves
    ax = axes[0]
    for config_name, result in results.items():
        if metric in result:
            data = result[metric]
            x = np.linspace(0, 100, len(data))  # Normalize to percentage
            ax.plot(x, data, label=result.get('name', config_name), linewidth=2)

    ax.set_xlabel('Training Progress (%)')
    ax.set_ylabel('Evaluation Reward')
    ax.set_title(f'{title} - Learning Curves')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Final performance bar chart
    ax = axes[1]
    names = []
    best_rewards = []
    delivery_rates = []

    for config_name, result in results.items():
        names.append(result.get('name', config_name))
        best_rewards.append(result.get('best_reward', 0))
        delivery_rates.append(result.get('final_delivery_rate', 0) * 100)

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, best_rewards, width, label='Best Reward', color='steelblue')
    ax.set_ylabel('Best Reward', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, delivery_rates, width, label='Delivery Rate %', color='coral')
    ax2.set_ylabel('Delivery Rate (%)', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.set_ylim([0, 100])

    ax.set_xticks(x)
    ax.set_xticklabels([n[:15] + '...' if len(n) > 15 else n for n in names],
                       rotation=45, ha='right')
    ax.set_title(f'{title} - Final Performance')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")

    return fig


# ============================================================================
# Network Topology Visualization
# ============================================================================

def plot_network_topology(
    node_positions: np.ndarray,
    node_types: List[int],
    edges: List[Tuple[int, int]],
    save_path: Optional[str] = None,
    title: str = "SAGIN Network Topology",
    highlight_path: Optional[List[int]] = None,
    node_labels: Optional[List[str]] = None
):
    """
    Plot 2D network topology.

    Args:
        node_positions: Node positions [N, 2] or [N, 3]
        node_types: Node type for each node (0=satellite, 1=UAV, 2=ground)
        edges: List of (src, dst) edges
        save_path: Path to save figure
        title: Plot title
        highlight_path: Path to highlight (list of node indices)
        node_labels: Optional labels for nodes
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use only x, y for 2D plot
    if node_positions.shape[1] == 3:
        positions_2d = node_positions[:, :2]
    else:
        positions_2d = node_positions

    # Color mapping for node types
    type_colors = {0: 'gold', 1: 'skyblue', 2: 'lightgreen'}
    type_names = {0: 'Satellite', 1: 'UAV', 2: 'Ground'}
    type_markers = {0: 's', 1: '^', 2: 'o'}
    type_sizes = {0: 300, 1: 200, 2: 150}

    # Plot edges
    for src, dst in edges:
        ax.plot([positions_2d[src, 0], positions_2d[dst, 0]],
                [positions_2d[src, 1], positions_2d[dst, 1]],
                'gray', alpha=0.3, linewidth=1, zorder=1)

    # Highlight path if provided
    if highlight_path is not None and len(highlight_path) > 1:
        path_x = [positions_2d[n, 0] for n in highlight_path]
        path_y = [positions_2d[n, 1] for n in highlight_path]
        ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8,
                label='Routing Path', zorder=2)

        # Arrow for direction
        for i in range(len(highlight_path) - 1):
            dx = positions_2d[highlight_path[i+1], 0] - positions_2d[highlight_path[i], 0]
            dy = positions_2d[highlight_path[i+1], 1] - positions_2d[highlight_path[i], 1]
            ax.annotate('', xy=(positions_2d[highlight_path[i+1], 0],
                               positions_2d[highlight_path[i+1], 1]),
                       xytext=(positions_2d[highlight_path[i], 0],
                              positions_2d[highlight_path[i], 1]),
                       arrowprops=dict(arrowstyle='->', color='red',
                                      lw=2, mutation_scale=15),
                       zorder=3)

    # Plot nodes by type
    legend_handles = []
    for node_type in sorted(set(node_types)):
        mask = np.array(node_types) == node_type
        indices = np.where(mask)[0]

        scatter = ax.scatter(
            positions_2d[mask, 0],
            positions_2d[mask, 1],
            c=type_colors[node_type],
            marker=type_markers[node_type],
            s=type_sizes[node_type],
            edgecolors='black',
            linewidths=1,
            label=type_names[node_type],
            zorder=4
        )
        legend_handles.append(scatter)

        # Add labels
        if node_labels is None:
            for idx in indices:
                ax.annotate(str(idx),
                           (positions_2d[idx, 0], positions_2d[idx, 1]),
                           textcoords="offset points",
                           xytext=(0, 8),
                           ha='center',
                           fontsize=8)

    # Highlight source and destination in path
    if highlight_path is not None and len(highlight_path) >= 2:
        # Source
        ax.scatter([positions_2d[highlight_path[0], 0]],
                  [positions_2d[highlight_path[0], 1]],
                  c='lime', s=400, marker='*', edgecolors='black',
                  linewidths=2, zorder=5, label='Source')
        # Destination
        ax.scatter([positions_2d[highlight_path[-1], 0]],
                  [positions_2d[highlight_path[-1], 1]],
                  c='red', s=400, marker='*', edgecolors='black',
                  linewidths=2, zorder=5, label='Destination')

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved topology to: {save_path}")

    return fig


def plot_3d_topology(
    node_positions: np.ndarray,
    node_types: List[int],
    edges: List[Tuple[int, int]],
    save_path: Optional[str] = None,
    title: str = "SAGIN 3D Network Topology",
    highlight_path: Optional[List[int]] = None
):
    """
    Plot 3D network topology showing satellite altitudes.

    Args:
        node_positions: Node positions [N, 3] (x, y, z)
        node_types: Node type for each node
        edges: List of edges
        save_path: Path to save figure
        title: Plot title
        highlight_path: Path to highlight
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    type_colors = {0: 'gold', 1: 'skyblue', 2: 'lightgreen'}
    type_names = {0: 'Satellite', 1: 'UAV', 2: 'Ground'}
    type_sizes = {0: 100, 1: 60, 2: 40}

    # Plot edges
    for src, dst in edges:
        ax.plot3D([node_positions[src, 0], node_positions[dst, 0]],
                  [node_positions[src, 1], node_positions[dst, 1]],
                  [node_positions[src, 2], node_positions[dst, 2]],
                  'gray', alpha=0.2, linewidth=0.5)

    # Highlight path
    if highlight_path is not None and len(highlight_path) > 1:
        for i in range(len(highlight_path) - 1):
            ax.plot3D([node_positions[highlight_path[i], 0],
                      node_positions[highlight_path[i+1], 0]],
                     [node_positions[highlight_path[i], 1],
                      node_positions[highlight_path[i+1], 1]],
                     [node_positions[highlight_path[i], 2],
                      node_positions[highlight_path[i+1], 2]],
                     'r-', linewidth=3, alpha=0.8)

    # Plot nodes
    for node_type in sorted(set(node_types)):
        mask = np.array(node_types) == node_type
        ax.scatter(node_positions[mask, 0],
                  node_positions[mask, 1],
                  node_positions[mask, 2],
                  c=type_colors[node_type],
                  s=type_sizes[node_type],
                  label=type_names[node_type],
                  edgecolors='black',
                  linewidths=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================================
# Performance Distribution Plots
# ============================================================================

def plot_delay_distribution(
    delays: List[float],
    save_path: Optional[str] = None,
    title: str = "End-to-End Delay Distribution",
    compare_with: Optional[Dict[str, List[float]]] = None
):
    """
    Plot delay distribution as CDF and histogram.

    Args:
        delays: List of delay values
        save_path: Path to save figure
        title: Plot title
        compare_with: Optional dict of other delay lists to compare
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CDF
    ax = axes[0]
    sorted_delays = np.sort(delays)
    cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
    ax.plot(sorted_delays, cdf, linewidth=2, label='Our Method')

    if compare_with:
        for name, other_delays in compare_with.items():
            sorted_other = np.sort(other_delays)
            cdf_other = np.arange(1, len(sorted_other) + 1) / len(sorted_other)
            ax.plot(sorted_other, cdf_other, linewidth=2, label=name, linestyle='--')

    ax.set_xlabel('End-to-End Delay (ms)')
    ax.set_ylabel('CDF')
    ax.set_title(f'{title} - CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(delays, bins=30, density=True, alpha=0.7, label='Our Method',
            edgecolor='black')

    if compare_with:
        for name, other_delays in compare_with.items():
            ax.hist(other_delays, bins=30, density=True, alpha=0.5,
                   label=name, edgecolor='black')

    ax.set_xlabel('End-to-End Delay (ms)')
    ax.set_ylabel('Density')
    ax.set_title(f'{title} - Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics_summary(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Performance Metrics Summary"
):
    """
    Plot radar chart of performance metrics.

    Args:
        metrics: Dictionary of metric name -> value (normalized 0-1)
        save_path: Path to save figure
        title: Plot title
    """
    categories = list(metrics.keys())
    values = list(metrics.values())

    # Number of variables
    N = len(categories)

    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the plot

    values += values[:1]  # Close the plot

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, size=14, y=1.08)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================================
# Utility Functions
# ============================================================================

def save_all_figures(output_dir: str, results: Dict):
    """Save all visualization figures for a training run."""
    os.makedirs(output_dir, exist_ok=True)

    # Training curves
    if 'episode_rewards' in results:
        plot_training_curves(
            results['episode_rewards'],
            results.get('eval_rewards'),
            results.get('losses'),
            results.get('eval_delivery_rates'),
            save_path=os.path.join(output_dir, 'training_curves.png')
        )

    # Close all figures to free memory
    plt.close('all')


def load_and_plot_ablation_results(results_path: str, output_dir: str):
    """Load ablation results from JSON and create comparison plots."""
    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    plot_comparison(
        results,
        metric='eval_rewards',
        save_path=os.path.join(output_dir, 'ablation_comparison.png'),
        title='Ablation Study'
    )

    plt.close('all')


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='SAGIN Visualization Tools')
    parser.add_argument('--results', '-r', type=str,
                       help='Path to results JSON file')
    parser.add_argument('--output', '-o', type=str, default='plots',
                       help='Output directory')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with synthetic data')

    args = parser.parse_args()

    if args.demo:
        # Create demo plots with synthetic data
        print("Creating demo visualizations...")

        # Synthetic training data
        np.random.seed(42)
        n_episodes = 1000
        episode_rewards = np.cumsum(np.random.randn(n_episodes) * 0.5) - 100
        episode_rewards = episode_rewards + np.linspace(0, 50, n_episodes)

        eval_rewards = episode_rewards[::100] + np.random.randn(10) * 5
        delivery_rates = np.clip(0.3 + np.linspace(0, 0.5, 10) + np.random.randn(10) * 0.05, 0, 1)

        os.makedirs(args.output, exist_ok=True)

        plot_training_curves(
            episode_rewards.tolist(),
            eval_rewards.tolist(),
            delivery_rates=delivery_rates.tolist(),
            save_path=os.path.join(args.output, 'demo_training.png'),
            title='Demo Training Progress'
        )

        # Synthetic network topology
        n_nodes = 19
        positions = np.random.rand(n_nodes, 2) * 10000
        positions[:3, :] *= 1  # Satellites spread out
        node_types = [0]*3 + [1]*6 + [2]*10

        edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)
                if np.random.rand() < 0.3]

        plot_network_topology(
            positions,
            node_types,
            edges,
            save_path=os.path.join(args.output, 'demo_topology.png'),
            title='Demo Network Topology',
            highlight_path=[15, 6, 1, 4, 12]
        )

        print(f"Demo plots saved to: {args.output}")

    elif args.results:
        load_and_plot_ablation_results(args.results, args.output)
