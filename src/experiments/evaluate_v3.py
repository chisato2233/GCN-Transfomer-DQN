"""
Evaluation script for V3 Per-Neighbor Q-Value Agent.

Compares V3 against baselines (Greedy, Random, Shortest Path).

Usage:
    python src/experiments/evaluate_v3.py --checkpoint logs/v3_xxx/checkpoints/best_model.pt --config configs/routing_config.yaml
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn_gcn_transformer_v3 import DQNGCNTransformerAgentV3


class BaselineRouter:
    """Baseline routing algorithms."""

    def __init__(self, env: SAGINRoutingEnv):
        self.env = env

    def get_greedy_action(self, current: int, destination: int,
                          neighbors: List[int], action_mask: np.ndarray,
                          visited_nodes: set = None) -> int:
        """Greedy: choose neighbor closest to destination (LOCAL info only)."""
        dest_pos = self.env.topology.nodes[destination].position

        if visited_nodes is None:
            visited_nodes = set()

        best_action = -1
        best_dist = float('inf')

        # First: find unvisited neighbor closest to destination
        for i, neighbor in enumerate(neighbors):
            if i < len(action_mask) and action_mask[i] > 0:
                if neighbor not in visited_nodes:
                    neighbor_pos = self.env.topology.nodes[neighbor].position
                    dist = np.linalg.norm(neighbor_pos - dest_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_action = i

        # Fallback: any valid neighbor
        if best_action == -1:
            for i, neighbor in enumerate(neighbors):
                if i < len(action_mask) and action_mask[i] > 0:
                    neighbor_pos = self.env.topology.nodes[neighbor].position
                    dist = np.linalg.norm(neighbor_pos - dest_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_action = i

        return best_action if best_action >= 0 else 0

    def get_random_action(self, action_mask: np.ndarray) -> int:
        """Random action from valid actions."""
        valid_indices = np.where(action_mask > 0)[0]
        if len(valid_indices) > 0:
            return int(np.random.choice(valid_indices))
        return 0

    def get_shortest_path_action(self, current: int, destination: int,
                                  neighbors: List[int], action_mask: np.ndarray) -> int:
        """Shortest path action (GLOBAL info - upper bound)."""
        path, _ = self.env.topology.shortest_path(current, destination)

        if len(path) < 2:
            valid_indices = np.where(action_mask > 0)[0]
            return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0

        next_node = path[1]
        for i, neighbor in enumerate(neighbors):
            if neighbor == next_node and i < len(action_mask) and action_mask[i] > 0:
                return i

        valid_indices = np.where(action_mask > 0)[0]
        return int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0


def evaluate_v3(agent, env, num_episodes: int = 200) -> Dict:
    """Evaluate V3 agent."""
    metrics = defaultdict(list)
    max_steps = env.max_hops * 2

    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"  V3 Agent: {ep}/{num_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            topo_data = env.get_topology_aware_features()
            simplified_hist = env.get_simplified_history()
            action_mask = obs['action_mask']

            action = agent.select_action(
                topo_data['neighbor_topology_features'],
                topo_data['neighbor_mask'],
                simplified_hist,
                action_mask,
                training=False
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        metrics['reward'].append(episode_reward)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['hops'].append(info.get('hop_count', 0))
        metrics['delay'].append(info.get('total_delay', 0))

        if info.get('success', False):
            optimal_hops = info.get('optimal_hops', -1)
            if optimal_hops > 0:
                metrics['efficiency'].append(optimal_hops / info.get('hop_count', optimal_hops))

    return {
        'avg_reward': float(np.mean(metrics['reward'])),
        'std_reward': float(np.std(metrics['reward'])),
        'success_rate': float(np.mean(metrics['success']) * 100),
        'avg_hops': float(np.mean(metrics['hops'])),
        'avg_delay': float(np.mean(metrics['delay'])),
        'path_efficiency': float(np.mean(metrics['efficiency'])) if metrics['efficiency'] else 0.0
    }


def evaluate_baseline(env, baseline_type: str, num_episodes: int = 200) -> Dict:
    """Evaluate baseline algorithm."""
    baseline = BaselineRouter(env)
    metrics = defaultdict(list)
    max_steps = env.max_hops * 2

    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"  {baseline_type}: {ep}/{num_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        visited_nodes = {env.current_node}

        while not done and steps < max_steps:
            action_mask = obs['action_mask']
            current_node = env.current_node
            destination = env.destination
            neighbors = env.topology.get_neighbors(current_node)

            if baseline_type == 'greedy':
                action = baseline.get_greedy_action(current_node, destination, neighbors, action_mask, visited_nodes)
            elif baseline_type == 'random':
                action = baseline.get_random_action(action_mask)
            elif baseline_type == 'shortest_path':
                action = baseline.get_shortest_path_action(current_node, destination, neighbors, action_mask)
            else:
                raise ValueError(f"Unknown baseline: {baseline_type}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            visited_nodes.add(env.current_node)

        metrics['reward'].append(episode_reward)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['hops'].append(info.get('hop_count', 0))
        metrics['delay'].append(info.get('total_delay', 0))

        if info.get('success', False):
            optimal_hops = info.get('optimal_hops', -1)
            if optimal_hops > 0:
                metrics['efficiency'].append(optimal_hops / info.get('hop_count', optimal_hops))

    return {
        'avg_reward': float(np.mean(metrics['reward'])),
        'std_reward': float(np.std(metrics['reward'])),
        'success_rate': float(np.mean(metrics['success']) * 100),
        'avg_hops': float(np.mean(metrics['hops'])),
        'avg_delay': float(np.mean(metrics['delay'])),
        'path_efficiency': float(np.mean(metrics['efficiency'])) if metrics['efficiency'] else 0.0
    }


def print_results(results: Dict[str, Dict]):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("PERFORMANCE COMPARISON (V3 Per-Neighbor Q-Value)")
    print("=" * 90)
    print(f"{'Method':<25} {'Info Type':<10} {'Success%':>10} {'Avg Hops':>10} {'Efficiency':>12} {'Avg Delay':>12}")
    print("-" * 90)

    info_types = {
        'V3 Per-Neighbor': 'Local',
        'Greedy': 'Local',
        'Random': 'Local',
        'Shortest Path': 'Global',
    }

    sorted_results = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)

    for method, m in sorted_results:
        eff_str = f"{m['path_efficiency']*100:.1f}%" if m['path_efficiency'] > 0 else "N/A"
        info_type = info_types.get(method, 'Local')
        print(f"{method:<25} {info_type:<10} {m['success_rate']:>9.1f}% {m['avg_hops']:>10.1f} {eff_str:>12} {m['avg_delay']:>11.2f}ms")

    print("=" * 90)

    # Analysis
    if 'V3 Per-Neighbor' in results and 'Greedy' in results:
        v3 = results['V3 Per-Neighbor']['success_rate']
        greedy = results['Greedy']['success_rate']
        diff = v3 - greedy
        print(f"\nV3 vs Greedy: {diff:+.1f}% ({'V3 wins' if diff > 0 else 'Greedy wins' if diff < 0 else 'Tie'})")


def main():
    parser = argparse.ArgumentParser(description='Evaluate V3 Agent')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to V3 model checkpoint')
    parser.add_argument('--config', type=str, default='configs/routing_config.yaml', help='Config file')
    parser.add_argument('--episodes', type=int, default=200, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("V3 Per-Neighbor Q-Value Evaluation")
    print("=" * 60)

    # Create environment
    env = SAGINRoutingEnv(config)
    print(f"Environment: {env.topology.total_nodes} nodes")

    # Load V3 agent
    agent_config = {
        'num_actions': config['environment']['max_neighbors'],
        'max_neighbors': config['gcn']['max_neighbors'],
        'history_length': config['transformer']['history_length'],
        # V3 Enhanced: 14 dims = 8 routing + 6 three-layer features
        'topology_feature_dim': 14,
        'simplified_history_dim': 6,
        'neighbor_hidden_dim': 64,
        'temporal_d_model': 32,
        'device': config['hardware']['device'],
        'use_transformer': True,
    }

    agent = DQNGCNTransformerAgentV3(agent_config)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0
    print(f"Loaded checkpoint: {args.checkpoint}")

    results = {}

    # Evaluate V3
    print(f"\n[1] Evaluating V3 Per-Neighbor Agent ({args.episodes} episodes)...")
    results['V3 Per-Neighbor'] = evaluate_v3(agent, env, args.episodes)
    print(f"    Success Rate: {results['V3 Per-Neighbor']['success_rate']:.1f}%")

    # Evaluate baselines
    print(f"\n[2] Evaluating Greedy Baseline...")
    results['Greedy'] = evaluate_baseline(env, 'greedy', args.episodes)
    print(f"    Success Rate: {results['Greedy']['success_rate']:.1f}%")

    print(f"\n[3] Evaluating Random Baseline...")
    results['Random'] = evaluate_baseline(env, 'random', args.episodes)
    print(f"    Success Rate: {results['Random']['success_rate']:.1f}%")

    print(f"\n[4] Evaluating Shortest Path (Global Upper Bound)...")
    results['Shortest Path'] = evaluate_baseline(env, 'shortest_path', args.episodes)
    print(f"    Success Rate: {results['Shortest Path']['success_rate']:.1f}%")

    # Print results
    print_results(results)

    # Save results
    output_file = 'evaluation_results_v3.yaml'
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
