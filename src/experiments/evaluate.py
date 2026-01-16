"""
Evaluation script for trained GCN-Transformer routing model.

Compares the trained model against baseline algorithms:
- Shortest Path (Dijkstra)
- Random Routing
- Greedy (closest to destination)

[FIXED VERSION] Key changes:
- Uses env.get_state_history() for consistency with training
- Better metrics reporting
- Added path efficiency analysis
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import heapq

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn_gcn_transformer import DQNGCNTransformerAgent


def get_graph_features(env: SAGINRoutingEnv) -> Dict[str, np.ndarray]:
    """Extract graph features from environment for GCN input."""
    local_graph = env.get_local_graph_data()
    return {
        'current_node_features': local_graph['current_node_features'],
        'neighbor_features': local_graph['neighbor_features'],
        'neighbor_mask': local_graph['neighbor_mask']
    }


class BaselineRouter:
    """Baseline routing algorithms for comparison."""

    def __init__(self, env: SAGINRoutingEnv):
        self.env = env

    def dijkstra_path(self, source: int, destination: int) -> List[int]:
        """Find shortest path using Dijkstra's algorithm."""
        path, _ = self.env.topology.shortest_path(source, destination)
        return path if path else []

    def get_shortest_path_action(self, current: int, destination: int,
                                  neighbors: List[int], action_mask: np.ndarray) -> int:
        """Get action that follows shortest path."""
        path = self.dijkstra_path(current, destination)

        if len(path) < 2:
            # No path or already at destination, return random valid action
            valid_indices = np.where(action_mask > 0)[0]
            if len(valid_indices) > 0:
                return int(np.random.choice(valid_indices))
            return 0

        next_node = path[1]

        # Find action that leads to next_node
        for i, neighbor in enumerate(neighbors):
            if neighbor == next_node and i < len(action_mask) and action_mask[i] > 0:
                return i

        # Fallback to random valid action
        valid_indices = np.where(action_mask > 0)[0]
        if len(valid_indices) > 0:
            return int(np.random.choice(valid_indices))
        return 0

    def get_random_action(self, action_mask: np.ndarray) -> int:
        """Random action from valid actions."""
        valid_indices = np.where(action_mask > 0)[0]
        if len(valid_indices) > 0:
            return int(np.random.choice(valid_indices))
        return 0

    def get_greedy_action(self, current: int, destination: int,
                          neighbors: List[int], action_mask: np.ndarray) -> int:
        """Greedy action: choose neighbor closest to destination."""
        dest_pos = self.env.topology.nodes[destination].position

        best_action = 0
        best_dist = float('inf')

        for i, neighbor in enumerate(neighbors):
            if i < len(action_mask) and action_mask[i] > 0:
                neighbor_pos = self.env.topology.nodes[neighbor].position
                dist = np.linalg.norm(neighbor_pos - dest_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_action = i

        return best_action


def evaluate_model(agent: DQNGCNTransformerAgent,
                   env: SAGINRoutingEnv,
                   num_episodes: int = 100,
                   verbose: bool = True) -> Dict[str, float]:
    """
    Evaluate trained model performance.
    
    [FIXED] Uses env.get_state_history() for consistency with training.
    """
    metrics = defaultdict(list)
    path_efficiency_list = []

    for ep in range(num_episodes):
        if verbose and ep % 20 == 0:
            print(f"    Episode {ep}/{num_episodes}", end="\r", flush=True)
            
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        # [FIX] Use env.get_state_history() instead of manual initialization
        state = obs['state']
        state_history = env.get_state_history()

        while not done:
            # Get local graph data
            graph_features = get_graph_features(env)
            action_mask = obs['action_mask']

            # Select action (no exploration)
            action = agent.select_action(
                state, state_history,
                graph_features['current_node_features'],
                graph_features['neighbor_features'],
                graph_features['neighbor_mask'],
                action_mask, training=False
            )

            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            # [FIX] Update state and get history from env
            state = next_obs['state']
            state_history = env.get_state_history()
            obs = next_obs

        # Record metrics
        metrics['reward'].append(episode_reward)
        metrics['steps'].append(steps)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['delay'].append(info.get('total_delay', 0))
        metrics['hops'].append(info.get('hop_count', 0))
        
        # Calculate path efficiency
        optimal_hops = info.get('optimal_hops', -1)
        if info.get('success', False) and optimal_hops > 0:
            efficiency = optimal_hops / info.get('hop_count', optimal_hops)
            path_efficiency_list.append(efficiency)

    if verbose:
        print()  # Clear progress line

    return {
        'avg_reward': float(np.mean(metrics['reward'])),
        'std_reward': float(np.std(metrics['reward'])),
        'avg_steps': float(np.mean(metrics['steps'])),
        'success_rate': float(np.mean(metrics['success']) * 100),
        'avg_delay': float(np.mean(metrics['delay'])),
        'avg_hops': float(np.mean(metrics['hops'])),
        'path_efficiency': float(np.mean(path_efficiency_list)) if path_efficiency_list else 0.0
    }


def evaluate_baseline(env: SAGINRoutingEnv,
                      baseline_type: str,
                      num_episodes: int = 100,
                      verbose: bool = True) -> Dict[str, float]:
    """Evaluate baseline algorithm performance."""
    baseline = BaselineRouter(env)
    metrics = defaultdict(list)
    path_efficiency_list = []

    for ep in range(num_episodes):
        if verbose and ep % 20 == 0:
            print(f"    Episode {ep}/{num_episodes}", end="\r", flush=True)
            
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action_mask = obs['action_mask']
            current_node = env.current_node
            destination = env.destination
            neighbors = env.topology.get_neighbors(current_node)

            # Get action based on baseline type
            if baseline_type == 'shortest_path':
                action = baseline.get_shortest_path_action(
                    current_node, destination, neighbors, action_mask
                )
            elif baseline_type == 'random':
                action = baseline.get_random_action(action_mask)
            elif baseline_type == 'greedy':
                action = baseline.get_greedy_action(
                    current_node, destination, neighbors, action_mask
                )
            else:
                raise ValueError(f"Unknown baseline: {baseline_type}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            obs = next_obs

        metrics['reward'].append(episode_reward)
        metrics['steps'].append(steps)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['delay'].append(info.get('total_delay', 0))
        metrics['hops'].append(info.get('hop_count', 0))
        
        # Calculate path efficiency
        optimal_hops = info.get('optimal_hops', -1)
        if info.get('success', False) and optimal_hops > 0:
            efficiency = optimal_hops / info.get('hop_count', optimal_hops)
            path_efficiency_list.append(efficiency)

    if verbose:
        print()  # Clear progress line

    return {
        'avg_reward': float(np.mean(metrics['reward'])),
        'std_reward': float(np.std(metrics['reward'])),
        'avg_steps': float(np.mean(metrics['steps'])),
        'success_rate': float(np.mean(metrics['success']) * 100),
        'avg_delay': float(np.mean(metrics['delay'])),
        'avg_hops': float(np.mean(metrics['hops'])),
        'path_efficiency': float(np.mean(path_efficiency_list)) if path_efficiency_list else 0.0
    }


def print_comparison_table(results: Dict[str, Dict[str, float]]):
    """Print comparison table of all methods."""
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON")
    print("=" * 100)

    # Header
    print(f"{'Method':<25} {'Reward':>14} {'Success%':>10} {'Avg Hops':>10} {'Avg Delay':>12} {'Efficiency':>12}")
    print("-" * 100)

    # Data rows
    for method, metrics in results.items():
        reward_str = f"{metrics['avg_reward']:.1f}±{metrics['std_reward']:.1f}"
        efficiency_str = f"{metrics['path_efficiency']*100:.1f}%" if metrics['path_efficiency'] > 0 else "N/A"
        print(f"{method:<25} {reward_str:>14} {metrics['success_rate']:>9.1f}% "
              f"{metrics['avg_hops']:>10.1f} {metrics['avg_delay']:>11.2f}ms {efficiency_str:>12}")

    print("=" * 100)

    # Highlight best
    best_method = max(results.items(), key=lambda x: x[1]['success_rate'])
    print(f"\nBest method by success rate: {best_method[0]} ({best_method[1]['success_rate']:.1f}%)")
    
    # Compare model to baselines
    if 'GCN-Transformer-DQN' in results and 'Shortest Path' in results:
        model_success = results['GCN-Transformer-DQN']['success_rate']
        sp_success = results['Shortest Path']['success_rate']
        diff = model_success - sp_success
        print(f"Model vs Shortest Path: {diff:+.1f}% difference")


def main():
    parser = argparse.ArgumentParser(description='Evaluate routing model [FIXED]')
    parser.add_argument('--config', type=str, default='configs/routing_config.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("GCN-Transformer Routing Model Evaluation [FIXED]")
    print("=" * 60)

    # Create environment
    env = SAGINRoutingEnv(config)
    print(f"Environment created: {env.topology.total_nodes} nodes")

    # Build agent config
    agent_config = {
        'state_dim': env.state_dim,
        'num_actions': config['environment']['max_neighbors'],
        'node_feature_dim': config['gcn']['node_feature_dim'],
        'max_neighbors': config['gcn']['max_neighbors'],
        'history_length': config['transformer']['history_length'],
        'device': config['hardware']['device'],
        'lr': config['dqn']['lr'],
        'gamma': config['training']['gamma'],
        'use_double_dqn': config['dqn']['use_double_dqn'],
        'use_dueling': config['dqn']['use_dueling'],
        'use_per': False,
        'buffer_capacity': 1000,
        'batch_size': 64,
        'epsilon_start': 0.0,  # No exploration during eval
        'epsilon_end': 0.0,
        'target_update_freq': config['dqn']['target_update_freq'],
        'max_grad_norm': config['dqn']['max_grad_norm'],
        'tau': config['dqn'].get('tau', 0.005),
    }

    results = {}

    # Evaluate trained model
    if os.path.exists(args.checkpoint):
        print(f"\nLoading checkpoint: {args.checkpoint}")
        agent = DQNGCNTransformerAgent(agent_config)
        agent.load(args.checkpoint)
        agent.epsilon = 0.0  # Disable exploration

        print(f"Evaluating trained model ({args.episodes} episodes)...")
        results['GCN-Transformer-DQN'] = evaluate_model(agent, env, args.episodes)
        print(f"  Success Rate: {results['GCN-Transformer-DQN']['success_rate']:.1f}%")
        print(f"  Avg Hops: {results['GCN-Transformer-DQN']['avg_hops']:.1f}")
    else:
        print(f"\nWarning: Checkpoint not found at {args.checkpoint}")
        print("Skipping trained model evaluation.")

    # Evaluate baselines
    print(f"\nEvaluating Shortest Path baseline ({args.episodes} episodes)...")
    results['Shortest Path'] = evaluate_baseline(env, 'shortest_path', args.episodes)
    print(f"  Success Rate: {results['Shortest Path']['success_rate']:.1f}%")

    print(f"\nEvaluating Greedy baseline ({args.episodes} episodes)...")
    results['Greedy'] = evaluate_baseline(env, 'greedy', args.episodes)
    print(f"  Success Rate: {results['Greedy']['success_rate']:.1f}%")

    print(f"\nEvaluating Random baseline ({args.episodes} episodes)...")
    results['Random'] = evaluate_baseline(env, 'random', args.episodes)
    print(f"  Success Rate: {results['Random']['success_rate']:.1f}%")

    # Print comparison
    print_comparison_table(results)

    # Save results
    results_path = 'evaluation_results_fixed.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
