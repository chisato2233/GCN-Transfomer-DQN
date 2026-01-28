"""
Comprehensive Evaluation Script for SAGIN Routing Methods.

Compares our V3 Per-Neighbor Q-Value agent against:
1. Traditional Routing: OSPF, AODV, ECMP, Load-Aware
2. Classic RL: Q-Routing, Predictive Q-Routing
3. Recent DRL (2024-2025): GAT-DQN, GraphSAGE-DQN, GCN-DQN, Dueling DQN
4. Simple Baselines: Greedy, Random

References:
- Q-Routing: Boyan & Littman (1993), Q-adaptive (2024)
- GAT-DRL: "GNN for Routing Optimization" (MDPI 2024)
- GraphSAGE-DQN: "LEO Satellite Network Routing" (MDPI 2024)
- Dueling DQN: "Dueling DQN Routing in SDN" (Wireless Networks 2024)

Usage:
    python src/experiments/evaluate_comprehensive.py \
        --v3_checkpoint logs/v3_xxx/checkpoints/best_model.pt \
        --config configs/routing_config.yaml \
        --episodes 200
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
from datetime import datetime

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn_gcn_transformer_v3 import DQNGCNTransformerAgentV3
from src.baselines.q_routing import QRoutingAgent, PredictiveQRouting
from src.baselines.traditional_routing import OSPFRouter, ECMPRouter, AODVRouter, LoadAwareRouter
from src.baselines.gnn_baselines import SAGINGNNAgent


class SimpleBaselines:
    """Simple baseline algorithms."""

    def __init__(self, env: SAGINRoutingEnv):
        self.env = env

    def greedy_action(self, current: int, destination: int,
                      neighbors: List[int], action_mask: np.ndarray,
                      visited: set = None) -> int:
        """Greedy: choose neighbor closest to destination."""
        dest_pos = self.env.topology.nodes[destination].position
        visited = visited or set()

        best_action = -1
        best_dist = float('inf')

        # Prefer unvisited neighbors
        for i, neighbor in enumerate(neighbors):
            if i < len(action_mask) and action_mask[i] > 0:
                if neighbor not in visited:
                    pos = self.env.topology.nodes[neighbor].position
                    dist = np.linalg.norm(pos - dest_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_action = i

        # Fallback to any valid
        if best_action == -1:
            for i, neighbor in enumerate(neighbors):
                if i < len(action_mask) and action_mask[i] > 0:
                    pos = self.env.topology.nodes[neighbor].position
                    dist = np.linalg.norm(pos - dest_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_action = i

        return best_action if best_action >= 0 else 0

    def random_action(self, action_mask: np.ndarray) -> int:
        """Random valid action."""
        valid = np.where(action_mask > 0)[0]
        return int(np.random.choice(valid)) if len(valid) > 0 else 0


def evaluate_v3_agent(agent, env, num_episodes: int) -> Dict:
    """Evaluate V3 Per-Neighbor Q-Value agent."""
    metrics = defaultdict(list)
    max_steps = env.max_hops * 2

    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"    V3 Agent: {ep}/{num_episodes}...", flush=True)

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

    return compute_stats(metrics)


def evaluate_q_routing(env, num_episodes: int, variant: str = 'standard') -> Dict:
    """Evaluate Q-Routing variants."""
    num_nodes = env.topology.total_nodes

    if variant == 'standard':
        agent = QRoutingAgent(num_nodes)
    else:
        agent = PredictiveQRouting(num_nodes)

    metrics = defaultdict(list)
    max_steps = env.max_hops * 2

    # Training phase
    print(f"    {variant} Q-Routing: Training...", flush=True)
    for _ in range(500):  # Training episodes
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            current = env.current_node
            dest = env.destination
            neighbors = env.topology.get_neighbors(current)
            action_mask = obs['action_mask']

            action = agent.select_action(current, dest, neighbors, action_mask, training=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get transition info for Q-update
            next_neighbors = env.topology.get_neighbors(env.current_node)
            link = env.topology.get_link(current, neighbors[action] if action < len(neighbors) else current)
            trans_time = link.delay if link else 1.0

            agent.update(current, dest, neighbors[action] if action < len(neighbors) else current,
                        trans_time, next_neighbors)
            steps += 1

        agent.decay_epsilon()

    # Evaluation phase
    agent.epsilon = 0.0
    print(f"    {variant} Q-Routing: Evaluating...", flush=True)

    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"    {variant} Q-Routing: {ep}/{num_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            current = env.current_node
            dest = env.destination
            neighbors = env.topology.get_neighbors(current)
            action_mask = obs['action_mask']

            action = agent.select_action(current, dest, neighbors, action_mask, training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        metrics['reward'].append(episode_reward)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['hops'].append(info.get('hop_count', 0))
        metrics['delay'].append(info.get('total_delay', 0))

        if info.get('success', False) and info.get('optimal_hops', 0) > 0:
            metrics['efficiency'].append(info['optimal_hops'] / info.get('hop_count', 1))

    return compute_stats(metrics)


def evaluate_traditional(env, router_type: str, num_episodes: int) -> Dict:
    """Evaluate traditional routing protocols."""
    if router_type == 'ospf':
        router = OSPFRouter(env)
    elif router_type == 'ecmp':
        router = ECMPRouter(env)
    elif router_type == 'aodv':
        router = AODVRouter(env)
    elif router_type == 'load_aware':
        router = LoadAwareRouter(env)
    else:
        raise ValueError(f"Unknown router: {router_type}")

    metrics = defaultdict(list)
    max_steps = env.max_hops * 2

    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"    {router_type.upper()}: {ep}/{num_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        # Clear cache for fresh routing
        if hasattr(router, 'clear_cache'):
            router.clear_cache()

        while not done and steps < max_steps:
            current = env.current_node
            dest = env.destination
            neighbors = env.topology.get_neighbors(current)
            action_mask = obs['action_mask']

            action = router.get_next_hop(current, dest, neighbors, action_mask)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        metrics['reward'].append(episode_reward)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['hops'].append(info.get('hop_count', 0))
        metrics['delay'].append(info.get('total_delay', 0))

        if info.get('success', False) and info.get('optimal_hops', 0) > 0:
            metrics['efficiency'].append(info['optimal_hops'] / info.get('hop_count', 1))

    return compute_stats(metrics)


def evaluate_simple_baseline(env, baseline_type: str, num_episodes: int) -> Dict:
    """Evaluate simple baselines (greedy, random)."""
    baseline = SimpleBaselines(env)
    metrics = defaultdict(list)
    max_steps = env.max_hops * 2

    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"    {baseline_type}: {ep}/{num_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        visited = {env.current_node}

        while not done and steps < max_steps:
            action_mask = obs['action_mask']
            current = env.current_node
            dest = env.destination
            neighbors = env.topology.get_neighbors(current)

            if baseline_type == 'greedy':
                action = baseline.greedy_action(current, dest, neighbors, action_mask, visited)
            else:
                action = baseline.random_action(action_mask)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            visited.add(env.current_node)

        metrics['reward'].append(episode_reward)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['hops'].append(info.get('hop_count', 0))
        metrics['delay'].append(info.get('total_delay', 0))

        if info.get('success', False) and info.get('optimal_hops', 0) > 0:
            metrics['efficiency'].append(info['optimal_hops'] / info.get('hop_count', 1))

    return compute_stats(metrics)


def evaluate_gnn_baseline(env, network_type: str, config: dict,
                           num_episodes: int, train_episodes: int = 1000) -> Dict:
    """
    Train and evaluate GNN baseline agents (GAT, GraphSAGE, GCN, Dueling).

    Args:
        env: SAGIN environment
        network_type: One of 'gat', 'graphsage', 'gcn', 'dueling'
        config: Configuration dict
        num_episodes: Number of evaluation episodes
        train_episodes: Number of training episodes
    """
    agent_config = {
        'feature_dim': 14,  # 8 routing + 6 three-layer features
        'max_neighbors': config['environment']['max_neighbors'],
        'hidden_dim': 64,
        'n_heads': 4,
        'device': config['hardware']['device'],
        'lr': 1e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 128,
        'buffer_capacity': 50000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.995,
        'max_grad_norm': 1.0,
    }

    agent = SAGINGNNAgent(agent_config, network_type=network_type)
    max_steps = env.max_hops * 2

    # Training phase
    print(f"    {network_type.upper()}: Training ({train_episodes} episodes)...", flush=True)
    for ep in range(train_episodes):
        if ep % 200 == 0:
            print(f"    {network_type.upper()}: Training {ep}/{train_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        steps = 0

        # Get initial state
        topo_data = env.get_topology_aware_features()
        neighbor_features = topo_data['neighbor_topology_features']
        neighbor_mask = topo_data['neighbor_mask']
        action_mask = obs['action_mask']

        while not done and steps < max_steps:
            action = agent.select_action(
                neighbor_features, neighbor_mask, action_mask, training=True
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            # Get next state
            next_topo_data = env.get_topology_aware_features()
            next_neighbor_features = next_topo_data['neighbor_topology_features']
            next_neighbor_mask = next_topo_data['neighbor_mask']
            next_action_mask = obs['action_mask']

            # Store transition
            agent.store_transition(
                neighbor_features, neighbor_mask, action, reward,
                next_neighbor_features, next_neighbor_mask, done,
                action_mask, next_action_mask
            )

            # Train
            agent.train()

            # Update state
            neighbor_features = next_neighbor_features
            neighbor_mask = next_neighbor_mask
            action_mask = next_action_mask

        agent.end_episode()

    # Evaluation phase
    agent.epsilon = 0.0
    metrics = defaultdict(list)

    print(f"    {network_type.upper()}: Evaluating ({num_episodes} episodes)...", flush=True)
    for ep in range(num_episodes):
        if ep % 50 == 0:
            print(f"    {network_type.upper()}: {ep}/{num_episodes}...", flush=True)

        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            topo_data = env.get_topology_aware_features()
            neighbor_features = topo_data['neighbor_topology_features']
            neighbor_mask = topo_data['neighbor_mask']
            action_mask = obs['action_mask']

            action = agent.select_action(
                neighbor_features, neighbor_mask, action_mask, training=False
            )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        metrics['reward'].append(episode_reward)
        metrics['success'].append(1.0 if info.get('success', False) else 0.0)
        metrics['hops'].append(info.get('hop_count', 0))
        metrics['delay'].append(info.get('total_delay', 0))

        if info.get('success', False) and info.get('optimal_hops', 0) > 0:
            metrics['efficiency'].append(info['optimal_hops'] / info.get('hop_count', 1))

    return compute_stats(metrics)


def compute_stats(metrics: Dict) -> Dict:
    """Compute statistics from metrics."""
    return {
        'avg_reward': float(np.mean(metrics['reward'])),
        'std_reward': float(np.std(metrics['reward'])),
        'success_rate': float(np.mean(metrics['success']) * 100),
        'avg_hops': float(np.mean(metrics['hops'])),
        'avg_delay': float(np.mean(metrics['delay'])),
        'path_efficiency': float(np.mean(metrics['efficiency'])) if metrics['efficiency'] else 0.0
    }


def print_results(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE PERFORMANCE COMPARISON")
    print("=" * 100)

    # Categorize methods
    categories = {
        'Our Method': ['V3 Per-Neighbor'],
        'GNN-DRL (2024)': ['GAT-DQN', 'GraphSAGE-DQN', 'GCN-DQN', 'Dueling-DQN'],
        'Traditional (2024 refs)': ['OSPF', 'ECMP', 'AODV', 'Load-Aware'],
        'Classic RL': ['Q-Routing', 'PQ-Routing'],
        'Simple Baselines': ['Greedy', 'Random'],
    }

    info_types = {
        'V3 Per-Neighbor': 'Local+Temporal',
        'GAT-DQN': 'Local+GNN',
        'GraphSAGE-DQN': 'Local+GNN',
        'GCN-DQN': 'Local+GNN',
        'Dueling-DQN': 'Local',
        'OSPF': 'Global',
        'ECMP': 'Global',
        'AODV': 'On-demand',
        'Load-Aware': 'Global+Load',
        'Q-Routing': 'Local',
        'PQ-Routing': 'Local',
        'Greedy': 'Local',
        'Random': 'Local',
    }

    print(f"\n{'Method':<20} {'Category':<18} {'Info Type':<12} {'Success%':>10} {'Avg Hops':>10} {'Efficiency':>12} {'Avg Delay':>12}")
    print("-" * 100)

    # Sort by success rate
    sorted_results = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)

    for method, m in sorted_results:
        # Find category
        cat = 'Other'
        for c, methods in categories.items():
            if method in methods:
                cat = c
                break

        eff_str = f"{m['path_efficiency']*100:.1f}%" if m['path_efficiency'] > 0 else "N/A"
        info_type = info_types.get(method, 'Unknown')

        print(f"{method:<20} {cat:<18} {info_type:<12} {m['success_rate']:>9.1f}% {m['avg_hops']:>10.1f} {eff_str:>12} {m['avg_delay']:>11.2f}ms")

    print("=" * 100)

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS vs BASELINES")
    print("=" * 60)

    v3_rate = results.get('V3 Per-Neighbor', {}).get('success_rate', 0)

    comparisons = [
        ('Greedy', 'Simple heuristic baseline'),
        ('Q-Routing', 'Classic RL (Boyan 1993)'),
        ('OSPF', 'Traditional protocol (global info)'),
    ]

    for method, desc in comparisons:
        if method in results:
            other_rate = results[method]['success_rate']
            diff = v3_rate - other_rate
            status = '✓ Wins' if diff > 0 else '✗ Loses' if diff < 0 else '= Tie'
            print(f"V3 vs {method:<12}: {diff:+6.1f}%  ({status}) - {desc}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Routing Evaluation')
    parser.add_argument('--v3_checkpoint', type=str, required=True,
                        help='Path to V3 model checkpoint')
    parser.add_argument('--config', type=str, default='configs/routing_config.yaml',
                        help='Config file')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of evaluation episodes per method')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("SAGIN Routing - Comprehensive Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Create environment
    env = SAGINRoutingEnv(config)
    print(f"Environment: {env.topology.total_nodes} nodes")
    print(f"Evaluation episodes: {args.episodes}")

    # Load V3 agent
    agent_config = {
        'num_actions': config['environment']['max_neighbors'],
        'max_neighbors': config['gcn']['max_neighbors'],
        'history_length': config['transformer']['history_length'],
        'topology_feature_dim': 14,
        'simplified_history_dim': 6,
        'neighbor_hidden_dim': 64,
        'temporal_d_model': 32,
        'device': config['hardware']['device'],
        'use_transformer': True,
    }

    v3_agent = DQNGCNTransformerAgentV3(agent_config)
    v3_agent.load(args.v3_checkpoint)
    v3_agent.epsilon = 0.0
    print(f"Loaded V3 checkpoint: {args.v3_checkpoint}")

    results = {}

    # ==================== Evaluate All Methods ====================

    print("\n[1/12] Evaluating V3 Per-Neighbor Agent...")
    results['V3 Per-Neighbor'] = evaluate_v3_agent(v3_agent, env, args.episodes)
    print(f"    Success Rate: {results['V3 Per-Neighbor']['success_rate']:.1f}%")

    print("\n[2/12] Evaluating Greedy Baseline...")
    results['Greedy'] = evaluate_simple_baseline(env, 'greedy', args.episodes)
    print(f"    Success Rate: {results['Greedy']['success_rate']:.1f}%")

    print("\n[3/12] Evaluating Random Baseline...")
    results['Random'] = evaluate_simple_baseline(env, 'random', args.episodes)
    print(f"    Success Rate: {results['Random']['success_rate']:.1f}%")

    print("\n[4/12] Evaluating OSPF (Shortest Path)...")
    results['OSPF'] = evaluate_traditional(env, 'ospf', args.episodes)
    print(f"    Success Rate: {results['OSPF']['success_rate']:.1f}%")

    print("\n[5/12] Evaluating ECMP (Multi-Path)...")
    results['ECMP'] = evaluate_traditional(env, 'ecmp', args.episodes)
    print(f"    Success Rate: {results['ECMP']['success_rate']:.1f}%")

    print("\n[6/12] Evaluating AODV (On-demand)...")
    results['AODV'] = evaluate_traditional(env, 'aodv', args.episodes)
    print(f"    Success Rate: {results['AODV']['success_rate']:.1f}%")

    print("\n[7/12] Evaluating Q-Routing (Classic RL)...")
    results['Q-Routing'] = evaluate_q_routing(env, args.episodes, 'standard')
    print(f"    Success Rate: {results['Q-Routing']['success_rate']:.1f}%")

    print("\n[8/12] Evaluating Predictive Q-Routing...")
    results['PQ-Routing'] = evaluate_q_routing(env, args.episodes, 'predictive')
    print(f"    Success Rate: {results['PQ-Routing']['success_rate']:.1f}%")

    # ==================== GNN Baselines (2024) ====================
    gnn_train_episodes = 1000  # Training episodes for GNN baselines

    print("\n[9/12] Evaluating GAT-DQN (Graph Attention Network)...")
    results['GAT-DQN'] = evaluate_gnn_baseline(env, 'gat', config, args.episodes, gnn_train_episodes)
    print(f"    Success Rate: {results['GAT-DQN']['success_rate']:.1f}%")

    print("\n[10/12] Evaluating GraphSAGE-DQN...")
    results['GraphSAGE-DQN'] = evaluate_gnn_baseline(env, 'graphsage', config, args.episodes, gnn_train_episodes)
    print(f"    Success Rate: {results['GraphSAGE-DQN']['success_rate']:.1f}%")

    print("\n[11/12] Evaluating GCN-DQN (Standard GCN)...")
    results['GCN-DQN'] = evaluate_gnn_baseline(env, 'gcn', config, args.episodes, gnn_train_episodes)
    print(f"    Success Rate: {results['GCN-DQN']['success_rate']:.1f}%")

    print("\n[12/12] Evaluating Dueling-DQN (No GNN)...")
    results['Dueling-DQN'] = evaluate_gnn_baseline(env, 'dueling', config, args.episodes, gnn_train_episodes)
    print(f"    Success Rate: {results['Dueling-DQN']['success_rate']:.1f}%")

    # Print comparison
    print_results(results)

    # Save results
    output_file = 'evaluation_comprehensive.yaml'
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {output_file}")

    # Generate summary for paper
    print("\n" + "=" * 60)
    print("PAPER-READY SUMMARY")
    print("=" * 60)

    # Compute improvements
    v3_rate = results['V3 Per-Neighbor']['success_rate']
    gat_rate = results.get('GAT-DQN', {}).get('success_rate', 0)
    sage_rate = results.get('GraphSAGE-DQN', {}).get('success_rate', 0)
    gcn_rate = results.get('GCN-DQN', {}).get('success_rate', 0)

    print(f"""
Method Comparison ({args.episodes} episodes):

=== Our Method ===
- V3 Per-Neighbor (Ours): {v3_rate:.1f}% success

=== GNN-DRL Baselines (2024) ===
- GAT-DQN:       {results.get('GAT-DQN', {}).get('success_rate', 0):.1f}% success
- GraphSAGE-DQN: {results.get('GraphSAGE-DQN', {}).get('success_rate', 0):.1f}% success
- GCN-DQN:       {results.get('GCN-DQN', {}).get('success_rate', 0):.1f}% success
- Dueling-DQN:   {results.get('Dueling-DQN', {}).get('success_rate', 0):.1f}% success

=== Traditional & Classic ===
- OSPF (Global):  {results['OSPF']['success_rate']:.1f}% success
- Q-Routing:      {results['Q-Routing']['success_rate']:.1f}% success
- Greedy:         {results['Greedy']['success_rate']:.1f}% success

Key Findings:
1. V3 vs Best GNN Baseline: {v3_rate - max(gat_rate, sage_rate, gcn_rate):+.1f}%
2. V3 vs Greedy (local only): {v3_rate - results['Greedy']['success_rate']:+.1f}%
3. Per-Neighbor architecture preserves neighbor-action correspondence
""")


if __name__ == '__main__':
    main()
