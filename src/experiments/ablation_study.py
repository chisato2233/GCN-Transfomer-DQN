"""
Ablation Study Script for SAGIN Intelligent Routing.

Compares different model configurations:
1. Full model: GCN + Transformer + State Encoder
2. No GCN: Transformer + State Encoder only
3. No Transformer: GCN + State Encoder only
4. State Only: Just State Encoder (baseline)
5. With/Without PER
6. With/Without Double DQN
7. With/Without Dueling DQN
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn_gcn_transformer import DQNGCNTransformerAgent
from src.agents.dqn import DQNAgent
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker


# ============================================================================
# Ablation Configurations
# ============================================================================

ABLATION_CONFIGS = {
    # Full model with all components
    'full_model': {
        'name': 'GCN + Transformer + State (Full)',
        'use_gcn': True,
        'use_transformer': True,
        'use_dueling': True,
        'use_double_dqn': True,
        'use_per': False,
    },

    # Without GCN
    'no_gcn': {
        'name': 'Transformer + State (No GCN)',
        'use_gcn': False,
        'use_transformer': True,
        'use_dueling': True,
        'use_double_dqn': True,
        'use_per': False,
    },

    # Without Transformer
    'no_transformer': {
        'name': 'GCN + State (No Transformer)',
        'use_gcn': True,
        'use_transformer': False,
        'use_dueling': True,
        'use_double_dqn': True,
        'use_per': False,
    },

    # State only (baseline)
    'state_only': {
        'name': 'State Only (Baseline)',
        'use_gcn': False,
        'use_transformer': False,
        'use_dueling': True,
        'use_double_dqn': True,
        'use_per': False,
    },

    # Full model with PER
    'full_with_per': {
        'name': 'Full Model + PER',
        'use_gcn': True,
        'use_transformer': True,
        'use_dueling': True,
        'use_double_dqn': True,
        'use_per': True,
    },

    # Without Double DQN
    'no_double_dqn': {
        'name': 'Full Model (No Double DQN)',
        'use_gcn': True,
        'use_transformer': True,
        'use_dueling': True,
        'use_double_dqn': False,
        'use_per': False,
    },

    # Without Dueling
    'no_dueling': {
        'name': 'Full Model (No Dueling)',
        'use_gcn': True,
        'use_transformer': True,
        'use_dueling': False,
        'use_double_dqn': True,
        'use_per': False,
    },
}


# ============================================================================
# Ablation Agent (supports disabling GCN/Transformer)
# ============================================================================

class AblationAgent(DQNGCNTransformerAgent):
    """Agent that can disable GCN or Transformer for ablation studies."""

    def __init__(self, config: dict):
        self.use_gcn = config.get('use_gcn', True)
        self.use_transformer = config.get('use_transformer', True)
        super().__init__(config)

    def select_action(self,
                      state: np.ndarray,
                      state_history: np.ndarray,
                      current_node_features: np.ndarray,
                      neighbor_features: np.ndarray,
                      neighbor_mask: np.ndarray,
                      action_mask: np.ndarray,
                      training: bool = True) -> int:
        """Select action, potentially ignoring some features."""
        # If not using certain components, zero them out
        if not self.use_gcn:
            current_node_features = np.zeros_like(current_node_features)
            neighbor_features = np.zeros_like(neighbor_features)

        if not self.use_transformer:
            state_history = np.zeros_like(state_history)

        return super().select_action(
            state, state_history,
            current_node_features, neighbor_features, neighbor_mask,
            action_mask, training
        )


# ============================================================================
# Training Functions
# ============================================================================

def get_graph_features(env: SAGINRoutingEnv) -> Dict[str, np.ndarray]:
    """Extract graph features from environment."""
    local_graph = env.get_local_graph_data()
    return {
        'current_node_features': local_graph['current_node_features'],
        'neighbor_features': local_graph['neighbor_features'],
        'neighbor_mask': local_graph['neighbor_mask']
    }


def train_single_config(
    config: dict,
    ablation_config: dict,
    log_dir: str,
    num_episodes: int = 2000,
    eval_frequency: int = 100,
    num_eval_episodes: int = 10
) -> Dict:
    """Train a single ablation configuration."""

    config_name = ablation_config['name']
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")

    # Merge ablation config into agent config
    agent_config = deepcopy(config.get('agent', {}))
    agent_config.update(ablation_config)

    # Environment dimensions
    env = SAGINRoutingEnv(config)
    agent_config['state_dim'] = env.state_dim
    agent_config['num_actions'] = env.action_space.n
    agent_config['node_feature_dim'] = env.node_feature_dim
    agent_config['max_neighbors'] = env.max_neighbors
    agent_config['history_length'] = env.history_length

    # DQN parameters
    dqn_config = config.get('dqn', {})
    agent_config['lr'] = dqn_config.get('lr', 3e-4)
    agent_config['target_update_freq'] = dqn_config.get('target_update_freq', 100)

    # Buffer parameters
    buffer_config = config.get('buffer', {})
    agent_config['buffer_capacity'] = buffer_config.get('capacity', 100000)
    agent_config['batch_size'] = buffer_config.get('batch_size', 64)

    # Exploration
    exploration_config = config.get('exploration', {})
    agent_config['epsilon_start'] = exploration_config.get('epsilon_start', 1.0)
    agent_config['epsilon_end'] = exploration_config.get('epsilon_end', 0.01)
    agent_config['epsilon_decay'] = exploration_config.get('epsilon_decay', 0.995)

    # Training
    training_config = config.get('training', {})
    agent_config['gamma'] = training_config.get('gamma', 0.99)

    # Feature dimensions
    agent_config['gcn_output_dim'] = config.get('gcn', {}).get('output_dim', 64)
    agent_config['transformer_output_dim'] = config.get('transformer', {}).get('d_model', 64)
    agent_config['fused_dim'] = config.get('fusion', {}).get('output_dim', 128)
    agent_config['state_encoded_dim'] = 64

    # Hardware
    agent_config['device'] = config.get('hardware', {}).get('device', 'cuda')

    # Create agent
    agent = AblationAgent(agent_config)

    # Training metrics
    episode_rewards = []
    eval_rewards = []
    eval_delivery_rates = []
    best_reward = float('-inf')

    max_steps = training_config.get('max_steps_per_episode', 50)

    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc=config_name[:20], ncols=80):
        obs, info = env.reset()
        state = obs['state']
        action_mask = obs['action_mask']
        graph_features = get_graph_features(env)
        state_history = env.get_state_history()

        episode_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(
                state=state,
                state_history=state_history,
                current_node_features=graph_features['current_node_features'],
                neighbor_features=graph_features['neighbor_features'],
                neighbor_mask=graph_features['neighbor_mask'],
                action_mask=action_mask,
                training=True
            )

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            next_state = next_obs['state']
            next_action_mask = next_obs['action_mask']
            next_graph_features = get_graph_features(env)
            next_state_history = env.get_state_history()

            agent.store_transition(
                state=state,
                state_history=state_history,
                current_node_features=graph_features['current_node_features'],
                neighbor_features=graph_features['neighbor_features'],
                neighbor_mask=graph_features['neighbor_mask'],
                action=action,
                reward=reward,
                next_state=next_state,
                next_state_history=next_state_history,
                next_current_node_features=next_graph_features['current_node_features'],
                next_neighbor_features=next_graph_features['neighbor_features'],
                next_neighbor_mask=next_graph_features['neighbor_mask'],
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask
            )

            agent.train()
            episode_reward += reward

            if done:
                break

            state = next_state
            action_mask = next_action_mask
            graph_features = next_graph_features
            state_history = next_state_history

        agent.end_episode()
        episode_rewards.append(episode_reward)

        # Evaluation
        if episode % eval_frequency == 0:
            eval_reward, delivery_rate = evaluate(env, agent, num_eval_episodes)
            eval_rewards.append(eval_reward)
            eval_delivery_rates.append(delivery_rate)

            if eval_reward > best_reward:
                best_reward = eval_reward

    # Final results
    results = {
        'name': config_name,
        'config': ablation_config,
        'episode_rewards': episode_rewards,
        'eval_rewards': eval_rewards,
        'eval_delivery_rates': eval_delivery_rates,
        'best_reward': best_reward,
        'final_delivery_rate': eval_delivery_rates[-1] if eval_delivery_rates else 0.0,
        'avg_last_100_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    }

    return results


def evaluate(env, agent, num_episodes: int) -> Tuple[float, float]:
    """Evaluate agent performance."""
    total_reward = 0.0
    num_success = 0

    for _ in range(num_episodes):
        obs, info = env.reset()
        state = obs['state']
        action_mask = obs['action_mask']
        graph_features = get_graph_features(env)
        state_history = env.get_state_history()

        episode_reward = 0.0

        for _ in range(env.max_hops):
            action = agent.select_action(
                state=state,
                state_history=state_history,
                current_node_features=graph_features['current_node_features'],
                neighbor_features=graph_features['neighbor_features'],
                neighbor_mask=graph_features['neighbor_mask'],
                action_mask=action_mask,
                training=False
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

            state = next_obs['state']
            action_mask = next_obs['action_mask']
            graph_features = get_graph_features(env)
            state_history = env.get_state_history()

        total_reward += episode_reward
        if info.get('success', False):
            num_success += 1

    return total_reward / num_episodes, num_success / num_episodes


# ============================================================================
# Main
# ============================================================================

def run_ablation_study(config_path: str, output_dir: str, configs_to_run: List[str] = None):
    """Run complete ablation study."""

    # Load base config
    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(output_dir, f'ablation_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Determine which configs to run
    if configs_to_run is None:
        configs_to_run = list(ABLATION_CONFIGS.keys())

    print(f"\n{'='*60}")
    print("SAGIN Routing Ablation Study")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Configurations to run: {configs_to_run}")

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Training parameters
    num_episodes = base_config.get('training', {}).get('num_episodes', 2000)
    # Reduce episodes for faster ablation
    num_episodes = min(num_episodes, 2000)

    # Run each configuration
    all_results = {}

    for config_key in configs_to_run:
        if config_key not in ABLATION_CONFIGS:
            print(f"Unknown config: {config_key}, skipping...")
            continue

        ablation_config = ABLATION_CONFIGS[config_key]
        results = train_single_config(
            base_config,
            ablation_config,
            output_dir,
            num_episodes=num_episodes
        )

        all_results[config_key] = results

        # Save intermediate results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON
            json_results = {}
            for k, v in all_results.items():
                json_results[k] = {
                    'name': v['name'],
                    'best_reward': float(v['best_reward']),
                    'final_delivery_rate': float(v['final_delivery_rate']),
                    'avg_last_100_reward': float(v['avg_last_100_reward']),
                    'eval_rewards': [float(x) for x in v['eval_rewards']],
                    'eval_delivery_rates': [float(x) for x in v['eval_delivery_rates']]
                }
            json.dump(json_results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Ablation Study Results Summary")
    print(f"{'='*60}")
    print(f"{'Configuration':<35} {'Best Reward':>12} {'Delivery Rate':>14}")
    print("-" * 65)

    for config_key, results in all_results.items():
        print(f"{results['name']:<35} {results['best_reward']:>12.2f} {results['final_delivery_rate']*100:>13.1f}%")

    print(f"\nResults saved to: {output_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='SAGIN Routing Ablation Study')
    parser.add_argument('--config', '-c', type=str,
                       default='configs/routing_config.yaml',
                       help='Base configuration file')
    parser.add_argument('--output', '-o', type=str,
                       default='logs/ablation',
                       help='Output directory')
    parser.add_argument('--configs', '-C', nargs='+',
                       default=None,
                       help='Specific configs to run (default: all)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')

    args = parser.parse_args()

    if args.list_configs:
        print("\nAvailable ablation configurations:")
        for key, config in ABLATION_CONFIGS.items():
            print(f"  {key}: {config['name']}")
        return

    run_ablation_study(args.config, args.output, args.configs)


if __name__ == '__main__':
    main()
