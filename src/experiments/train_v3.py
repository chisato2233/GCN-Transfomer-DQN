"""
Training script for Per-Neighbor Q-Value Agent (Version 3).

核心改进：
1. Per-Neighbor Q-Value 架构 - 保持 neighbor-action 对应关系
2. 每个动作的 Q 值直接从对应邻居的特征计算
3. Transformer 作为全局时间上下文

Usage:
    python src/experiments/train_v3.py --config configs/routing_config.yaml
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
from datetime import datetime
from pathlib import Path

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn_gcn_transformer_v3 import DQNGCNTransformerAgentV3


def setup_logging(log_dir: str):
    """Setup logging directory."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{log_dir}/checkpoints").mkdir(exist_ok=True)
    return log_dir


def log_message(log_file: str, message: str):
    """Log message to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{timestamp} | {message}"
    print(full_message, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(full_message + '\n')


def evaluate_agent(agent, env, num_episodes: int = 50, max_steps: int = 50) -> dict:
    """Evaluate agent performance."""
    total_rewards = []
    total_successes = []
    total_hops = []
    total_delays = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            # Get topology-aware features
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

        total_rewards.append(episode_reward)
        total_successes.append(1.0 if info.get('success', False) else 0.0)
        if info.get('success', False):
            total_hops.append(info.get('hop_count', 0))
            total_delays.append(info.get('total_delay', 0))

    return {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'success_rate': np.mean(total_successes) * 100,
        'avg_hops': np.mean(total_hops) if total_hops else 0,
        'avg_delay': np.mean(total_delays) if total_delays else 0
    }


def train(config_path: str, num_episodes: int = 3000,
          use_transformer: bool = True, experiment_name: str = None):
    """
    Train Per-Neighbor Q-Value agent (V3).

    Args:
        config_path: Path to config file
        num_episodes: Number of training episodes
        use_transformer: Whether to use Transformer for temporal context
        experiment_name: Name for this experiment
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Setup experiment name
    if experiment_name is None:
        trans_str = "trans" if use_transformer else "notrans"
        experiment_name = f"v3_per_neighbor_{trans_str}"

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/{experiment_name}_{timestamp}"
    setup_logging(log_dir)
    log_file = f"{log_dir}/train.log"

    log_message(log_file, "=" * 60)
    log_message(log_file, "SAGIN 空天地三层网络联合优化 (V3 Enhanced)")
    log_message(log_file, f"use_transformer={use_transformer}")
    log_message(log_file, "=" * 60)
    log_message(log_file, "")
    log_message(log_file, "V3 Enhanced 特性:")
    log_message(log_file, "  1. Per-Neighbor Q-Value 架构 - 保持 neighbor-action 对应")
    log_message(log_file, "  2. 14维特征 = 8路由特征 + 6三层网络特征")
    log_message(log_file, "     - 节点类型 [satellite, uav, ground]")
    log_message(log_file, "     - 能量水平 (UAV关键)")
    log_message(log_file, "     - 队列拥塞度")
    log_message(log_file, "     - 层间切换指示")
    log_message(log_file, "  3. 三层联合优化奖励: 能量感知 + 拥塞感知 + 层间切换")
    log_message(log_file, "")

    # Create environment
    env = SAGINRoutingEnv(config)
    log_message(log_file, f"Environment: {env.topology.total_nodes} nodes")

    # Create agent config
    agent_config = {
        'num_actions': config['environment']['max_neighbors'],
        'max_neighbors': config['gcn']['max_neighbors'],
        'history_length': config['transformer']['history_length'],
        # V3 Enhanced: 14 dims = 8 routing + 6 three-layer features
        'topology_feature_dim': 14,  # 8 routing + 6 SAGIN layer features
        'simplified_history_dim': 6,  # Simplified history
        'neighbor_hidden_dim': 64,  # Per-neighbor encoding dimension
        'temporal_d_model': 32,  # Temporal context dimension
        'device': config['hardware']['device'],
        'lr': config['dqn']['lr'],
        'gamma': config['training']['gamma'],
        'tau': config['dqn'].get('tau', 0.005),
        'epsilon_start': config['exploration']['epsilon_start'],
        'epsilon_end': config['exploration']['epsilon_end'],
        'epsilon_decay': config['exploration']['epsilon_decay'],
        'buffer_capacity': config['buffer']['capacity'],
        'batch_size': config['buffer']['batch_size'],
        'max_grad_norm': config['dqn']['max_grad_norm'],
        'use_transformer': use_transformer,
    }

    agent = DQNGCNTransformerAgentV3(agent_config)
    log_message(log_file, f"Agent created: Per-Neighbor Q-Value + Temporal Context")
    log_message(log_file, f"Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")

    # Training parameters
    eval_frequency = config['training'].get('eval_frequency', 50)
    num_eval_episodes = config['training'].get('num_eval_episodes', 50)
    max_steps = env.max_hops * 2
    patience = config['training'].get('patience', 800)
    warmup_episodes = config['training'].get('warmup_episodes', 300)

    # Training tracking
    best_eval_reward = float('-inf')
    best_success_rate = 0.0
    best_eval_episode = 0
    episodes_without_improvement = 0

    log_message(log_file, f"\nStarting training for {num_episodes} episodes...")
    log_message(log_file, f"Eval frequency: {eval_frequency}, Patience: {patience}, Warmup: {warmup_episodes}")
    log_message(log_file, "-" * 60)

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < max_steps:
            # Get features
            topo_data = env.get_topology_aware_features()
            simplified_hist = env.get_simplified_history()
            action_mask = obs['action_mask']

            # Select action
            action = agent.select_action(
                topo_data['neighbor_topology_features'],
                topo_data['neighbor_mask'],
                simplified_hist,
                action_mask,
                training=True
            )

            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Get next state features
            next_topo_data = env.get_topology_aware_features()
            next_simplified_hist = env.get_simplified_history()
            next_action_mask = next_obs['action_mask']

            # Store transition
            agent.store_transition(
                topo_data['neighbor_topology_features'],
                topo_data['neighbor_mask'],
                simplified_hist,
                action,
                reward,
                next_topo_data['neighbor_topology_features'],
                next_topo_data['neighbor_mask'],
                next_simplified_hist,
                done,
                action_mask,
                next_action_mask
            )

            # Train
            agent.train()

            episode_reward += reward
            steps += 1
            obs = next_obs

        # End of episode
        agent.end_episode()

        # Evaluation
        if episode % eval_frequency == 0:
            eval_results = evaluate_agent(agent, env, num_eval_episodes, max_steps)

            log_message(
                log_file,
                f"Ep {episode:4d}: Reward={eval_results['avg_reward']:7.2f} ± {eval_results['std_reward']:5.2f}, "
                f"Success={eval_results['success_rate']:5.1f}%, "
                f"Hops={eval_results['avg_hops']:4.1f}, "
                f"ε={agent.epsilon:.3f}, "
                f"LR={agent.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Check for improvement (track both reward and success rate)
            improved = False
            if eval_results['avg_reward'] > best_eval_reward:
                best_eval_reward = eval_results['avg_reward']
                improved = True
            if eval_results['success_rate'] > best_success_rate:
                best_success_rate = eval_results['success_rate']
                improved = True

            if improved:
                best_eval_episode = episode
                episodes_without_improvement = 0
                agent.save(f"{log_dir}/checkpoints/best_model.pt")
                log_message(log_file, f"  ★ New best! Reward={best_eval_reward:.2f}, Success={best_success_rate:.1f}%")
            else:
                episodes_without_improvement += eval_frequency

            # Early stopping check (only after warmup)
            if episode > warmup_episodes and episodes_without_improvement >= patience:
                log_message(log_file, f"\nEarly stopping at episode {episode}")
                break

        # Periodic checkpoint
        if episode % 500 == 0:
            agent.save(f"{log_dir}/checkpoints/checkpoint_{episode}.pt")

    # Save final model
    agent.save(f"{log_dir}/checkpoints/final_model.pt")

    log_message(log_file, "=" * 60)
    log_message(log_file, "Training Complete!")
    log_message(log_file, f"Best Reward: {best_eval_reward:.2f} at episode {best_eval_episode}")
    log_message(log_file, f"Best Success Rate: {best_success_rate:.1f}%")
    log_message(log_file, "=" * 60)

    # Final evaluation with best model
    log_message(log_file, "\nFinal Evaluation (100 episodes):")
    agent.load(f"{log_dir}/checkpoints/best_model.pt")
    agent.epsilon = 0.0
    final_results = evaluate_agent(agent, env, 100, max_steps)
    log_message(log_file, f"  Success Rate: {final_results['success_rate']:.1f}%")
    log_message(log_file, f"  Avg Reward: {final_results['avg_reward']:.2f} ± {final_results['std_reward']:.2f}")
    log_message(log_file, f"  Avg Hops: {final_results['avg_hops']:.1f}")
    log_message(log_file, f"  Avg Delay: {final_results['avg_delay']:.4f}")

    # Save config used
    with open(f"{log_dir}/config_used.yaml", 'w') as f:
        yaml.dump(config, f)

    return {
        'best_reward': best_eval_reward,
        'best_success_rate': best_success_rate,
        'best_episode': best_eval_episode,
        'final_success_rate': final_results['success_rate'],
        'final_avg_reward': final_results['avg_reward'],
        'final_avg_hops': final_results['avg_hops'],
        'log_dir': log_dir
    }


def main():
    parser = argparse.ArgumentParser(description='Train Per-Neighbor Q-Value Agent (V3)')
    parser.add_argument('--config', type=str, default='configs/routing_config.yaml',
                        help='Path to config file')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Number of training episodes')
    parser.add_argument('--no-transformer', action='store_true',
                        help='Disable Transformer for temporal context')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Run training
    results = train(
        config_path=args.config,
        num_episodes=args.episodes,
        use_transformer=not args.no_transformer,
        experiment_name=args.name
    )

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best reward: {results['best_reward']:.2f}")
    print(f"Best success rate: {results['best_success_rate']:.1f}%")
    print(f"Final success rate: {results['final_success_rate']:.1f}%")
    print(f"Final avg hops: {results['final_avg_hops']:.1f}")
    print(f"Logs saved to: {results['log_dir']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
