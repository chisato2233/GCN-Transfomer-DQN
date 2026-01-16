"""
Main training script for SAGIN intelligent routing.

Trains DQN agent with GCN-Transformer architecture.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
from typing import Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn import DQNAgent
from src.utils.logger import Logger
from src.utils.metrics import MetricsTracker, ConvergenceChecker


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_agent_config(config: dict, env: SAGINRoutingEnv) -> dict:
    """Create agent configuration from main config."""
    return {
        # Dimensions
        'state_dim': env.state_dim,
        'num_actions': env.max_neighbors,
        'max_neighbors': env.max_neighbors,
        'history_length': config.get('transformer', {}).get('history_length', 10),

        # Device
        'device': config.get('hardware', {}).get('device', 'cuda'),

        # Training
        'gamma': config.get('training', {}).get('gamma', 0.99),
        'lr': config.get('dqn', {}).get('lr', 1e-4),
        'batch_size': config.get('buffer', {}).get('batch_size', 64),
        'target_update_freq': config.get('dqn', {}).get('target_update_freq', 100),
        'max_grad_norm': config.get('dqn', {}).get('max_grad_norm', 1.0),

        # Algorithm variants
        'use_double_dqn': config.get('dqn', {}).get('use_double_dqn', True),
        'use_dueling': config.get('dqn', {}).get('use_dueling', True),
        'use_per': config.get('buffer', {}).get('use_per', False),

        # Buffer
        'buffer_capacity': config.get('buffer', {}).get('capacity', 100000),
        'per_alpha': config.get('buffer', {}).get('per_alpha', 0.6),
        'per_beta_start': config.get('buffer', {}).get('per_beta_start', 0.4),
        'per_beta_frames': config.get('buffer', {}).get('per_beta_frames', 100000),

        # Exploration
        'epsilon_start': config.get('exploration', {}).get('epsilon_start', 1.0),
        'epsilon_end': config.get('exploration', {}).get('epsilon_end', 0.01),
        'epsilon_decay': config.get('exploration', {}).get('epsilon_decay', 0.995),

        # Model components
        'use_gcn': config.get('use_gcn', True),
        'use_transformer': config.get('use_transformer', True),
        'gcn': config.get('gcn', {}),
        'transformer': config.get('transformer', {}),
        'fusion': config.get('fusion', {}),
        'q_hidden_dims': config.get('dqn', {}).get('hidden_dims', [256, 128, 64])
    }


def train(config: dict):
    """Main training function."""
    # Set seed
    seed = config.get('seed', 42)
    deterministic = config.get('deterministic', False)
    set_seed(seed, deterministic)

    # Initialize logger
    logger = Logger(
        log_dir=config.get('logging', {}).get('log_dir', 'logs'),
        experiment_name='sagin_routing',
        level=config.get('logging', {}).get('level', 'INFO'),
        use_tensorboard=config.get('logging', {}).get('tensorboard', True)
    )

    logger.save_config(config)
    logger.info("=" * 60)
    logger.info("SAGIN Intelligent Routing Training")
    logger.info("=" * 60)

    # Initialize environment
    env = SAGINRoutingEnv(config)
    logger.info(f"Environment created:")
    logger.info(f"  - Total nodes: {env.topology.total_nodes}")
    logger.info(f"  - Satellites: {env.topology.num_satellites}")
    logger.info(f"  - UAVs: {env.topology.num_uavs}")
    logger.info(f"  - Ground nodes: {env.topology.num_ground}")
    logger.info(f"  - State dim: {env.state_dim}")
    logger.info(f"  - Action space: {env.max_neighbors}")

    # Initialize agent
    agent_config = create_agent_config(config, env)
    agent = DQNAgent(agent_config)

    logger.info(f"\nAgent created:")
    logger.info(f"  - Device: {agent.device}")
    logger.info(f"  - Use GCN: {agent_config['use_gcn']}")
    logger.info(f"  - Use Transformer: {agent_config['use_transformer']}")
    logger.info(f"  - Use Double DQN: {agent_config['use_double_dqn']}")
    logger.info(f"  - Use PER: {agent_config['use_per']}")

    # Save model info
    logger.save_model_info(agent.q_network)

    # Initialize metrics
    metrics = MetricsTracker(window_size=100)
    convergence_checker = ConvergenceChecker(
        patience=config.get('training', {}).get('patience', 500),
        min_delta=config.get('training', {}).get('min_improvement', 0.01)
    )

    # Training parameters
    training_config = config.get('training', {})
    num_episodes = training_config.get('num_episodes', 5000)
    max_steps = training_config.get('max_steps_per_episode', 50)
    warmup_episodes = training_config.get('warmup_episodes', 50)
    eval_frequency = training_config.get('eval_frequency', 100)
    save_frequency = training_config.get('save_frequency', 500)
    log_frequency = config.get('logging', {}).get('log_frequency', 10)

    best_reward = float('-inf')
    best_delivery_rate = 0.0

    logger.info(f"\nStarting training for {num_episodes} episodes...")
    logger.info("=" * 60)

    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training", ncols=80, leave=True):
        # Reset environment
        obs, info = env.reset()
        state_history = env.get_state_history()

        episode_reward = 0.0
        episode_loss = 0.0
        episode_q_value = 0.0
        train_count = 0

        for step in range(max_steps):
            # Get local graph data for GCN
            local_data = env.get_local_graph_data()

            # Select action
            action = agent.select_action(
                state=obs['state'],
                action_mask=obs['action_mask'],
                state_history=state_history,
                current_node_features=local_data['current_node_features'],
                neighbor_features=local_data['neighbor_features'],
                neighbor_mask=local_data['neighbor_mask'],
                training=True
            )

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Get next state history
            next_state_history = env.get_state_history()

            # Store transition
            agent.store_transition(
                state=obs['state'],
                action=action,
                reward=reward,
                next_state=next_obs['state'],
                done=done,
                action_mask=obs['action_mask'],
                next_action_mask=next_obs['action_mask']
            )

            # Train after warmup
            if episode > warmup_episodes:
                train_info = agent.train()
                episode_loss += train_info['loss']
                episode_q_value += train_info['q_value']
                train_count += 1

            episode_reward += reward
            obs = next_obs
            state_history = next_state_history

            if done:
                break

        # End of episode
        agent.end_episode()

        # Update metrics
        success = info.get('success', False)
        metrics.update_episode(
            reward=episode_reward,
            length=step + 1,
            success=success,
            delay=info.get('total_delay', 0),
            hops=info.get('hop_count', 0),
            optimal_hops=info.get('optimal_hops'),
            packet_dropped=info.get('packet_dropped', False)
        )

        if train_count > 0:
            metrics.update_training(
                loss=episode_loss / train_count,
                q_value=episode_q_value / train_count
            )

        metrics.record_history(episode, agent.epsilon)

        # Logging
        if episode % log_frequency == 0:
            summary = metrics.get_summary()
            # Use tqdm.write to avoid conflict with progress bar
            tqdm.write(metrics.format_summary(episode, agent.epsilon))

            # TensorBoard logging
            logger.log_scalar('Train/Reward', summary['avg_reward'], episode)
            logger.log_scalar('Train/DeliveryRate', summary['delivery_rate'], episode)
            logger.log_scalar('Train/AvgDelay', summary['avg_delay'], episode)
            logger.log_scalar('Train/AvgHops', summary['avg_hops'], episode)
            logger.log_scalar('Train/PathOptimality', summary['path_optimality'], episode)
            logger.log_scalar('Train/Loss', summary['avg_loss'], episode)
            logger.log_scalar('Train/QValue', summary['avg_q_value'], episode)
            logger.log_scalar('Train/Epsilon', agent.epsilon, episode)
            logger.log_scalar('Train/BufferSize', len(agent.replay_buffer), episode)

        # Evaluation
        if episode % eval_frequency == 0:
            eval_metrics = evaluate(env, agent, num_episodes=10, max_steps=max_steps)

            logger.log_scalar('Eval/Reward', eval_metrics['avg_reward'], episode)
            logger.log_scalar('Eval/DeliveryRate', eval_metrics['delivery_rate'], episode)
            logger.log_scalar('Eval/AvgDelay', eval_metrics['avg_delay'], episode)
            logger.log_scalar('Eval/AvgHops', eval_metrics['avg_hops'], episode)

            logger.info(
                f"  Eval: Reward={eval_metrics['avg_reward']:.2f}, "
                f"Delivery={eval_metrics['delivery_rate']*100:.1f}%, "
                f"Delay={eval_metrics['avg_delay']:.2f}ms, "
                f"Hops={eval_metrics['avg_hops']:.1f}"
            )

            # Save best model
            if eval_metrics['avg_reward'] > best_reward:
                best_reward = eval_metrics['avg_reward']
                agent.save(str(logger.checkpoint_dir / 'best_model.pt'))
                logger.info(f"  New best model saved (reward: {best_reward:.2f})")

            if eval_metrics['delivery_rate'] > best_delivery_rate:
                best_delivery_rate = eval_metrics['delivery_rate']

            # Check convergence
            converged, episodes_without_improvement = convergence_checker.check(
                eval_metrics['avg_reward']
            )
            if converged:
                logger.info(f"\nTraining converged after {episode} episodes")
                break

        # Periodic checkpoint
        if episode % save_frequency == 0:
            agent.save(str(logger.checkpoint_dir / f'checkpoint_ep{episode}.pt'))
            metrics.save(logger.metrics_dir)
            logger.info(f"  Checkpoint saved at episode {episode}")

    # Training complete
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Reward: {best_reward:.2f}")
    logger.info(f"Best Delivery Rate: {best_delivery_rate*100:.1f}%")
    logger.info(metrics.format_detailed_summary())

    # Save final model and metrics
    agent.save(str(logger.checkpoint_dir / 'final_model.pt'))
    metrics.save(logger.metrics_dir)

    # Log final hyperparameters
    logger.log_hparams(
        hparam_dict={
            'lr': agent_config['lr'],
            'gamma': agent_config['gamma'],
            'batch_size': agent_config['batch_size'],
            'use_gcn': agent_config['use_gcn'],
            'use_transformer': agent_config['use_transformer'],
            'use_double_dqn': agent_config['use_double_dqn'],
            'use_per': agent_config['use_per']
        },
        metric_dict={
            'best_reward': best_reward,
            'best_delivery_rate': best_delivery_rate
        }
    )

    logger.close()

    return best_reward, best_delivery_rate


def evaluate(env: SAGINRoutingEnv,
             agent: DQNAgent,
             num_episodes: int = 10,
             max_steps: int = 50) -> Dict:
    """
    Evaluate agent performance.

    Args:
        env: Environment
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Dictionary of evaluation metrics
    """
    total_reward = 0.0
    total_delay = 0.0
    total_hops = 0.0
    successes = 0
    episodes_with_success = 0

    for _ in range(num_episodes):
        obs, info = env.reset()
        state_history = env.get_state_history()
        episode_reward = 0.0

        for step in range(max_steps):
            local_data = env.get_local_graph_data()

            action = agent.select_action(
                state=obs['state'],
                action_mask=obs['action_mask'],
                state_history=state_history,
                current_node_features=local_data['current_node_features'],
                neighbor_features=local_data['neighbor_features'],
                neighbor_mask=local_data['neighbor_mask'],
                training=False  # No exploration during evaluation
            )

            obs, reward, terminated, truncated, info = env.step(action)
            state_history = env.get_state_history()
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward += episode_reward

        if info.get('success', False):
            successes += 1
            total_delay += info.get('total_delay', 0)
            total_hops += info.get('hop_count', 0)
            episodes_with_success += 1

    return {
        'avg_reward': total_reward / num_episodes,
        'delivery_rate': successes / num_episodes,
        'avg_delay': total_delay / max(episodes_with_success, 1),
        'avg_hops': total_hops / max(episodes_with_success, 1)
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train SAGIN Intelligent Routing Agent'
    )
    parser.add_argument(
        '--config', type=str,
        default='configs/routing_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device (cuda/cpu, overrides config)'
    )
    parser.add_argument(
        '--episodes', type=int, default=None,
        help='Number of training episodes (overrides config)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command line overrides
    if args.seed is not None:
        config['seed'] = args.seed

    if args.device is not None:
        if 'hardware' not in config:
            config['hardware'] = {}
        config['hardware']['device'] = args.device

    if args.episodes is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['num_episodes'] = args.episodes

    # Run training
    train(config)


if __name__ == '__main__':
    main()
