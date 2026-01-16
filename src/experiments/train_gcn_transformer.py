"""
Training script for DQN with GCN + Transformer integration.

This script trains the intelligent routing agent using:
- GCN for spatial graph topology features
- Transformer for temporal state sequence patterns
- Gated feature fusion
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.sagin_env import SAGINRoutingEnv
from src.agents.dqn_gcn_transformer import DQNGCNTransformerAgent
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsTracker


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_graph_features(env: SAGINRoutingEnv) -> Dict[str, np.ndarray]:
    """
    Extract graph features from environment for GCN input.

    Returns:
        Dictionary with current_node_features, neighbor_features, neighbor_mask
    """
    local_graph = env.get_local_graph_data()

    return {
        'current_node_features': local_graph['current_node_features'],
        'neighbor_features': local_graph['neighbor_features'],
        'neighbor_mask': local_graph['neighbor_mask']
    }


def train(config: dict, log_dir: str):
    """Main training loop with GCN + Transformer."""
    # Setup
    logger = setup_logger("train_gcn_transformer", log_dir)
    metrics = MetricsTracker(window_size=100)

    logger.info("=" * 60)
    logger.info("SAGIN Routing Training with GCN + Transformer")
    logger.info("=" * 60)

    # Create environment
    env = SAGINRoutingEnv(config)
    logger.info(f"Environment created: {env.topology.total_nodes} nodes")
    logger.info(f"State dimension: {env.state_dim}")
    logger.info(f"Action space: {env.action_space.n}")

    # Build agent config from all config sections
    agent_config = {}

    # Environment dimensions
    agent_config['state_dim'] = env.state_dim
    agent_config['num_actions'] = env.action_space.n
    agent_config['node_feature_dim'] = env.node_feature_dim
    agent_config['max_neighbors'] = env.max_neighbors
    agent_config['history_length'] = env.history_length

    # DQN parameters
    dqn_config = config.get('dqn', {})
    agent_config['use_dueling'] = dqn_config.get('use_dueling', True)
    agent_config['use_double_dqn'] = dqn_config.get('use_double_dqn', True)
    agent_config['lr'] = dqn_config.get('lr', 3e-4)
    agent_config['target_update_freq'] = dqn_config.get('target_update_freq', 100)
    agent_config['max_grad_norm'] = dqn_config.get('max_grad_norm', 1.0)

    # Buffer parameters
    buffer_config = config.get('buffer', {})
    agent_config['buffer_capacity'] = buffer_config.get('capacity', 100000)
    agent_config['batch_size'] = buffer_config.get('batch_size', 64)

    # Exploration parameters
    exploration_config = config.get('exploration', {})
    agent_config['epsilon_start'] = exploration_config.get('epsilon_start', 1.0)
    agent_config['epsilon_end'] = exploration_config.get('epsilon_end', 0.01)
    agent_config['epsilon_decay'] = exploration_config.get('epsilon_decay', 0.995)

    # Training parameters
    training_config_agent = config.get('training', {})
    agent_config['gamma'] = training_config_agent.get('gamma', 0.99)

    # GCN parameters
    gcn_config = config.get('gcn', {})
    agent_config['gcn_output_dim'] = gcn_config.get('output_dim', 64)

    # Transformer parameters
    transformer_config = config.get('transformer', {})
    agent_config['transformer_output_dim'] = transformer_config.get('d_model', 64)

    # Fusion parameters
    fusion_config = config.get('fusion', {})
    agent_config['fused_dim'] = fusion_config.get('output_dim', 128)
    agent_config['state_encoded_dim'] = 64

    # PER parameters
    buffer_config = config.get('buffer', {})
    agent_config['use_per'] = buffer_config.get('use_per', False)
    agent_config['per_alpha'] = buffer_config.get('per_alpha', 0.6)
    agent_config['per_beta_start'] = buffer_config.get('per_beta_start', 0.4)
    agent_config['per_beta_frames'] = buffer_config.get('per_beta_frames', 100000)

    # Hardware
    hardware_config = config.get('hardware', {})
    agent_config['device'] = hardware_config.get('device', 'cuda')

    # Create agent
    agent = DQNGCNTransformerAgent(agent_config)
    logger.info(f"Agent created with GCN + Transformer")
    logger.info(f"  - GCN output dim: {agent_config.get('gcn_output_dim', 64)}")
    logger.info(f"  - Transformer output dim: {agent_config.get('transformer_output_dim', 64)}")
    logger.info(f"  - Fused dim: {agent_config.get('fused_dim', 128)}")
    logger.info(f"  - Device: {agent.device}")

    # Training parameters
    training_config = config.get('training', {})
    num_episodes = training_config.get('num_episodes', 5000)
    max_steps = training_config.get('max_steps_per_episode', 100)
    eval_frequency = training_config.get('eval_frequency', 100)
    save_frequency = training_config.get('save_frequency', 500)
    log_frequency = training_config.get('log_frequency', 10)
    num_eval_episodes = training_config.get('num_eval_episodes', 10)

    # Checkpoint directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Best model tracking
    best_reward = float('-inf')
    best_delivery_rate = 0.0
    best_episode = 0
    no_improvement_count = 0
    patience = training_config.get('patience', 300)

    logger.info(f"Starting training for {num_episodes} episodes...")

    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training", ncols=100):
        obs, info = env.reset()
        state = obs['state']
        action_mask = obs['action_mask']

        # Get graph features and state history
        graph_features = get_graph_features(env)
        state_history = env.get_state_history()

        episode_reward = 0.0
        episode_loss = 0.0
        num_steps = 0

        for step in range(max_steps):
            # Select action with all features
            action = agent.select_action(
                state=state,
                state_history=state_history,
                current_node_features=graph_features['current_node_features'],
                neighbor_features=graph_features['neighbor_features'],
                neighbor_mask=graph_features['neighbor_mask'],
                action_mask=action_mask,
                training=True
            )

            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Get next state features
            next_state = next_obs['state']
            next_action_mask = next_obs['action_mask']
            next_graph_features = get_graph_features(env)
            next_state_history = env.get_state_history()

            # Store transition with all features
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

            # Train
            train_info = agent.train()
            episode_loss += train_info['loss']

            episode_reward += reward
            num_steps += 1

            if done:
                break

            # Update for next step
            state = next_state
            action_mask = next_action_mask
            graph_features = next_graph_features
            state_history = next_state_history

        # End episode
        agent.end_episode()

        # Record metrics
        avg_loss = episode_loss / max(num_steps, 1)
        metrics.update_episode(
            reward=episode_reward,
            length=num_steps,
            success=next_info.get('success', False),
            delay=next_info.get('total_delay', 0),
            hops=next_info.get('hop_count', 0),
            optimal_hops=next_info.get('optimal_hops', 1),
            packet_dropped=next_info.get('packet_dropped', False)
        )
        metrics.update_training(
            loss=avg_loss,
            q_value=train_info.get('q_value', 0)
        )

        # Logging
        if episode % log_frequency == 0:
            tqdm.write(metrics.format_summary(episode, agent.epsilon))

        # Evaluation
        if episode % eval_frequency == 0:
            eval_reward, eval_delivery = evaluate(env, agent, num_eval_episodes)

            tqdm.write(
                f"  Eval: Reward={eval_reward:.2f}, Delivery={eval_delivery*100:.1f}%"
            )

            # Save best model
            if eval_reward > best_reward:
                best_reward = eval_reward
                best_delivery_rate = eval_delivery
                best_episode = episode
                no_improvement_count = 0
                agent.save(os.path.join(checkpoint_dir, 'best_model.pt'))
                tqdm.write(f"  New best model saved (reward: {best_reward:.2f}, delivery: {eval_delivery*100:.1f}%)")
            else:
                no_improvement_count += eval_frequency
                if no_improvement_count >= patience:
                    tqdm.write(f"  Early stopping at episode {episode} (no improvement for {patience} episodes)")
                    tqdm.write(f"  Best model was at episode {best_episode} with reward {best_reward:.2f}")
                    break

        # Periodic checkpoint
        if episode % save_frequency == 0:
            agent.save(os.path.join(checkpoint_dir, f'checkpoint_{episode}.pt'))
            tqdm.write(f"  Checkpoint saved at episode {episode}")

    # Save final model
    agent.save(os.path.join(checkpoint_dir, 'final_model.pt'))

    # Final summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Reward: {best_reward:.2f}")
    logger.info(f"Best Delivery Rate: {best_delivery_rate*100:.1f}%")
    logger.info("=" * 60)

    # Print metrics summary
    print(metrics.format_detailed_summary())


def evaluate(env: SAGINRoutingEnv,
             agent: DQNGCNTransformerAgent,
             num_episodes: int) -> tuple:
    """
    Evaluate agent performance.

    Returns:
        Tuple of (average_reward, delivery_rate)
    """
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
                training=False  # No exploration during eval
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

    avg_reward = total_reward / num_episodes
    delivery_rate = num_success / num_episodes

    return avg_reward, delivery_rate


def main():
    parser = argparse.ArgumentParser(
        description='Train SAGIN Routing with GCN + Transformer'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/routing_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-dir', '-l',
        type=str,
        default=None,
        help='Log directory (default: auto-generated)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load config
    config = load_config(args.config)

    # Setup log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('logs', f'gcn_transformer_{timestamp}')
    else:
        log_dir = args.log_dir

    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    print(f"Log directory: {log_dir}")
    print(f"Config: {args.config}")

    # Train
    train(config, log_dir)


if __name__ == '__main__':
    main()
