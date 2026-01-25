"""
Ablation Study Training Script (V3 Enhanced).

消融实验：验证V3各组件对模型性能的贡献

实验设计：
1. Full Model (V3): Per-Neighbor + Transformer + 14维特征 (完整模型)
2. w/o Transformer: 去掉时序建模，只用Per-Neighbor编码
3. w/o Three-Layer: 只用8维路由特征，去掉6维三层特征
4. w/o Per-Neighbor: 使用全局池化（丢失邻居-动作对应）
5. MLP Only: 简单MLP，无图结构感知

Usage:
    python src/experiments/train_ablation.py --ablation full --episodes 2000
    python src/experiments/train_ablation.py --ablation no_transformer --episodes 2000
    python src/experiments/train_ablation.py --ablation no_threelayer --episodes 2000
    python src/experiments/train_ablation.py --ablation no_perneighbor --episodes 2000
    python src/experiments/train_ablation.py --ablation mlp_only --episodes 2000
    python src/experiments/train_ablation.py --ablation all --episodes 2000
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime
from collections import deque
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.env.sagin_env import SAGINRoutingEnv


# ============================================================
# Ablation Model Variants
# ============================================================

class NeighborEncoder(nn.Module):
    """Shared neighbor encoder."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class FullModelV3(nn.Module):
    """Full V3: Per-Neighbor Q-Value + Transformer + 14-dim features"""

    def __init__(self, config: dict):
        super().__init__()
        self.topology_dim = config.get('topology_feature_dim', 14)
        self.history_dim = config.get('simplified_history_dim', 6)
        self.hidden_dim = config.get('neighbor_hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        self.neighbor_encoder = NeighborEncoder(self.topology_dim, self.hidden_dim)

        self.history_embed = nn.Linear(self.history_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_proj = nn.Linear(32, self.hidden_dim)

        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, neighbor_features, neighbor_mask, history):
        neighbor_hidden = self.neighbor_encoder(neighbor_features)

        history_embed = self.history_embed(history)
        temporal_out = self.transformer(history_embed)
        temporal_context = self.temporal_proj(temporal_out[:, -1, :])

        temporal_expanded = temporal_context.unsqueeze(1).expand(-1, self.max_neighbors, -1)
        combined = torch.cat([neighbor_hidden, temporal_expanded], dim=-1)
        q_values = self.q_head(combined).squeeze(-1)
        q_values = q_values.masked_fill(neighbor_mask == 0, float('-inf'))

        return q_values


class NoTransformerModel(nn.Module):
    """Ablation: w/o Transformer - only neighbor features"""

    def __init__(self, config: dict):
        super().__init__()
        self.topology_dim = config.get('topology_feature_dim', 14)
        self.hidden_dim = config.get('neighbor_hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        self.neighbor_encoder = NeighborEncoder(self.topology_dim, self.hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, neighbor_features, neighbor_mask, history):
        neighbor_hidden = self.neighbor_encoder(neighbor_features)
        q_values = self.q_head(neighbor_hidden).squeeze(-1)
        q_values = q_values.masked_fill(neighbor_mask == 0, float('-inf'))
        return q_values


class NoThreeLayerModel(nn.Module):
    """Ablation: w/o Three-Layer Features - only 8-dim routing features"""

    def __init__(self, config: dict):
        super().__init__()
        self.topology_dim = 8  # Only routing features
        self.history_dim = config.get('simplified_history_dim', 6)
        self.hidden_dim = config.get('neighbor_hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        self.neighbor_encoder = NeighborEncoder(self.topology_dim, self.hidden_dim)

        self.history_embed = nn.Linear(self.history_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_proj = nn.Linear(32, self.hidden_dim)

        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, neighbor_features, neighbor_mask, history):
        # Only use first 8 dimensions
        neighbor_features_8dim = neighbor_features[:, :, :8]
        neighbor_hidden = self.neighbor_encoder(neighbor_features_8dim)

        history_embed = self.history_embed(history)
        temporal_out = self.transformer(history_embed)
        temporal_context = self.temporal_proj(temporal_out[:, -1, :])

        temporal_expanded = temporal_context.unsqueeze(1).expand(-1, self.max_neighbors, -1)
        combined = torch.cat([neighbor_hidden, temporal_expanded], dim=-1)
        q_values = self.q_head(combined).squeeze(-1)
        q_values = q_values.masked_fill(neighbor_mask == 0, float('-inf'))
        return q_values


class NoPerNeighborModel(nn.Module):
    """Ablation: w/o Per-Neighbor - global pooling (loses correspondence)"""

    def __init__(self, config: dict):
        super().__init__()
        self.topology_dim = config.get('topology_feature_dim', 14)
        self.hidden_dim = config.get('neighbor_hidden_dim', 64)
        self.max_neighbors = config.get('max_neighbors', 8)

        self.neighbor_encoder = NeighborEncoder(self.topology_dim, self.hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.max_neighbors)
        )

    def forward(self, neighbor_features, neighbor_mask, history):
        neighbor_hidden = self.neighbor_encoder(neighbor_features)

        # Global mean pooling (loses correspondence!)
        mask_expanded = neighbor_mask.unsqueeze(-1)
        masked_hidden = neighbor_hidden * mask_expanded
        pooled = masked_hidden.sum(dim=1) / (neighbor_mask.sum(dim=1, keepdim=True) + 1e-6)

        q_values = self.q_head(pooled)
        q_values = q_values.masked_fill(neighbor_mask == 0, float('-inf'))
        return q_values


class MLPOnlyModel(nn.Module):
    """Ablation: MLP only - no graph structure"""

    def __init__(self, config: dict):
        super().__init__()
        self.topology_dim = config.get('topology_feature_dim', 14)
        self.hidden_dim = config.get('neighbor_hidden_dim', 64)

        self.mlp = nn.Sequential(
            nn.Linear(self.topology_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, neighbor_features, neighbor_mask, history):
        q_values = self.mlp(neighbor_features).squeeze(-1)
        q_values = q_values.masked_fill(neighbor_mask == 0, float('-inf'))
        return q_values


# ============================================================
# Ablation Agent
# ============================================================

class AblationAgent:
    """Agent for ablation study."""

    def __init__(self, config: dict, ablation_type: str):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.ablation_type = ablation_type
        self.max_neighbors = config.get('max_neighbors', 8)

        # Create model based on ablation type
        model_classes = {
            'full': FullModelV3,
            'no_transformer': NoTransformerModel,
            'no_threelayer': NoThreeLayerModel,
            'no_perneighbor': NoPerNeighborModel,
            'mlp_only': MLPOnlyModel,
        }

        if ablation_type not in model_classes:
            raise ValueError(f"Unknown ablation type: {ablation_type}")

        ModelClass = model_classes[ablation_type]
        self.q_network = ModelClass(config).to(self.device)
        self.target_network = ModelClass(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.get('lr', 1e-3))
        self.buffer = deque(maxlen=config.get('buffer_capacity', 10000))
        self.batch_size = config.get('batch_size', 64)

        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)

        # Training frequency control (FIX: don't train every step!)
        self.train_freq = config.get('train_freq', 4)  # Train every N steps
        self.target_update_freq = config.get('target_update_freq', 100)  # Update target every N trains
        self.step_count = 0
        self.train_count = 0

    def select_action(self, neighbor_features, neighbor_mask, history, action_mask, training=True):
        valid = np.where(action_mask > 0)[0]

        # [FIX] If no valid actions, return 0 (will be handled as invalid in env)
        if len(valid) == 0:
            return 0

        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid))

        with torch.no_grad():
            nf = torch.FloatTensor(neighbor_features).unsqueeze(0).to(self.device)
            nm = torch.FloatTensor(neighbor_mask).unsqueeze(0).to(self.device)
            h = torch.FloatTensor(history).unsqueeze(0).to(self.device)

            q_values = self.q_network(nf, nm, h).squeeze(0).cpu().numpy()
            q_values[action_mask == 0] = float('-inf')

            # [FIX] If all q_values are -inf, choose randomly from valid
            if np.all(np.isinf(q_values)):
                return int(np.random.choice(valid))

            return int(np.argmax(q_values))

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def maybe_train(self):
        """Train only every train_freq steps."""
        self.step_count += 1

        # Only train every N steps
        if self.step_count % self.train_freq != 0:
            return 0.0

        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        nf = torch.FloatTensor(np.array([s[0] for s in states])).to(self.device)
        nm = torch.FloatTensor(np.array([s[1] for s in states])).to(self.device)
        h = torch.FloatTensor(np.array([s[2] for s in states])).to(self.device)

        nf_next = torch.FloatTensor(np.array([s[0] for s in next_states])).to(self.device)
        nm_next = torch.FloatTensor(np.array([s[1] for s in next_states])).to(self.device)
        h_next = torch.FloatTensor(np.array([s[2] for s in next_states])).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(nf, nm, h)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(nf_next, nm_next, h_next)
            next_q_max = next_q.max(dim=1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q_max

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self._update_target()

        return loss.item()

    def _update_target(self):
        """Soft update target network."""
        for t, s in zip(self.target_network.parameters(), self.q_network.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'ablation_type': self.ablation_type,
        }, path)


# ============================================================
# Training Functions
# ============================================================

def train_ablation(ablation_type: str, config: dict, num_episodes: int, log_dir: str):
    """Train an ablation variant."""

    print(f"\n{'='*60}")
    print(f"ABLATION: {ablation_type.upper()}")
    print(f"{'='*60}")

    env = SAGINRoutingEnv(config)

    agent_config = {
        'topology_feature_dim': 14,
        'simplified_history_dim': 6,
        'neighbor_hidden_dim': 64,
        'max_neighbors': config['environment']['max_neighbors'],
        'history_length': config['transformer']['history_length'],
        'device': config['hardware']['device'],
        'lr': 1e-3,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
    }

    agent = AblationAgent(agent_config, ablation_type)

    best_success_rate = 0
    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)

    max_steps_per_episode = config.get('training', {}).get('max_steps_per_episode', 50)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0  # [FIX] Add step counter to prevent infinite loops

        while not done:
            topo_data = env.get_topology_aware_features()
            history = env.get_simplified_history()
            action_mask = obs['action_mask']

            action = agent.select_action(
                topo_data['neighbor_topology_features'],
                topo_data['neighbor_mask'],
                history,
                action_mask
            )

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # [FIX] Force termination after max steps to prevent infinite loops
            step_count += 1
            if step_count >= max_steps_per_episode:
                done = True
                truncated = True

            next_topo = env.get_topology_aware_features()
            next_history = env.get_simplified_history()

            state = (topo_data['neighbor_topology_features'],
                    topo_data['neighbor_mask'], history)
            next_state = (next_topo['neighbor_topology_features'],
                         next_topo['neighbor_mask'], next_history)

            agent.store(state, action, reward, next_state, float(done))
            agent.maybe_train()  # Only trains every N steps now

            episode_reward += reward
            obs = next_obs

        agent.decay_epsilon()

        recent_rewards.append(episode_reward)
        recent_successes.append(1.0 if next_info.get('success', False) else 0.0)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            success_rate = np.mean(recent_successes) * 100

            print(f"Ep {episode+1}/{num_episodes} | "
                  f"Reward: {avg_reward:.2f} | "
                  f"Success: {success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(f"{log_dir}/{ablation_type}_best.pt")

    agent.save(f"{log_dir}/{ablation_type}_final.pt")
    print(f"\n{ablation_type.upper()} Complete! Best: {best_success_rate:.1f}%")

    return best_success_rate


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Training')
    parser.add_argument('--ablation', type=str, required=True,
                        choices=['full', 'no_transformer', 'no_threelayer',
                                'no_perneighbor', 'mlp_only', 'all'],
                        help='Ablation type')
    parser.add_argument('--config', type=str, default='configs/routing_config.yaml')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/ablation_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("SAGIN Routing - Ablation Study")
    print(f"Log Dir: {log_dir}")
    print("=" * 60)

    results = {}

    if args.ablation == 'all':
        ablations = ['full', 'no_transformer', 'no_threelayer', 'no_perneighbor', 'mlp_only']
    else:
        ablations = [args.ablation]

    for ablation in ablations:
        success_rate = train_ablation(ablation, config, args.episodes, log_dir)
        results[ablation] = success_rate

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)
    print(f"{'Variant':<20} {'Success Rate':>15}")
    print("-" * 40)
    for variant, rate in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{variant:<20} {rate:>14.1f}%")
    print("=" * 60)

    with open(f"{log_dir}/results.yaml", 'w') as f:
        yaml.dump(results, f)


if __name__ == '__main__':
    main()
