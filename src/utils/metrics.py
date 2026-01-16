"""
Metrics tracking utilities for training and evaluation.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional


class MetricsTracker:
    """
    Tracks and computes training and evaluation metrics.
    
    Maintains rolling windows for smooth metric reporting.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Episode metrics (rolling windows)
        self.rewards = deque(maxlen=window_size)
        self.lengths = deque(maxlen=window_size)
        self.successes = deque(maxlen=window_size)
        self.delays = deque(maxlen=window_size)
        self.hops = deque(maxlen=window_size)
        self.optimal_hops = deque(maxlen=window_size)
        self.drops = deque(maxlen=window_size)
        
        # Training metrics
        self.losses = deque(maxlen=window_size)
        self.q_values = deque(maxlen=window_size)
        
        # Statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.best_delivery_rate = 0.0
    
    def update_episode(self,
                       reward: float,
                       length: int,
                       success: bool,
                       delay: float,
                       hops: int,
                       optimal_hops: int = None,
                       packet_dropped: bool = False):
        """Update metrics after an episode."""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.successes.append(float(success))
        self.delays.append(delay)
        self.hops.append(hops)
        self.drops.append(float(packet_dropped))
        
        if optimal_hops is not None:
            self.optimal_hops.append(optimal_hops)
        
        self.total_episodes += 1
        self.total_steps += length
        
        # Update best metrics
        avg_reward = np.mean(self.rewards)
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
        
        delivery_rate = np.mean(self.successes)
        if delivery_rate > self.best_delivery_rate:
            self.best_delivery_rate = delivery_rate
    
    def update_training(self, loss: float, q_value: float):
        """Update training metrics."""
        self.losses.append(loss)
        self.q_values.append(q_value)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'std_reward': np.std(self.rewards) if self.rewards else 0.0,
            'min_reward': np.min(self.rewards) if self.rewards else 0.0,
            'max_reward': np.max(self.rewards) if self.rewards else 0.0,
            'delivery_rate': np.mean(self.successes) if self.successes else 0.0,
            'drop_rate': np.mean(self.drops) if self.drops else 0.0,
            'avg_delay': np.mean(self.delays) if self.delays else 0.0,
            'avg_hops': np.mean(self.hops) if self.hops else 0.0,
            'avg_length': np.mean(self.lengths) if self.lengths else 0.0,
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'avg_q_value': np.mean(self.q_values) if self.q_values else 0.0,
        }
        
        # Path optimality
        if self.optimal_hops and self.hops:
            valid_pairs = [
                (h, o) for h, o, s in zip(self.hops, self.optimal_hops, self.successes)
                if s > 0 and o > 0
            ]
            if valid_pairs:
                optimality = np.mean([o / h for h, o in valid_pairs])
                metrics['path_optimality'] = optimality
            else:
                metrics['path_optimality'] = 0.0
        else:
            metrics['path_optimality'] = 0.0
        
        return metrics
    
    def format_summary(self, episode: int, epsilon: float = None) -> str:
        """Format metrics summary for logging."""
        metrics = self.get_metrics()
        
        parts = [
            f"Episode {episode:5d}",
            f"Reward: {metrics['avg_reward']:8.2f}",
            f"Delivery: {metrics['delivery_rate']*100:5.1f}%",
            f"Delay: {metrics['avg_delay']:6.2f}ms",
            f"Hops: {metrics['avg_hops']:4.1f}",
            f"Loss: {metrics['avg_loss']:.4f}",
        ]
        
        if epsilon is not None:
            parts.append(f"Epsilon: {epsilon:.3f}")
        
        return " | ".join(parts)
    
    def format_detailed_summary(self) -> str:
        """Format detailed summary for end of training."""
        metrics = self.get_metrics()
        
        lines = [
            "=" * 60,
            "Metrics Summary",
            "=" * 60,
            "",
            "Reward Metrics:",
            f"  Average Reward: {metrics['avg_reward']:.4f} (+/- {metrics['std_reward']:.4f})",
            f"  Best Reward: {self.best_reward:.4f}",
            f"  Range: [{metrics['min_reward']:.4f}, {metrics['max_reward']:.4f}]",
            "",
            "Routing Performance:",
            f"  Delivery Rate: {metrics['delivery_rate']*100:.2f}%",
            f"  Drop Rate: {metrics['drop_rate']*100:.2f}%",
            f"  Average Delay: {metrics['avg_delay']:.4f} ms",
            f"  Average Hops: {metrics['avg_hops']:.2f}",
            f"  Path Optimality: {metrics['path_optimality']*100:.2f}%",
            "",
            "Training Metrics:",
            f"  Average Loss: {metrics['avg_loss']:.6f}",
            f"  Average Q-Value: {metrics['avg_q_value']:.4f}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics."""
        self.rewards.clear()
        self.lengths.clear()
        self.successes.clear()
        self.delays.clear()
        self.hops.clear()
        self.optimal_hops.clear()
        self.drops.clear()
        self.losses.clear()
        self.q_values.clear()


class EvaluationMetrics:
    """
    Metrics specifically for evaluation runs.
    """
    
    def __init__(self):
        self.rewards = []
        self.successes = []
        self.delays = []
        self.hops = []
        self.paths = []
    
    def add_episode(self,
                    reward: float,
                    success: bool,
                    delay: float,
                    hops: int,
                    path: List[int] = None):
        """Add metrics from an evaluation episode."""
        self.rewards.append(reward)
        self.successes.append(float(success))
        self.delays.append(delay)
        self.hops.append(hops)
        if path is not None:
            self.paths.append(path)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'std_reward': np.std(self.rewards) if self.rewards else 0.0,
            'success_rate': np.mean(self.successes) * 100 if self.successes else 0.0,
            'avg_delay': np.mean(self.delays) if self.delays else 0.0,
            'avg_hops': np.mean(self.hops) if self.hops else 0.0,
        }
    
    def reset(self):
        """Reset all metrics."""
        self.rewards.clear()
        self.successes.clear()
        self.delays.clear()
        self.hops.clear()
        self.paths.clear()
