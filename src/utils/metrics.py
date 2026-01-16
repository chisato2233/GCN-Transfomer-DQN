"""
Evaluation metrics for SAGIN routing training.
"""

import numpy as np
import json
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    reward: float = 0.0
    length: int = 0
    success: bool = False
    delay: float = 0.0
    hops: int = 0
    optimal_hops: int = -1
    packet_dropped: bool = False


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Size of the sliding window for moving averages
        """
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all metrics."""
        # Episode-level metrics (sliding window)
        self.episode_rewards = deque(maxlen=self.window_size)
        self.episode_lengths = deque(maxlen=self.window_size)
        self.episode_success = deque(maxlen=self.window_size)

        # Routing-specific metrics
        self.delivery_rates = deque(maxlen=self.window_size)
        self.avg_delays = deque(maxlen=self.window_size)
        self.avg_hops = deque(maxlen=self.window_size)
        self.path_optimality = deque(maxlen=self.window_size)
        self.drop_rates = deque(maxlen=self.window_size)

        # Training metrics
        self.losses = deque(maxlen=self.window_size)
        self.q_values = deque(maxlen=self.window_size)
        self.td_errors = deque(maxlen=self.window_size)

        # History for plotting
        self.history = {
            'episode': [],
            'reward': [],
            'avg_reward': [],
            'delivery_rate': [],
            'avg_delay': [],
            'avg_hops': [],
            'path_optimality': [],
            'loss': [],
            'q_value': [],
            'epsilon': []
        }

        # Best metrics
        self.best_reward = float('-inf')
        self.best_delivery_rate = 0.0
        self.best_episode = 0

    def update_episode(self,
                       reward: float,
                       length: int,
                       success: bool,
                       delay: float = 0.0,
                       hops: int = 0,
                       optimal_hops: Optional[int] = None,
                       packet_dropped: bool = False):
        """
        Update metrics after an episode.

        Args:
            reward: Total episode reward
            length: Episode length (steps)
            success: Whether packet reached destination
            delay: Total end-to-end delay (if successful)
            hops: Total hops (if successful)
            optimal_hops: Optimal path hops (for path optimality)
            packet_dropped: Whether packet was dropped
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_success.append(1.0 if success else 0.0)

        # Delivery and drop rates
        self.delivery_rates.append(1.0 if success else 0.0)
        self.drop_rates.append(1.0 if packet_dropped else 0.0)

        # Only record delay/hops for successful deliveries
        if success:
            self.avg_delays.append(delay)
            self.avg_hops.append(hops)

            # Path optimality (ratio of optimal to actual hops)
            if optimal_hops is not None and optimal_hops > 0 and hops > 0:
                self.path_optimality.append(optimal_hops / hops)

        # Update best metrics
        avg_reward = np.mean(self.episode_rewards)
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward

    def update_training(self,
                        loss: float,
                        q_value: float,
                        td_error: Optional[float] = None):
        """
        Update training metrics.

        Args:
            loss: Training loss
            q_value: Average Q-value
            td_error: TD error (optional)
        """
        self.losses.append(loss)
        self.q_values.append(q_value)
        if td_error is not None:
            self.td_errors.append(td_error)

    def get_summary(self) -> Dict[str, float]:
        """Get current metrics summary."""
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else 0.0

        def safe_std(arr):
            return float(np.std(arr)) if len(arr) > 0 else 0.0

        return {
            # Reward metrics
            'avg_reward': safe_mean(self.episode_rewards),
            'std_reward': safe_std(self.episode_rewards),
            'max_reward': float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
            'min_reward': float(min(self.episode_rewards)) if self.episode_rewards else 0.0,

            # Episode metrics
            'avg_length': safe_mean(self.episode_lengths),
            'success_rate': safe_mean(self.episode_success),

            # Routing metrics
            'delivery_rate': safe_mean(self.delivery_rates),
            'drop_rate': safe_mean(self.drop_rates),
            'avg_delay': safe_mean(self.avg_delays),
            'avg_hops': safe_mean(self.avg_hops),
            'path_optimality': safe_mean(self.path_optimality),

            # Training metrics
            'avg_loss': safe_mean(self.losses),
            'avg_q_value': safe_mean(self.q_values),
            'avg_td_error': safe_mean(self.td_errors),

            # Best metrics
            'best_reward': self.best_reward if self.best_reward != float('-inf') else 0.0
        }

    def record_history(self, episode: int, epsilon: float):
        """Record metrics to history for later plotting."""
        summary = self.get_summary()

        self.history['episode'].append(episode)
        self.history['reward'].append(
            float(self.episode_rewards[-1]) if self.episode_rewards else 0.0
        )
        self.history['avg_reward'].append(summary['avg_reward'])
        self.history['delivery_rate'].append(summary['delivery_rate'])
        self.history['avg_delay'].append(summary['avg_delay'])
        self.history['avg_hops'].append(summary['avg_hops'])
        self.history['path_optimality'].append(summary['path_optimality'])
        self.history['loss'].append(summary['avg_loss'])
        self.history['q_value'].append(summary['avg_q_value'])
        self.history['epsilon'].append(epsilon)

        # Update best episode
        if summary['avg_reward'] >= self.best_reward:
            self.best_episode = episode

    def save(self, path: Path):
        """Save metrics history to JSON file."""
        metrics_file = path / "metrics.json" if path.is_dir() else path

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                'history': self.history,
                'summary': self.get_summary(),
                'best_episode': self.best_episode
            }, f, indent=2)

    def load(self, path: Path):
        """Load metrics history from JSON file."""
        metrics_file = path / "metrics.json" if path.is_dir() else path

        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.history = data.get('history', self.history)
            self.best_episode = data.get('best_episode', 0)

    def format_summary(self, episode: int, epsilon: float) -> str:
        """Format metrics summary for logging."""
        s = self.get_summary()
        return (
            f"Episode {episode:5d} | "
            f"Reward: {s['avg_reward']:7.2f} | "
            f"Delivery: {s['delivery_rate']*100:5.1f}% | "
            f"Delay: {s['avg_delay']:6.2f}ms | "
            f"Hops: {s['avg_hops']:4.1f} | "
            f"Loss: {s['avg_loss']:.4f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    def format_detailed_summary(self) -> str:
        """Format detailed metrics summary."""
        s = self.get_summary()
        lines = [
            "=" * 60,
            "Metrics Summary",
            "=" * 60,
            "",
            "Reward Metrics:",
            f"  Average Reward: {s['avg_reward']:.4f} (+/- {s['std_reward']:.4f})",
            f"  Best Reward: {s['best_reward']:.4f}",
            f"  Range: [{s['min_reward']:.4f}, {s['max_reward']:.4f}]",
            "",
            "Routing Performance:",
            f"  Delivery Rate: {s['delivery_rate']*100:.2f}%",
            f"  Drop Rate: {s['drop_rate']*100:.2f}%",
            f"  Average Delay: {s['avg_delay']:.4f} ms",
            f"  Average Hops: {s['avg_hops']:.2f}",
            f"  Path Optimality: {s['path_optimality']*100:.2f}%",
            "",
            "Training Metrics:",
            f"  Average Loss: {s['avg_loss']:.6f}",
            f"  Average Q-Value: {s['avg_q_value']:.4f}",
            "",
            "=" * 60
        ]
        return "\n".join(lines)


class BaselineComparator:
    """Compare algorithm performance against baselines."""

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}

    def add_result(self, algorithm: str, metrics: Dict[str, float]):
        """Add results for an algorithm."""
        self.results[algorithm] = metrics.copy()

    def compare(self) -> Dict[str, Dict[str, float]]:
        """Compare all algorithms across metrics."""
        comparison = {}
        metrics_to_compare = [
            'delivery_rate', 'avg_delay', 'avg_hops',
            'path_optimality', 'avg_reward'
        ]

        for metric in metrics_to_compare:
            comparison[metric] = {
                alg: self.results[alg].get(metric, 0.0)
                for alg in self.results
            }

        return comparison

    def get_improvement(self,
                        our_alg: str,
                        baseline: str,
                        metric: str,
                        higher_is_better: bool = True) -> float:
        """
        Calculate relative improvement over baseline.

        Args:
            our_alg: Our algorithm name
            baseline: Baseline algorithm name
            metric: Metric to compare
            higher_is_better: Whether higher values are better

        Returns:
            Improvement percentage
        """
        if baseline not in self.results or our_alg not in self.results:
            return 0.0

        base_val = self.results[baseline].get(metric, 0.0)
        our_val = self.results[our_alg].get(metric, 0.0)

        if base_val == 0:
            return 0.0 if our_val == 0 else float('inf')

        if higher_is_better:
            return (our_val - base_val) / abs(base_val) * 100
        else:
            return (base_val - our_val) / abs(base_val) * 100

    def format_comparison(self) -> str:
        """Format comparison table."""
        if not self.results:
            return "No results to compare."

        algorithms = list(self.results.keys())
        metrics = ['delivery_rate', 'avg_delay', 'avg_hops', 'avg_reward']

        # Header
        header = f"{'Metric':<20}" + "".join(f"{alg:>15}" for alg in algorithms)
        separator = "-" * len(header)

        lines = [separator, header, separator]

        for metric in metrics:
            row = f"{metric:<20}"
            for alg in algorithms:
                val = self.results[alg].get(metric, 0.0)
                if metric == 'delivery_rate':
                    row += f"{val*100:>14.1f}%"
                else:
                    row += f"{val:>15.2f}"
            lines.append(row)

        lines.append(separator)
        return "\n".join(lines)

    def save(self, path: Path):
        """Save comparison results."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'comparison': self.compare()
            }, f, indent=2)


class ConvergenceChecker:
    """Check for training convergence."""

    def __init__(self,
                 patience: int = 100,
                 min_delta: float = 0.01,
                 window_size: int = 50):
        """
        Initialize convergence checker.

        Args:
            patience: Episodes without improvement before convergence
            min_delta: Minimum improvement to reset patience
            window_size: Window for smoothing metrics
        """
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size

        self.best_value = float('-inf')
        self.episodes_without_improvement = 0
        self.history = deque(maxlen=window_size)

    def check(self, value: float) -> Tuple[bool, int]:
        """
        Check if training has converged.

        Args:
            value: Current metric value

        Returns:
            Tuple of (converged, episodes_without_improvement)
        """
        self.history.append(value)
        smoothed_value = np.mean(self.history)

        if smoothed_value > self.best_value + self.min_delta:
            self.best_value = smoothed_value
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1

        converged = self.episodes_without_improvement >= self.patience

        return converged, self.episodes_without_improvement

    def reset(self):
        """Reset the checker."""
        self.best_value = float('-inf')
        self.episodes_without_improvement = 0
        self.history.clear()
