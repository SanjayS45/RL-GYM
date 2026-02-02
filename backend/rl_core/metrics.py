"""Training metrics collection and computation for RL algorithms."""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int
    total_reward: float
    episode_length: int
    success: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration(self) -> float:
        """Duration of the episode in seconds."""
        return self.end_time - self.start_time


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    loss: Optional[float] = None
    q_value: Optional[float] = None
    entropy: Optional[float] = None
    kl_divergence: Optional[float] = None
    gradient_norm: Optional[float] = None
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None
    alpha: Optional[float] = None


class MetricsCollector:
    """Collects and computes training metrics."""
    
    def __init__(
        self,
        window_size: int = 100,
        compute_moving_average: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Window size for moving averages
            compute_moving_average: Whether to compute moving averages
        """
        self.window_size = window_size
        self.compute_moving_average = compute_moving_average
        
        # Episode metrics
        self.episode_rewards: deque = deque(maxlen=window_size)
        self.episode_lengths: deque = deque(maxlen=window_size)
        self.episode_successes: deque = deque(maxlen=window_size)
        self.episode_durations: deque = deque(maxlen=window_size)
        
        # Step metrics
        self.losses: deque = deque(maxlen=window_size)
        self.q_values: deque = deque(maxlen=window_size)
        self.entropies: deque = deque(maxlen=window_size)
        self.gradient_norms: deque = deque(maxlen=window_size)
        
        # Counters
        self.total_episodes = 0
        self.total_steps = 0
        self.total_updates = 0
        
        # Timing
        self.start_time = time.time()
        self._episode_start_time: Optional[float] = None
        
        # Full history (for logging)
        self.history: List[Dict[str, Any]] = []
    
    def start_episode(self) -> None:
        """Mark the start of a new episode."""
        self._episode_start_time = time.time()
    
    def end_episode(
        self,
        reward: float,
        length: int,
        success: bool = False,
        info: Optional[Dict[str, Any]] = None
    ) -> EpisodeMetrics:
        """
        Record end of episode and compute metrics.
        
        Args:
            reward: Total episode reward
            length: Episode length (steps)
            success: Whether episode was successful
            info: Additional episode info
            
        Returns:
            EpisodeMetrics object
        """
        end_time = time.time()
        start_time = self._episode_start_time or end_time
        
        # Create metrics object
        metrics = EpisodeMetrics(
            episode_id=self.total_episodes,
            total_reward=reward,
            episode_length=length,
            success=success,
            info=info or {},
            start_time=start_time,
            end_time=end_time
        )
        
        # Update running statistics
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(1 if success else 0)
        self.episode_durations.append(metrics.duration)
        
        self.total_episodes += 1
        self.total_steps += length
        
        # Reset episode timer
        self._episode_start_time = None
        
        return metrics
    
    def record_step(self, metrics: StepMetrics) -> None:
        """
        Record metrics for a training step.
        
        Args:
            metrics: Step metrics object
        """
        if metrics.loss is not None:
            self.losses.append(metrics.loss)
        if metrics.q_value is not None:
            self.q_values.append(metrics.q_value)
        if metrics.entropy is not None:
            self.entropies.append(metrics.entropy)
        if metrics.gradient_norm is not None:
            self.gradient_norms.append(metrics.gradient_norm)
        
        self.total_updates += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
            'elapsed_time': time.time() - self.start_time,
        }
        
        # Episode statistics
        if self.episode_rewards:
            rewards = list(self.episode_rewards)
            stats.update({
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
            })
        
        if self.episode_lengths:
            lengths = list(self.episode_lengths)
            stats.update({
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
            })
        
        if self.episode_successes:
            stats['success_rate'] = np.mean(list(self.episode_successes))
        
        # Training statistics
        if self.losses:
            stats['mean_loss'] = np.mean(list(self.losses))
        
        if self.q_values:
            stats['mean_q_value'] = np.mean(list(self.q_values))
        
        if self.entropies:
            stats['mean_entropy'] = np.mean(list(self.entropies))
        
        if self.gradient_norms:
            stats['mean_gradient_norm'] = np.mean(list(self.gradient_norms))
        
        # FPS calculation
        elapsed = stats['elapsed_time']
        if elapsed > 0:
            stats['fps'] = self.total_steps / elapsed
        
        return stats
    
    def log_to_history(self, step: int, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log current statistics to history.
        
        Args:
            step: Current training step
            extra: Additional data to log
        """
        entry = {
            'step': step,
            'timestamp': time.time(),
            **self.get_statistics(),
        }
        
        if extra:
            entry.update(extra)
        
        self.history.append(entry)
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_successes.clear()
        self.episode_durations.clear()
        self.losses.clear()
        self.q_values.clear()
        self.entropies.clear()
        self.gradient_norms.clear()
        self.total_episodes = 0
        self.total_steps = 0
        self.total_updates = 0
        self.start_time = time.time()
        self.history.clear()


class RewardTracker:
    """Tracks rewards with exponential moving average."""
    
    def __init__(self, alpha: float = 0.01):
        """
        Initialize reward tracker.
        
        Args:
            alpha: EMA smoothing factor
        """
        self.alpha = alpha
        self.ema: Optional[float] = None
        self.total: float = 0.0
        self.count: int = 0
    
    def update(self, reward: float) -> None:
        """Update tracker with new reward."""
        self.total += reward
        self.count += 1
        
        if self.ema is None:
            self.ema = reward
        else:
            self.ema = self.alpha * reward + (1 - self.alpha) * self.ema
    
    @property
    def mean(self) -> float:
        """Get mean reward."""
        return self.total / max(1, self.count)
    
    @property
    def smoothed(self) -> float:
        """Get smoothed (EMA) reward."""
        return self.ema or 0.0
    
    def reset(self) -> None:
        """Reset tracker."""
        self.ema = None
        self.total = 0.0
        self.count = 0


def compute_discounted_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    last_value: float = 0.0
) -> np.ndarray:
    """
    Compute discounted returns.
    
    Args:
        rewards: Array of rewards
        dones: Array of done flags
        gamma: Discount factor
        last_value: Value estimate for final state
        
    Returns:
        Array of discounted returns
    """
    returns = np.zeros_like(rewards)
    running_return = last_value
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return * (1 - dones[t])
        returns[t] = running_return
    
    return returns


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    last_value: float = 0.0
) -> tuple:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        last_value: Value estimate for final state
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    
    return advantages, returns

