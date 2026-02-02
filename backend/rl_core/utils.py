"""
Utility functions for RL training.
"""

import random
from typing import Dict, Any, Optional, Union
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance.
    
    ev = 1 - Var[y_true - y_pred] / Var[y_true]
    
    Args:
        y_pred: Predicted values
        y_true: True values
        
    Returns:
        Explained variance (1.0 = perfect prediction)
    """
    assert y_pred.shape == y_true.shape
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - np.var(y_true - y_pred) / var_y


def polyak_update(
    source: torch.nn.Module,
    target: torch.nn.Module,
    tau: float
) -> None:
    """
    Soft update of target network parameters.
    
    target = tau * source + (1 - tau) * target
    
    Args:
        source: Source network
        target: Target network
        tau: Interpolation parameter (0 < tau <= 1)
    """
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * source_param.data)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        dones: Array of done flags
        next_value: Value estimate for final state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: GAE advantages
        returns: Discounted returns (advantages + values)
    """
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0
    
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values
    return advantages, returns


def normalize_obs(
    obs: Union[np.ndarray, torch.Tensor],
    obs_mean: Union[np.ndarray, torch.Tensor],
    obs_std: Union[np.ndarray, torch.Tensor],
    clip: float = 10.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize observations using running statistics.
    
    Args:
        obs: Observations to normalize
        obs_mean: Running mean
        obs_std: Running standard deviation
        clip: Clipping value
        
    Returns:
        Normalized observations
    """
    if isinstance(obs, np.ndarray):
        return np.clip((obs - obs_mean) / (obs_std + 1e-8), -clip, clip)
    else:
        return torch.clamp((obs - obs_mean) / (obs_std + 1e-8), -clip, clip)


class RunningMeanStd:
    """
    Running mean and standard deviation calculator.
    Uses Welford's online algorithm for numerical stability.
    """
    
    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int
    ) -> None:
        """Update from precomputed moments."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        
        self.var = M2 / total_count
        self.count = total_count
    
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var)


def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    
    Args:
        initial_value: Initial value
        final_value: Final value
        
    Returns:
        Schedule function that takes progress (0 to 1) and returns current value
    """
    def schedule(progress: float) -> float:
        return final_value + (initial_value - final_value) * (1 - progress)
    return schedule


def get_schedule_fn(value: Union[float, str]):
    """
    Get schedule function from value or schedule specification.
    
    Args:
        value: Either a float (constant) or schedule string (e.g., "linear_0.001")
        
    Returns:
        Schedule function
    """
    if isinstance(value, (int, float)):
        return lambda _: float(value)
    
    if value.startswith("linear_"):
        initial_value = float(value.split("_")[1])
        return linear_schedule(initial_value)
    
    raise ValueError(f"Unknown schedule: {value}")

