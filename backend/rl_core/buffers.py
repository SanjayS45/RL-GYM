"""
Experience replay buffers for RL algorithms.
"""

from typing import Dict, Optional, Tuple, Union, NamedTuple
import numpy as np
import torch

from .base import BaseBuffer


class ReplayBufferSamples(NamedTuple):
    """Named tuple for replay buffer samples."""
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer(BaseBuffer):
    """
    Standard experience replay buffer for off-policy algorithms.
    
    Stores transitions (s, a, r, s', done) and samples uniformly.
    
    Args:
        buffer_size: Maximum buffer size
        observation_shape: Shape of observations
        action_dim: Dimension of actions
        device: Device to put tensors on
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(buffer_size, observation_shape, action_dim)
        
        self.device = torch.device(device)
        
        # Allocate buffer arrays
        self.observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """Add a transition to the buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0
    
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """Sample a batch of transitions."""
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)
        
        return ReplayBufferSamples(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
        )


class RolloutBufferSamples(NamedTuple):
    """Named tuple for rollout buffer samples."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """
    Rollout buffer for on-policy algorithms (PPO, A2C).
    
    Stores complete rollouts with computed advantages and returns.
    
    Args:
        buffer_size: Rollout length (steps per environment)
        observation_shape: Shape of observations
        action_dim: Dimension of actions
        device: Device to put tensors on
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        n_envs: Number of parallel environments
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_envs = n_envs
        
        self.pos = 0
        self.full = False
        
        # Allocate buffer arrays
        self.observations = np.zeros((buffer_size, n_envs, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        # Computed after rollout
        self.advantages = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        self.generator_ready = False
    
    def reset(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.generator_ready = False
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """Add a step to the rollout buffer."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob.flatten()
        
        self.pos += 1
        self.full = self.pos == self.buffer_size
    
    def compute_returns_and_advantages(self, last_values: np.ndarray, dones: np.ndarray) -> None:
        """
        Compute GAE advantages and returns after rollout is complete.
        
        Args:
            last_values: Value estimates for final observations
            dones: Done flags for final step
        """
        last_values = last_values.flatten()
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            self.advantages[step] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
        
        self.returns = self.advantages + self.values
        self.generator_ready = True
    
    def get(self, batch_size: Optional[int] = None) -> RolloutBufferSamples:
        """
        Get all data from the buffer.
        
        Args:
            batch_size: If provided, yield batches of this size
            
        Yields:
            Batches of rollout data
        """
        assert self.generator_ready, "Call compute_returns_and_advantages() first!"
        
        # Flatten buffer
        total_size = self.buffer_size * self.n_envs
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        
        # Reshape arrays: (buffer_size, n_envs, ...) -> (total_size, ...)
        observations = self.observations.reshape(total_size, *self.observation_shape)
        actions = self.actions.reshape(total_size, self.action_dim)
        values = self.values.reshape(total_size)
        log_probs = self.log_probs.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if batch_size is None:
            batch_size = total_size
        
        start_idx = 0
        while start_idx < total_size:
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            yield RolloutBufferSamples(
                observations=torch.as_tensor(observations[batch_indices], device=self.device),
                actions=torch.as_tensor(actions[batch_indices], device=self.device),
                old_values=torch.as_tensor(values[batch_indices], device=self.device),
                old_log_probs=torch.as_tensor(log_probs[batch_indices], device=self.device),
                advantages=torch.as_tensor(advantages[batch_indices], device=self.device),
                returns=torch.as_tensor(returns[batch_indices], device=self.device),
            )
            
            start_idx += batch_size


class PrioritizedReplayBuffer(BaseBuffer):
    """
    Prioritized Experience Replay buffer.
    
    Samples transitions with probability proportional to their TD error.
    
    Args:
        buffer_size: Maximum buffer size
        observation_shape: Shape of observations
        action_dim: Dimension of actions
        device: Device to put tensors on
        alpha: Priority exponent (0 = uniform, 1 = proportional)
        beta: Importance sampling exponent (start value)
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(buffer_size, observation_shape, action_dim)
        
        self.device = torch.device(device)
        self.alpha = alpha
        self.beta = beta
        
        # Allocate buffer arrays
        self.observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        
        # Priorities
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        info: Optional[Dict] = None
    ) -> None:
        """Add a transition with maximum priority."""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.priorities[self.pos] = self.max_priority ** self.alpha
        
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0
    
    def sample(self, batch_size: int, beta: Optional[float] = None) -> Tuple[ReplayBufferSamples, np.ndarray, np.ndarray]:
        """
        Sample a batch with prioritized sampling.
        
        Returns:
            samples: Buffer samples
            indices: Sampled indices (for updating priorities)
            weights: Importance sampling weights
        """
        if beta is None:
            beta = self.beta
        
        max_idx = self.buffer_size if self.full else self.pos
        
        # Compute sampling probabilities
        priorities = self.priorities[:max_idx]
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(max_idx, size=batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (max_idx * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        samples = ReplayBufferSamples(
            observations=torch.as_tensor(self.observations[indices], device=self.device),
            actions=torch.as_tensor(self.actions[indices], device=self.device),
            next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
            rewards=torch.as_tensor(self.rewards[indices], device=self.device),
            dones=torch.as_tensor(self.dones[indices], device=self.device),
        )
        
        return samples, indices, torch.as_tensor(weights, device=self.device, dtype=torch.float32)
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

