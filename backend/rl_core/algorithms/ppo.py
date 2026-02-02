"""
Proximal Policy Optimization (PPO) Algorithm Implementation.

References:
    - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..base import BaseAgent
from ..networks import ActorCritic
from ..buffers import RolloutBuffer
from ..utils import explained_variance, get_schedule_fn


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.
    
    Uses clipped surrogate objective for stable policy updates.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate (or schedule)
        n_steps: Number of steps per rollout
        batch_size: Minibatch size for updates
        n_epochs: Number of optimization epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping parameter
        clip_range_vf: Value function clipping (None = no clipping)
        normalize_advantage: Normalize advantages
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        target_kl: Target KL divergence (early stopping)
        device: Device to use
    """
    
    def __init__(
        self,
        env: Any,
        learning_rate: Union[float, str] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        hidden_dims: list = [256, 256],
        continuous: bool = False,
        device: str = "auto",
    ):
        super().__init__(env, 3e-4, gamma, device)
        
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.continuous = continuous
        
        # Learning rate schedule
        self.lr_schedule = get_schedule_fn(learning_rate)
        self.learning_rate = self.lr_schedule(1.0)
        
        # Get dimensions from environment
        self.obs_dim = np.prod(env.observation_space.shape)
        if hasattr(env.action_space, 'n'):
            self.action_dim = env.action_space.n
            self.continuous = False
        else:
            self.action_dim = env.action_space.shape[0]
            self.continuous = True
        
        # Create actor-critic network
        self.policy = ActorCritic(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            shared_layers=0,
            continuous=self.continuous
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            n_steps,
            env.observation_space.shape,
            self.action_dim if self.continuous else 1,
            self.device,
            gamma,
            gae_lambda,
            n_envs=1
        )
    
    def _update_learning_rate(self, progress: float) -> None:
        """Update learning rate based on progress."""
        self.learning_rate = self.lr_schedule(1.0 - progress)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Get action from observation."""
        obs_tensor = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)
            
            if deterministic and not self.continuous:
                # For discrete actions, take argmax
                policy_output, _ = self.policy(obs_tensor)
                action = policy_output.argmax(dim=-1)
        
        return action.cpu().numpy(), {
            "log_prob": log_prob.cpu().numpy(),
            "value": value.cpu().numpy()
        }
    
    def collect_rollout(self) -> bool:
        """Collect a complete rollout."""
        self.rollout_buffer.reset()
        
        obs = self._last_obs
        episode_rewards = []
        episode_lengths = []
        current_reward = 0
        current_length = 0
        
        for step in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)
            
            action_np = action.cpu().numpy()
            if not self.continuous:
                action_np = action_np.flatten()
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(
                action_np[0] if not self.continuous else action_np.flatten()
            )
            done = terminated or truncated
            
            current_reward += reward
            current_length += 1
            
            # Store transition
            self.rollout_buffer.add(
                obs.reshape(1, -1),
                action_np.reshape(1, -1) if self.continuous else action_np.reshape(1, 1),
                np.array([reward]),
                np.array([done]),
                value.cpu().numpy(),
                log_prob.cpu().numpy()
            )
            
            obs = next_obs
            
            if done:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                self.training_history["episode_rewards"].append(current_reward)
                self.training_history["episode_lengths"].append(current_length)
                self.num_episodes += 1
                
                obs, _ = self.env.reset()
                current_reward = 0
                current_length = 0
        
        self._last_obs = obs
        
        # Compute returns and advantages
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            last_value = self.policy.get_value(obs_tensor).cpu().numpy()
        
        self.rollout_buffer.compute_returns_and_advantages(
            last_value,
            np.array([done])
        )
        
        return True
    
    def train_epoch(self) -> Dict[str, float]:
        """Perform one epoch of PPO updates."""
        clip_fracs = []
        policy_losses = []
        value_losses = []
        entropy_losses = []
        approx_kl_divs = []
        
        for batch in self.rollout_buffer.get(self.batch_size):
            # Get current policy outputs
            if self.continuous:
                actions = batch.actions
            else:
                actions = batch.actions.squeeze(-1).long()
            
            _, log_prob, entropy, value = self.policy.get_action_and_value(
                batch.observations,
                actions
            )
            
            # Ratio for importance sampling
            log_ratio = log_prob - batch.old_log_probs
            ratio = log_ratio.exp()
            
            # Approximate KL divergence
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                approx_kl_divs.append(approx_kl)
            
            # Clipped surrogate objective
            advantages = batch.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            # Clip fraction for logging
            clip_fracs.append(((ratio - 1).abs() > self.clip_range).float().mean().item())
            
            # Value loss
            if self.clip_range_vf is not None:
                value_clipped = batch.old_values + torch.clamp(
                    value - batch.old_values, -self.clip_range_vf, self.clip_range_vf
                )
                value_loss_1 = (value - batch.returns) ** 2
                value_loss_2 = (value_clipped - batch.returns) ** 2
                value_loss = torch.max(value_loss_1, value_loss_2).mean()
            else:
                value_loss = ((value - batch.returns) ** 2).mean()
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(-entropy_loss.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "approx_kl": np.mean(approx_kl_divs),
            "clip_fraction": np.mean(clip_fracs),
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 1
    ) -> "PPO":
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training steps
            callback: Optional callback function
            log_interval: Logging frequency (in rollouts)
            
        Returns:
            self
        """
        # Initialize
        obs, _ = self.env.reset()
        self._last_obs = obs
        
        n_rollouts = total_timesteps // self.n_steps
        
        for rollout in range(n_rollouts):
            progress = rollout / n_rollouts
            self._update_learning_rate(progress)
            
            # Collect rollout
            self.collect_rollout()
            self.num_timesteps += self.n_steps
            
            # Train for n_epochs
            all_metrics = []
            for epoch in range(self.n_epochs):
                metrics = self.train_epoch()
                all_metrics.append(metrics)
                
                # Early stopping based on KL divergence
                if self.target_kl is not None and metrics["approx_kl"] > self.target_kl:
                    break
            
            # Average metrics
            avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
            self.training_history["losses"].append(avg_metrics["policy_loss"])
            self.training_history["learning_rates"].append(self.learning_rate)
            
            # Callback
            if callback is not None:
                callback(self)
        
        return self
    
    def save(self, path: str) -> None:
        """Save agent to disk."""
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "training_history": self.training_history,
        }, path)
    
    @classmethod
    def load(cls, path: str, env: Any = None) -> "PPO":
        """Load agent from disk."""
        checkpoint = torch.load(path)
        agent = cls(env)
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.num_timesteps = checkpoint["num_timesteps"]
        agent.num_episodes = checkpoint["num_episodes"]
        agent.training_history = checkpoint["training_history"]
        return agent

