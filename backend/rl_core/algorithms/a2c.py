"""
Advantage Actor-Critic (A2C) Algorithm Implementation.

A2C is a synchronous variant of A3C that runs multiple environments in parallel.

References:
    - Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..networks import ActorCritic
from ..buffers import RolloutBuffer
from ..utils import explained_variance, get_schedule_fn


class A2C(BaseAgent):
    """
    Advantage Actor-Critic (A2C) Agent.
    
    Synchronous variant of A3C with parallel environment execution.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate (or schedule)
        n_steps: Number of steps per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (1.0 = Monte Carlo, 0.0 = TD(0))
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        normalize_advantage: Whether to normalize advantages
        device: Device to use
    """
    
    def __init__(
        self,
        env: Any,
        learning_rate: Union[float, str] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = False,
        hidden_dims: list = [64, 64],
        continuous: bool = False,
        device: str = "auto",
    ):
        super().__init__(env, 7e-4, gamma, device)
        
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.continuous = continuous
        
        # Learning rate schedule
        self.lr_schedule = get_schedule_fn(learning_rate)
        self.learning_rate = self.lr_schedule(1.0)
        
        # Get dimensions
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
            shared_layers=1,  # A2C typically uses shared layers
            continuous=self.continuous
        ).to(self.device)
        
        # Single optimizer for entire network
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=self.learning_rate,
            alpha=0.99,
            eps=1e-5
        )
        
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
                policy_output, _ = self.policy(obs_tensor)
                action = policy_output.argmax(dim=-1)
        
        return action.cpu().numpy(), {
            "log_prob": log_prob.cpu().numpy(),
            "value": value.cpu().numpy()
        }
    
    def collect_rollout(self) -> bool:
        """Collect a short rollout for A2C."""
        self.rollout_buffer.reset()
        
        obs = self._last_obs
        episode_rewards = []
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
    
    def train(self) -> Dict[str, float]:
        """Perform A2C update (single pass through all data)."""
        # Get all data from buffer (no batching for A2C)
        for batch in self.rollout_buffer.get(batch_size=None):
            # Get current policy outputs
            if self.continuous:
                actions = batch.actions
            else:
                actions = batch.actions.squeeze(-1).long()
            
            _, log_prob, entropy, value = self.policy.get_action_and_value(
                batch.observations,
                actions
            )
            
            # Advantages
            advantages = batch.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss (negative because we want gradient ascent)
            policy_loss = -(log_prob * advantages).mean()
            
            # Value loss
            value_loss = F.mse_loss(value, batch.returns)
            
            # Entropy loss (negative because we want to maximize entropy)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Compute explained variance for logging
        with torch.no_grad():
            ev = explained_variance(
                batch.old_values.cpu().numpy(),
                batch.returns.cpu().numpy()
            )
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "explained_variance": ev,
            "loss": loss.item(),
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100
    ) -> "A2C":
        """
        Train the A2C agent.
        
        Args:
            total_timesteps: Total training steps
            callback: Optional callback function
            log_interval: Logging frequency
            
        Returns:
            self
        """
        # Initialize
        obs, _ = self.env.reset()
        self._last_obs = obs
        
        n_updates = total_timesteps // self.n_steps
        
        for update in range(n_updates):
            progress = update / n_updates
            self._update_learning_rate(progress)
            
            # Collect rollout
            self.collect_rollout()
            self.num_timesteps += self.n_steps
            
            # Train
            metrics = self.train()
            self.training_history["losses"].append(metrics["loss"])
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
    def load(cls, path: str, env: Any = None) -> "A2C":
        """Load agent from disk."""
        checkpoint = torch.load(path)
        agent = cls(env)
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.num_timesteps = checkpoint["num_timesteps"]
        agent.num_episodes = checkpoint["num_episodes"]
        agent.training_history = checkpoint["training_history"]
        return agent

