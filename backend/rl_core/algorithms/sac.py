"""
Soft Actor-Critic (SAC) Algorithm Implementation.

References:
    - Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
    - Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..base import BaseAgent
from ..networks import MLP
from ..buffers import ReplayBuffer
from ..utils import polyak_update


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    
    Outputs mean and log_std, uses reparameterization trick for sampling.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        action_scale: float = 1.0,
    ):
        super().__init__()
        
        self.action_scale = action_scale
        
        # Feature extractor
        self.feature = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and log_std."""
        features = self.feature(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.
        
        Returns:
            action: Sampled action (after tanh squashing)
            log_prob: Log probability of action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Sample from Gaussian
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash through tanh
        action = torch.tanh(x_t) * self.action_scale
        
        # Compute log probability with Jacobian correction
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - (action / self.action_scale).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action, optionally deterministic."""
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean) * self.action_scale
        else:
            action, _ = self.sample(obs)
            return action


class QNetwork(nn.Module):
    """Twin Q-network for SAC."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
    ):
        super().__init__()
        
        # Two Q-networks for twin Q-learning
        self.q1 = MLP(obs_dim + action_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + action_dim, 1, hidden_dims)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both networks."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get Q-value from first network only."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class SAC(BaseAgent):
    """
    Soft Actor-Critic (SAC) Agent.
    
    Maximum entropy RL algorithm for continuous control.
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate for all networks
        buffer_size: Replay buffer size
        learning_starts: Steps before training starts
        batch_size: Minibatch size
        tau: Soft update coefficient
        gamma: Discount factor
        train_freq: Update frequency
        gradient_steps: Gradient steps per update
        ent_coef: Entropy coefficient (or "auto" for automatic tuning)
        target_entropy: Target entropy ("auto" = -action_dim)
        use_sde: Use State Dependent Exploration
        device: Device to use
    """
    
    def __init__(
        self,
        env: Any,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        hidden_dims: list = [256, 256],
        device: str = "auto",
    ):
        super().__init__(env, learning_rate, gamma, device)
        
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        
        # Get dimensions
        self.obs_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.shape[0]
        self.action_scale = float(env.action_space.high[0])
        
        # Create networks
        self.actor = GaussianPolicy(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            self.action_scale
        ).to(self.device)
        
        self.critic = QNetwork(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic_target = QNetwork(self.obs_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Entropy coefficient (alpha)
        self.auto_entropy = ent_coef == "auto"
        if self.auto_entropy:
            # Target entropy = -dim(A)
            if target_entropy == "auto":
                self.target_entropy = -self.action_dim
            else:
                self.target_entropy = target_entropy
            
            # Learnable log_alpha
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
            self.ent_coef = self.log_alpha.exp().item()
        else:
            self.ent_coef = ent_coef
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env.observation_space.shape,
            self.action_dim,
            self.device
        )
    
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
            action = self.actor.get_action(obs_tensor, deterministic)
        
        return action.cpu().numpy().flatten(), None
    
    def _train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        samples = self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(samples.next_observations)
            next_q1, next_q2 = self.critic_target(samples.next_observations, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.ent_coef * next_log_probs
            target_q = samples.rewards.unsqueeze(-1) + (1 - samples.dones.unsqueeze(-1)) * self.gamma * next_q
        
        # Update critic
        current_q1, current_q2 = self.critic(samples.observations, samples.actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions, log_probs = self.actor.sample(samples.observations)
        q1, q2 = self.critic(samples.observations, actions)
        min_q = torch.min(q1, q2)
        
        actor_loss = (self.ent_coef * log_probs - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update entropy coefficient
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.ent_coef = self.log_alpha.exp().item()
        
        # Update target networks
        polyak_update(self.critic, self.critic_target, self.tau)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            "alpha": self.ent_coef,
            "log_prob": log_probs.mean().item(),
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100
    ) -> "SAC":
        """
        Train the SAC agent.
        
        Args:
            total_timesteps: Total training steps
            callback: Optional callback function
            log_interval: Logging frequency
            
        Returns:
            self
        """
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(total_timesteps):
            self.num_timesteps = step
            
            # Get action
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict(obs, deterministic=False)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.replay_buffer.add(obs, action, reward, next_obs, done, info)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            # Handle episode end
            if done:
                self.training_history["episode_rewards"].append(episode_reward)
                self.training_history["episode_lengths"].append(episode_length)
                self.num_episodes += 1
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Training
            if step >= self.learning_starts and step % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    metrics = self._train_step()
                self.training_history["losses"].append(metrics["critic_loss"])
            
            # Callback
            if callback is not None:
                callback(self)
        
        return self
    
    def save(self, path: str) -> None:
        """Save agent to disk."""
        save_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "training_history": self.training_history,
        }
        if self.auto_entropy:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str, env: Any = None) -> "SAC":
        """Load agent from disk."""
        checkpoint = torch.load(path)
        agent = cls(env)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.critic_target.load_state_dict(checkpoint["critic_target"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.num_timesteps = checkpoint["num_timesteps"]
        agent.num_episodes = checkpoint["num_episodes"]
        agent.training_history = checkpoint["training_history"]
        if "log_alpha" in checkpoint:
            agent.log_alpha = checkpoint["log_alpha"]
            agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        return agent

