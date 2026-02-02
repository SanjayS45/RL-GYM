"""
Deep Q-Network (DQN) Algorithm Implementation.

References:
    - Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
    - Mnih et al., "Human-level control through deep RL" (2015)
"""

import copy
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseAgent
from ..networks import MLP, CNN
from ..buffers import ReplayBuffer, PrioritizedReplayBuffer
from ..utils import polyak_update, linear_schedule


class QNetwork(nn.Module):
    """Q-Network for DQN."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256],
        use_dueling: bool = False,
    ):
        super().__init__()
        self.use_dueling = use_dueling
        
        if use_dueling:
            # Dueling DQN architecture
            self.feature = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1])
            self.value_stream = nn.Linear(hidden_dims[-1], 1)
            self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.q_net = MLP(obs_dim, action_dim, hidden_dims)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.use_dueling:
            features = self.feature(obs)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Combine value and advantage
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values
        else:
            return self.q_net(obs)


class DQN(BaseAgent):
    """
    Deep Q-Network (DQN) Agent.
    
    Supports:
        - Double DQN
        - Dueling DQN
        - Prioritized Experience Replay
        - N-step returns
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        learning_starts: Steps before training starts
        batch_size: Minibatch size
        tau: Soft update coefficient
        gamma: Discount factor
        train_freq: Update frequency (steps)
        target_update_interval: Target network update frequency
        exploration_fraction: Fraction of training for epsilon decay
        exploration_initial_eps: Initial epsilon
        exploration_final_eps: Final epsilon
        max_grad_norm: Maximum gradient norm for clipping
        use_double_dqn: Use Double DQN
        use_dueling: Use Dueling architecture
        use_prioritized_replay: Use Prioritized Experience Replay
        device: Device to use
    """
    
    def __init__(
        self,
        env: Any,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 64,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
        hidden_dims: list = [256, 256],
        device: str = "auto",
    ):
        super().__init__(env, learning_rate, gamma, device)
        
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        
        # Get dimensions from environment
        self.obs_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.n
        
        # Create Q-networks
        self.q_network = QNetwork(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            use_dueling
        ).to(self.device)
        
        self.target_network = QNetwork(
            self.obs_dim,
            self.action_dim,
            hidden_dims,
            use_dueling
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size,
                env.observation_space.shape,
                1,  # Action is scalar for discrete
                self.device
            )
        else:
            self.replay_buffer = ReplayBuffer(
                buffer_size,
                env.observation_space.shape,
                1,
                self.device
            )
        
        # Exploration schedule
        self.epsilon = exploration_initial_eps
    
    def _get_epsilon(self, progress: float) -> float:
        """Get current epsilon based on training progress."""
        if progress < self.exploration_fraction:
            # Linear decay
            return self.exploration_initial_eps - progress / self.exploration_fraction * (
                self.exploration_initial_eps - self.exploration_final_eps
            )
        return self.exploration_final_eps
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Get action from observation."""
        if not deterministic and np.random.random() < self.epsilon:
            action = np.array([self.env.action_space.sample()])
        else:
            obs_tensor = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax(dim=1).cpu().numpy()
        
        return action, {"q_values": None}
    
    def _train_step(self, progress: float) -> Dict[str, float]:
        """Perform one training step."""
        # Sample from buffer
        if self.use_prioritized_replay:
            # Anneal beta from 0.4 to 1.0
            beta = 0.4 + progress * (1.0 - 0.4)
            samples, indices, weights = self.replay_buffer.sample(self.batch_size, beta)
        else:
            samples = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, device=self.device)
        
        # Compute current Q values
        current_q = self.q_network(samples.observations)
        current_q = current_q.gather(1, samples.actions.long()).squeeze(-1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: select action with online network, evaluate with target
                next_actions = self.q_network(samples.next_observations).argmax(dim=1, keepdim=True)
                next_q = self.target_network(samples.next_observations).gather(1, next_actions).squeeze(-1)
            else:
                next_q = self.target_network(samples.next_observations).max(dim=1)[0]
            
            target_q = samples.rewards + (1 - samples.dones) * self.gamma * next_q
        
        # Compute TD error
        td_error = current_q - target_q
        
        # Compute loss (weighted for PER)
        loss = (weights * (td_error ** 2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update priorities for PER
        if self.use_prioritized_replay:
            priorities = td_error.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        return {
            "loss": loss.item(),
            "q_values": current_q.mean().item(),
            "td_error": td_error.abs().mean().item(),
        }
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100
    ) -> "DQN":
        """
        Train the DQN agent.
        
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
            progress = step / total_timesteps
            self.epsilon = self._get_epsilon(progress)
            
            # Get action
            action, _ = self.predict(obs, deterministic=False)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action[0])
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
                metrics = self._train_step(progress)
                self.training_history["losses"].append(metrics["loss"])
            
            # Update target network
            if step % self.target_update_interval == 0:
                if self.tau == 1.0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                else:
                    polyak_update(self.q_network, self.target_network, self.tau)
            
            # Callback
            if callback is not None:
                callback(self)
        
        return self
    
    def save(self, path: str) -> None:
        """Save agent to disk."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps,
            "num_episodes": self.num_episodes,
            "training_history": self.training_history,
        }, path)
    
    @classmethod
    def load(cls, path: str, env: Any = None) -> "DQN":
        """Load agent from disk."""
        checkpoint = torch.load(path)
        agent = cls(env)
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.num_timesteps = checkpoint["num_timesteps"]
        agent.num_episodes = checkpoint["num_episodes"]
        agent.training_history = checkpoint["training_history"]
        return agent

