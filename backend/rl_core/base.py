"""
Base classes for RL agents, policies, and replay buffers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class BasePolicy(ABC, nn.Module):
    """Abstract base class for all policies."""
    
    def __init__(self, observation_space: Any, action_space: Any):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
    
    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy network."""
        pass
    
    @abstractmethod
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Get action from observation."""
        pass
    
    def save(self, path: str) -> None:
        """Save policy parameters."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load policy parameters."""
        self.load_state_dict(torch.load(path))


class BaseBuffer(ABC):
    """Abstract base class for experience replay buffers."""
    
    def __init__(self, buffer_size: int, observation_shape: Tuple[int, ...], action_dim: int):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.pos = 0
        self.full = False
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        pass
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.buffer_size if self.full else self.pos


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(
        self,
        env: Any,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = "auto"
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Training state
        self.num_timesteps = 0
        self.num_episodes = 0
        self._last_obs = None
        
        # Logging
        self.logger = None
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "losses": [],
            "learning_rates": [],
        }
    
    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        log_interval: int = 100
    ) -> "BaseAgent":
        """Train the agent."""
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Get action from observation."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent to disk."""
        pass
    
    @abstractmethod
    def load(cls, path: str, env: Any = None) -> "BaseAgent":
        """Load agent from disk."""
        pass
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Return current training metrics."""
        return {
            "timesteps": self.num_timesteps,
            "episodes": self.num_episodes,
            "mean_reward": np.mean(self.training_history["episode_rewards"][-100:]) if self.training_history["episode_rewards"] else 0,
            "mean_length": np.mean(self.training_history["episode_lengths"][-100:]) if self.training_history["episode_lengths"] else 0,
        }

