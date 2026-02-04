"""
Training Session
Manages individual training sessions.
"""

from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio
import numpy as np

from rl_core.algorithms import DQN, PPO, SAC, A2C
from environments import GridWorldEnv, NavigationEnv, PlatformerEnv


@dataclass
class SessionConfig:
    """Configuration for a training session."""
    
    algorithm: str = "PPO"
    environment: str = "navigation"
    environment_config: str = "simple_obstacles"
    total_timesteps: int = 100000
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    goal_text: Optional[str] = None
    render_frequency: int = 100
    checkpoint_frequency: int = 10000


@dataclass
class SessionMetrics:
    """Metrics for a training session."""
    
    current_step: int = 0
    current_episode: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    max_reward: float = 0.0
    loss: float = 0.0
    episode_length: float = 0.0
    rewards_history: list = field(default_factory=list)
    losses_history: list = field(default_factory=list)


class TrainingSession:
    """
    Manages a single training session.
    
    Handles:
    - Environment creation
    - Agent initialization
    - Training loop
    - Metrics collection
    - Checkpointing
    """
    
    ALGORITHMS = {
        "DQN": DQN,
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C,
    }
    
    ENVIRONMENTS = {
        "grid_world": GridWorldEnv,
        "navigation": NavigationEnv,
        "platformer": PlatformerEnv,
    }
    
    def __init__(
        self,
        config: SessionConfig,
        on_update: Optional[Callable] = None
    ):
        """
        Initialize training session.
        
        Args:
            config: Session configuration
            on_update: Callback for progress updates
        """
        self.id = str(uuid.uuid4())[:8]
        self.config = config
        self.on_update = on_update
        
        self.status = "initialized"
        self.metrics = SessionMetrics()
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        self.env = None
        self.agent = None
        self._stop_requested = False
        self._pause_requested = False
    
    def _create_environment(self):
        """Create the training environment."""
        env_class = self.ENVIRONMENTS.get(self.config.environment)
        if env_class is None:
            raise ValueError(f"Unknown environment: {self.config.environment}")
        
        self.env = env_class(render_mode="state")
        return self.env
    
    def _create_agent(self):
        """Create the RL agent."""
        agent_class = self.ALGORITHMS.get(self.config.algorithm)
        if agent_class is None:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        self.agent = agent_class(self.env, **self.config.hyperparameters)
        return self.agent
    
    async def start(self):
        """Start the training session."""
        try:
            self.status = "running"
            self.started_at = datetime.now()
            
            # Create environment and agent
            self._create_environment()
            self._create_agent()
            
            # Run training loop
            await self._training_loop()
            
            if not self._stop_requested:
                self.status = "completed"
            else:
                self.status = "stopped"
            
            self.completed_at = datetime.now()
            
        except Exception as e:
            self.status = "error"
            raise e
    
    async def _training_loop(self):
        """Main training loop."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.total_timesteps):
            if self._stop_requested:
                break
            
            while self._pause_requested:
                await asyncio.sleep(0.1)
            
            self.metrics.current_step = step
            
            # Get action from agent
            action, _ = self.agent.predict(obs, deterministic=False)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(
                action[0] if hasattr(action, '__len__') else action
            )
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Handle episode end
            if done:
                self.metrics.current_episode += 1
                self.metrics.total_reward = episode_reward
                self.metrics.rewards_history.append(episode_reward)
                
                # Update mean/max
                recent = self.metrics.rewards_history[-100:]
                self.metrics.mean_reward = float(np.mean(recent))
                self.metrics.max_reward = float(np.max(recent))
                self.metrics.episode_length = episode_length
                
                # Notify update
                if self.on_update:
                    await self.on_update(self.get_state())
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
            
            # Periodic state update
            if step % self.config.render_frequency == 0 and self.on_update:
                state = self.get_state()
                state["env_state"] = self.env.get_render_state()
                await self.on_update(state)
            
            # Small delay to allow other tasks
            if step % 100 == 0:
                await asyncio.sleep(0)
    
    def pause(self):
        """Pause training."""
        self._pause_requested = True
        self.status = "paused"
    
    def resume(self):
        """Resume training."""
        self._pause_requested = False
        self.status = "running"
    
    def stop(self):
        """Stop training."""
        self._stop_requested = True
        self.status = "stopping"
    
    def get_state(self) -> Dict[str, Any]:
        """Get current session state."""
        return {
            "id": self.id,
            "status": self.status,
            "config": {
                "algorithm": self.config.algorithm,
                "environment": self.config.environment,
                "total_timesteps": self.config.total_timesteps,
            },
            "metrics": {
                "current_step": self.metrics.current_step,
                "current_episode": self.metrics.current_episode,
                "total_reward": self.metrics.total_reward,
                "mean_reward": self.metrics.mean_reward,
                "max_reward": self.metrics.max_reward,
                "loss": self.metrics.loss,
            },
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }
    
    def save_checkpoint(self, path: str):
        """Save a checkpoint."""
        if self.agent:
            self.agent.save(path)
    
    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        if self.agent:
            self.agent.load(path)

