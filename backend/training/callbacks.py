"""
Training Callbacks
Callbacks for monitoring and controlling training.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
import json


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_step(self, step: int, info: Dict[str, Any]) -> bool:
        """
        Called after each training step.
        
        Args:
            step: Current step number
            info: Step information
            
        Returns:
            True to continue, False to stop training
        """
        pass
    
    def on_episode_end(self, episode: int, info: Dict[str, Any]):
        """Called at the end of each episode."""
        pass
    
    def on_training_start(self, info: Dict[str, Any]):
        """Called at the start of training."""
        pass
    
    def on_training_end(self, info: Dict[str, Any]):
        """Called at the end of training."""
        pass


class LoggingCallback(TrainingCallback):
    """
    Callback for logging training progress.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_interval: int = 100,
        verbose: int = 1
    ):
        """
        Initialize logging callback.
        
        Args:
            log_dir: Directory for log files
            log_interval: Steps between logs
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.verbose = verbose
        
        self.log_file = None
        self.history: List[Dict[str, Any]] = []
    
    def on_training_start(self, info: Dict[str, Any]):
        """Initialize logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.json"
        
        if self.verbose > 0:
            print(f"Training started. Logging to: {self.log_file}")
    
    def on_step(self, step: int, info: Dict[str, Any]) -> bool:
        """Log step information."""
        if step % self.log_interval == 0:
            entry = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **info
            }
            self.history.append(entry)
            
            if self.verbose >= 2:
                print(f"Step {step}: {info}")
        
        return True
    
    def on_episode_end(self, episode: int, info: Dict[str, Any]):
        """Log episode completion."""
        entry = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            **info
        }
        self.history.append(entry)
        
        if self.verbose >= 1:
            reward = info.get("episode_reward", "N/A")
            length = info.get("episode_length", "N/A")
            print(f"Episode {episode}: reward={reward}, length={length}")
    
    def on_training_end(self, info: Dict[str, Any]):
        """Save final logs."""
        if self.log_file:
            with open(self.log_file, "w") as f:
                json.dump({
                    "history": self.history,
                    "final_info": info,
                }, f, indent=2)
        
        if self.verbose > 0:
            print(f"Training ended. Logs saved to: {self.log_file}")


class CheckpointCallback(TrainingCallback):
    """
    Callback for saving model checkpoints.
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_interval: int = 10000,
        save_best: bool = True,
        keep_last_n: int = 5
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            save_dir: Directory for checkpoints
            save_interval: Steps between saves
            save_best: Save best model by reward
            keep_last_n: Number of recent checkpoints to keep
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.save_best = save_best
        self.keep_last_n = keep_last_n
        
        self.best_reward = float("-inf")
        self.checkpoints: List[Path] = []
        self.agent = None
    
    def set_agent(self, agent):
        """Set the agent to checkpoint."""
        self.agent = agent
    
    def on_step(self, step: int, info: Dict[str, Any]) -> bool:
        """Check for checkpoint save."""
        if step % self.save_interval == 0 and step > 0:
            self._save_checkpoint(step, info)
        
        return True
    
    def on_episode_end(self, episode: int, info: Dict[str, Any]):
        """Check for best model save."""
        if self.save_best:
            reward = info.get("episode_reward", 0)
            if reward > self.best_reward:
                self.best_reward = reward
                self._save_best(episode, info)
    
    def _save_checkpoint(self, step: int, info: Dict[str, Any]):
        """Save a checkpoint."""
        if self.agent is None:
            return
        
        path = self.save_dir / f"checkpoint_step_{step}.pt"
        self.agent.save(str(path))
        self.checkpoints.append(path)
        
        # Clean up old checkpoints
        while len(self.checkpoints) > self.keep_last_n:
            old = self.checkpoints.pop(0)
            if old.exists():
                old.unlink()
    
    def _save_best(self, episode: int, info: Dict[str, Any]):
        """Save best model."""
        if self.agent is None:
            return
        
        path = self.save_dir / "best_model.pt"
        self.agent.save(str(path))
        
        # Save metadata
        meta_path = self.save_dir / "best_model_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "episode": episode,
                "reward": info.get("episode_reward"),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)


class EarlyStoppingCallback(TrainingCallback):
    """
    Callback for early stopping based on reward plateau.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_improvement: float = 0.01,
        check_interval: int = 10
    ):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Episodes without improvement before stopping
            min_improvement: Minimum improvement threshold
            check_interval: Episodes between checks
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.check_interval = check_interval
        
        self.best_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.reward_history: List[float] = []
    
    def on_step(self, step: int, info: Dict[str, Any]) -> bool:
        """Check on each step (no-op here)."""
        return True
    
    def on_episode_end(self, episode: int, info: Dict[str, Any]):
        """Check for early stopping."""
        reward = info.get("episode_reward", 0)
        self.reward_history.append(reward)
        
        if episode % self.check_interval == 0 and episode > 0:
            # Calculate mean recent reward
            recent = self.reward_history[-self.check_interval:]
            mean_reward = sum(recent) / len(recent)
            
            if mean_reward > self.best_reward + self.min_improvement:
                self.best_reward = mean_reward
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += self.check_interval
            
            if self.episodes_without_improvement >= self.patience:
                print(f"Early stopping: No improvement for {self.patience} episodes")
                # Note: actual stopping would need to signal the training loop


class CompositeCallback(TrainingCallback):
    """
    Combines multiple callbacks.
    """
    
    def __init__(self, callbacks: List[TrainingCallback]):
        """
        Initialize composite callback.
        
        Args:
            callbacks: List of callbacks to combine
        """
        self.callbacks = callbacks
    
    def on_training_start(self, info: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_training_start(info)
    
    def on_step(self, step: int, info: Dict[str, Any]) -> bool:
        for cb in self.callbacks:
            if not cb.on_step(step, info):
                return False
        return True
    
    def on_episode_end(self, episode: int, info: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_episode_end(episode, info)
    
    def on_training_end(self, info: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_training_end(info)

