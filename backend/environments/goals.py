"""
Goal definitions for RL-GYM environments.

Goals define the objectives that agents must achieve.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


class Goal(ABC):
    """
    Abstract base class for environment goals.
    
    Goals define:
    - Success conditions
    - Reward shaping
    - Termination conditions
    """
    
    def __init__(
        self,
        position: np.ndarray,
        size: np.ndarray,
        reward: float = 100.0,
        goal_type: str = "generic"
    ):
        """
        Initialize goal.
        
        Args:
            position: Goal position [x, y]
            size: Goal size [width, height]
            reward: Reward for achieving goal
            goal_type: Type identifier
        """
        self.position = np.array(position, dtype=np.float32)
        self.size = np.array(size, dtype=np.float32)
        self.reward = reward
        self.goal_type = goal_type
        self.achieved = False
    
    def reset(self) -> None:
        """Reset goal state."""
        self.achieved = False
    
    @abstractmethod
    def check_achieved(self, agent_position: np.ndarray, **kwargs) -> bool:
        """
        Check if goal has been achieved.
        
        Args:
            agent_position: Current agent position
            **kwargs: Additional state information
            
        Returns:
            True if goal is achieved
        """
        pass
    
    @abstractmethod
    def get_reward(self, agent_position: np.ndarray, **kwargs) -> float:
        """
        Calculate reward based on agent state.
        
        Args:
            agent_position: Current agent position
            **kwargs: Additional state information
            
        Returns:
            Reward value
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.goal_type,
            "position": self.position.tolist(),
            "size": self.size.tolist(),
            "reward": self.reward,
            "achieved": self.achieved,
        }


class PositionGoal(Goal):
    """
    Goal to reach a specific position.
    
    Agent must navigate to the goal area to succeed.
    """
    
    def __init__(
        self,
        position: np.ndarray,
        size: np.ndarray = np.array([30, 30]),
        reward: float = 100.0,
        distance_reward_scale: float = 0.1,
        sparse_reward: bool = False
    ):
        """
        Initialize position goal.
        
        Args:
            position: Target position
            size: Goal area size
            reward: Reward for reaching goal
            distance_reward_scale: Scale for distance-based shaping
            sparse_reward: If True, only give reward when goal achieved
        """
        super().__init__(position, size, reward, "position")
        self.distance_reward_scale = distance_reward_scale
        self.sparse_reward = sparse_reward
        self._prev_distance = None
    
    def reset(self) -> None:
        """Reset goal state."""
        super().reset()
        self._prev_distance = None
    
    def check_achieved(self, agent_position: np.ndarray, **kwargs) -> bool:
        """Check if agent has reached the goal area."""
        if self.achieved:
            return True
        
        half_size = self.size / 2
        in_goal = (abs(agent_position[0] - self.position[0]) < half_size[0] and
                   abs(agent_position[1] - self.position[1]) < half_size[1])
        
        if in_goal:
            self.achieved = True
        
        return self.achieved
    
    def get_reward(self, agent_position: np.ndarray, **kwargs) -> float:
        """
        Calculate reward based on distance to goal.
        
        Uses potential-based reward shaping to guide agent.
        """
        # Check if just achieved
        if self.check_achieved(agent_position):
            return self.reward
        
        if self.sparse_reward:
            return 0.0
        
        # Distance-based shaping
        current_distance = np.linalg.norm(agent_position - self.position)
        
        if self._prev_distance is None:
            self._prev_distance = current_distance
            return 0.0
        
        # Reward for getting closer (potential-based shaping)
        reward = (self._prev_distance - current_distance) * self.distance_reward_scale
        self._prev_distance = current_distance
        
        return reward


class CollectGoal(Goal):
    """
    Goal to collect items.
    
    Agent must collect one or more items scattered in the environment.
    """
    
    def __init__(
        self,
        items: List[np.ndarray],
        item_size: float = 15,
        reward_per_item: float = 10.0,
        completion_bonus: float = 50.0
    ):
        """
        Initialize collection goal.
        
        Args:
            items: List of item positions
            item_size: Size of each item
            reward_per_item: Reward for collecting each item
            completion_bonus: Bonus for collecting all items
        """
        # Calculate center position for visualization
        center = np.mean(items, axis=0)
        super().__init__(
            position=center,
            size=np.array([item_size, item_size]),
            reward=completion_bonus,
            goal_type="collect"
        )
        
        self.items = [np.array(item, dtype=np.float32) for item in items]
        self.item_size = item_size
        self.reward_per_item = reward_per_item
        self.completion_bonus = completion_bonus
        self.collected = [False] * len(items)
    
    def reset(self) -> None:
        """Reset goal state."""
        super().reset()
        self.collected = [False] * len(self.items)
    
    def check_achieved(self, agent_position: np.ndarray, **kwargs) -> bool:
        """Check if all items have been collected."""
        if self.achieved:
            return True
        
        self.achieved = all(self.collected)
        return self.achieved
    
    def get_reward(self, agent_position: np.ndarray, **kwargs) -> float:
        """Calculate reward for collecting items."""
        reward = 0.0
        
        for i, (item, collected) in enumerate(zip(self.items, self.collected)):
            if collected:
                continue
            
            # Check if agent is touching item
            distance = np.linalg.norm(agent_position - item)
            if distance < self.item_size:
                self.collected[i] = True
                reward += self.reward_per_item
        
        # Check for completion bonus
        if all(self.collected) and not self.achieved:
            self.achieved = True
            reward += self.completion_bonus
        
        return reward
    
    @property
    def items_remaining(self) -> int:
        """Get number of items remaining."""
        return sum(1 for c in self.collected if not c)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "items": [item.tolist() for item in self.items],
            "collected": self.collected,
            "items_remaining": self.items_remaining,
        })
        return data


class SurvivalGoal(Goal):
    """
    Goal to survive for a certain duration.
    
    Agent must avoid hazards and stay alive.
    """
    
    def __init__(
        self,
        target_steps: int = 500,
        reward_per_step: float = 0.1,
        completion_bonus: float = 100.0
    ):
        """
        Initialize survival goal.
        
        Args:
            target_steps: Number of steps to survive
            reward_per_step: Reward per survived step
            completion_bonus: Bonus for reaching target
        """
        super().__init__(
            position=np.array([0, 0]),  # Position not relevant
            size=np.array([0, 0]),
            reward=completion_bonus,
            goal_type="survival"
        )
        
        self.target_steps = target_steps
        self.reward_per_step = reward_per_step
        self.completion_bonus = completion_bonus
        self.steps_survived = 0
    
    def reset(self) -> None:
        """Reset goal state."""
        super().reset()
        self.steps_survived = 0
    
    def check_achieved(self, agent_position: np.ndarray, **kwargs) -> bool:
        """Check if survival target reached."""
        if self.achieved:
            return True
        
        self.achieved = self.steps_survived >= self.target_steps
        return self.achieved
    
    def get_reward(self, agent_position: np.ndarray, **kwargs) -> float:
        """Calculate survival reward."""
        self.steps_survived += 1
        
        if self.steps_survived >= self.target_steps and not self.achieved:
            self.achieved = True
            return self.reward_per_step + self.completion_bonus
        
        return self.reward_per_step
    
    @property
    def progress(self) -> float:
        """Get survival progress (0 to 1)."""
        return min(1.0, self.steps_survived / self.target_steps)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "target_steps": self.target_steps,
            "steps_survived": self.steps_survived,
            "progress": self.progress,
        })
        return data


class MultiGoal(Goal):
    """
    Composite goal combining multiple sub-goals.
    
    Can require all goals (AND) or any goal (OR) to be achieved.
    """
    
    def __init__(
        self,
        goals: List[Goal],
        require_all: bool = True,
        completion_bonus: float = 0.0
    ):
        """
        Initialize multi-goal.
        
        Args:
            goals: List of sub-goals
            require_all: If True, all goals must be achieved (AND)
            completion_bonus: Bonus when multi-goal is achieved
        """
        super().__init__(
            position=np.array([0, 0]),
            size=np.array([0, 0]),
            reward=completion_bonus,
            goal_type="multi"
        )
        
        self.goals = goals
        self.require_all = require_all
        self.completion_bonus = completion_bonus
    
    def reset(self) -> None:
        """Reset all sub-goals."""
        super().reset()
        for goal in self.goals:
            goal.reset()
    
    def check_achieved(self, agent_position: np.ndarray, **kwargs) -> bool:
        """Check if composite goal is achieved."""
        if self.achieved:
            return True
        
        achieved_goals = [goal.check_achieved(agent_position, **kwargs) for goal in self.goals]
        
        if self.require_all:
            self.achieved = all(achieved_goals)
        else:
            self.achieved = any(achieved_goals)
        
        return self.achieved
    
    def get_reward(self, agent_position: np.ndarray, **kwargs) -> float:
        """Sum rewards from all sub-goals."""
        reward = sum(goal.get_reward(agent_position, **kwargs) for goal in self.goals)
        
        if self.check_achieved(agent_position, **kwargs) and not self._gave_bonus:
            reward += self.completion_bonus
            self._gave_bonus = True
        
        return reward
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.goal_type,
            "require_all": self.require_all,
            "goals": [goal.to_dict() for goal in self.goals],
            "achieved": self.achieved,
        }


class TimedGoal(Goal):
    """
    Goal that must be achieved within a time limit.
    """
    
    def __init__(
        self,
        base_goal: Goal,
        time_limit: int,
        time_bonus_scale: float = 0.5
    ):
        """
        Initialize timed goal.
        
        Args:
            base_goal: The underlying goal to achieve
            time_limit: Maximum steps allowed
            time_bonus_scale: Bonus multiplier for remaining time
        """
        super().__init__(
            position=base_goal.position,
            size=base_goal.size,
            reward=base_goal.reward,
            goal_type="timed"
        )
        
        self.base_goal = base_goal
        self.time_limit = time_limit
        self.time_bonus_scale = time_bonus_scale
        self.steps_elapsed = 0
    
    def reset(self) -> None:
        """Reset goal state."""
        super().reset()
        self.base_goal.reset()
        self.steps_elapsed = 0
    
    def check_achieved(self, agent_position: np.ndarray, **kwargs) -> bool:
        """Check if base goal achieved within time limit."""
        self.steps_elapsed += 1
        return self.base_goal.check_achieved(agent_position, **kwargs)
    
    def get_reward(self, agent_position: np.ndarray, **kwargs) -> float:
        """Calculate reward with time bonus."""
        reward = self.base_goal.get_reward(agent_position, **kwargs)
        
        # Add time bonus if goal just achieved
        if self.base_goal.achieved and not self.achieved:
            self.achieved = True
            time_remaining = max(0, self.time_limit - self.steps_elapsed)
            reward += time_remaining * self.time_bonus_scale
        
        return reward
    
    @property
    def time_remaining(self) -> int:
        """Get remaining steps."""
        return max(0, self.time_limit - self.steps_elapsed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "base_goal": self.base_goal.to_dict(),
            "time_limit": self.time_limit,
            "steps_elapsed": self.steps_elapsed,
            "time_remaining": self.time_remaining,
        })
        return data


def create_goal_from_config(config: Dict[str, Any]) -> Goal:
    """
    Factory function to create goals from configuration.
    
    Args:
        config: Goal configuration dictionary
        
    Returns:
        Created goal instance
    """
    goal_type = config.get("type", "position")
    
    if goal_type == "position":
        return PositionGoal(
            position=np.array(config["position"]),
            size=np.array(config.get("size", [30, 30])),
            reward=config.get("reward", 100.0),
            distance_reward_scale=config.get("distance_reward_scale", 0.1),
            sparse_reward=config.get("sparse_reward", False)
        )
    elif goal_type == "collect":
        return CollectGoal(
            items=config["items"],
            item_size=config.get("item_size", 15),
            reward_per_item=config.get("reward_per_item", 10.0),
            completion_bonus=config.get("completion_bonus", 50.0)
        )
    elif goal_type == "survival":
        return SurvivalGoal(
            target_steps=config.get("target_steps", 500),
            reward_per_step=config.get("reward_per_step", 0.1),
            completion_bonus=config.get("completion_bonus", 100.0)
        )
    else:
        raise ValueError(f"Unknown goal type: {goal_type}")

