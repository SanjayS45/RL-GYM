"""
Reward Generator

Generates reward functions from parsed goals.
"""

from typing import Dict, Any, Callable, Optional
import numpy as np

from .goal_parser import ParsedGoal


class RewardGenerator:
    """
    Generates reward functions from ParsedGoal specifications.
    
    Creates composable reward components that can be combined
    for complex multi-objective reward shaping.
    """
    
    def __init__(self, env_width: int = 800, env_height: int = 600):
        """
        Initialize reward generator.
        
        Args:
            env_width: Environment width for normalization
            env_height: Environment height for normalization
        """
        self.env_width = env_width
        self.env_height = env_height
        self.max_distance = np.sqrt(env_width**2 + env_height**2)
    
    def generate(self, parsed_goal: ParsedGoal) -> Callable:
        """
        Generate a reward function from a parsed goal.
        
        Args:
            parsed_goal: Structured goal specification
            
        Returns:
            Callable reward function: (state, action, next_state, info) -> float
        """
        if parsed_goal.goal_type == "position":
            return self._generate_position_reward(parsed_goal)
        elif parsed_goal.goal_type == "collect":
            return self._generate_collect_reward(parsed_goal)
        elif parsed_goal.goal_type == "avoid":
            return self._generate_avoid_reward(parsed_goal)
        elif parsed_goal.goal_type == "survival":
            return self._generate_survival_reward(parsed_goal)
        else:
            return self._generate_default_reward(parsed_goal)
    
    def _generate_position_reward(self, goal: ParsedGoal) -> Callable:
        """Generate position-based reward function."""
        target = np.array(goal.target)
        goal_reward = goal.reward_config.get("goal_reward", 100.0)
        time_penalty = goal.reward_config.get("time_penalty", 0.01)
        distance_shaping = goal.reward_config.get("distance_shaping", 0.1)
        goal_radius = 30.0  # Distance to consider goal reached
        
        # State for potential-based shaping
        prev_distance = [None]
        
        def reward_fn(state: Dict, action: Any, next_state: Dict, info: Dict) -> float:
            """
            Calculate position-based reward.
            
            Reward components:
            1. Goal achievement bonus
            2. Potential-based distance shaping
            3. Time penalty
            """
            position = np.array(next_state.get("position", [0, 0]))
            distance = np.linalg.norm(position - target)
            
            reward = 0.0
            
            # Goal reached
            if distance < goal_radius:
                reward += goal_reward
                return reward
            
            # Potential-based reward shaping
            if prev_distance[0] is not None and distance_shaping > 0:
                # Reward for getting closer
                reward += (prev_distance[0] - distance) * distance_shaping
            
            prev_distance[0] = distance
            
            # Time penalty
            reward -= time_penalty
            
            return reward
        
        return reward_fn
    
    def _generate_collect_reward(self, goal: ParsedGoal) -> Callable:
        """Generate collection-based reward function."""
        items = [np.array(item) for item in goal.target]
        item_reward = goal.reward_config.get("item_reward", 10.0)
        completion_bonus = goal.reward_config.get("completion_bonus", 50.0)
        time_penalty = goal.reward_config.get("time_penalty", 0.01)
        collection_radius = 20.0
        
        # Track collected items
        collected = [False] * len(items)
        
        def reward_fn(state: Dict, action: Any, next_state: Dict, info: Dict) -> float:
            """Calculate collection-based reward."""
            position = np.array(next_state.get("position", [0, 0]))
            reward = 0.0
            
            # Check for item collection
            for i, (item, is_collected) in enumerate(zip(items, collected)):
                if is_collected:
                    continue
                
                distance = np.linalg.norm(position - item)
                if distance < collection_radius:
                    collected[i] = True
                    reward += item_reward
            
            # Completion bonus
            if all(collected):
                reward += completion_bonus
            
            # Time penalty
            reward -= time_penalty
            
            return reward
        
        return reward_fn
    
    def _generate_avoid_reward(self, goal: ParsedGoal) -> Callable:
        """Generate avoidance-based reward function."""
        survival_reward = goal.reward_config.get("survival_reward", 0.1)
        collision_penalty = goal.reward_config.get("collision_penalty", -10.0)
        danger_radius = 50.0
        
        def reward_fn(state: Dict, action: Any, next_state: Dict, info: Dict) -> float:
            """Calculate avoidance-based reward."""
            reward = survival_reward  # Base survival reward
            
            # Check for collisions
            if info.get("collision", False):
                reward += collision_penalty
            
            # Proximity penalty
            obstacles = info.get("obstacles", [])
            position = np.array(next_state.get("position", [0, 0]))
            
            for obs in obstacles:
                obs_pos = np.array(obs.get("position", [0, 0]))
                distance = np.linalg.norm(position - obs_pos)
                if distance < danger_radius:
                    # Gradual penalty for being close
                    proximity_penalty = 0.1 * (1 - distance / danger_radius)
                    reward -= proximity_penalty
            
            return reward
        
        return reward_fn
    
    def _generate_survival_reward(self, goal: ParsedGoal) -> Callable:
        """Generate survival-based reward function."""
        survival_per_step = goal.reward_config.get("survival_reward_per_step", 0.1)
        completion_bonus = goal.reward_config.get("completion_bonus", 100.0)
        target_steps = goal.conditions.get("duration_steps", 500)
        
        steps_survived = [0]
        
        def reward_fn(state: Dict, action: Any, next_state: Dict, info: Dict) -> float:
            """Calculate survival-based reward."""
            steps_survived[0] += 1
            
            reward = survival_per_step
            
            # Completion bonus
            if steps_survived[0] >= target_steps:
                reward += completion_bonus
            
            # Death penalty
            if info.get("terminated", False) and steps_survived[0] < target_steps:
                reward -= 50.0
            
            return reward
        
        return reward_fn
    
    def _generate_default_reward(self, goal: ParsedGoal) -> Callable:
        """Generate default reward function."""
        def reward_fn(state: Dict, action: Any, next_state: Dict, info: Dict) -> float:
            """Default reward: small positive for each step."""
            return 0.0
        
        return reward_fn
    
    def generate_termination_condition(self, parsed_goal: ParsedGoal) -> Callable:
        """
        Generate a termination condition from a parsed goal.
        
        Args:
            parsed_goal: Structured goal specification
            
        Returns:
            Callable: (state, info) -> bool (True if episode should terminate)
        """
        if parsed_goal.goal_type == "position":
            return self._generate_position_termination(parsed_goal)
        elif parsed_goal.goal_type == "collect":
            return self._generate_collect_termination(parsed_goal)
        elif parsed_goal.goal_type == "survival":
            return self._generate_survival_termination(parsed_goal)
        else:
            return lambda state, info: False
    
    def _generate_position_termination(self, goal: ParsedGoal) -> Callable:
        """Generate position goal termination."""
        target = np.array(goal.target)
        goal_radius = 30.0
        
        def check_termination(state: Dict, info: Dict) -> bool:
            position = np.array(state.get("position", [0, 0]))
            return np.linalg.norm(position - target) < goal_radius
        
        return check_termination
    
    def _generate_collect_termination(self, goal: ParsedGoal) -> Callable:
        """Generate collection goal termination."""
        num_items = goal.conditions.get("num_items", len(goal.target))
        
        def check_termination(state: Dict, info: Dict) -> bool:
            collected = info.get("items_collected", 0)
            return collected >= num_items
        
        return check_termination
    
    def _generate_survival_termination(self, goal: ParsedGoal) -> Callable:
        """Generate survival goal termination."""
        target_steps = goal.conditions.get("duration_steps", 500)
        
        def check_termination(state: Dict, info: Dict) -> bool:
            return info.get("steps", 0) >= target_steps
        
        return check_termination


class CompositeReward:
    """
    Combines multiple reward functions with configurable weights.
    """
    
    def __init__(self):
        """Initialize empty composite reward."""
        self.components: list = []
        self.weights: list = []
    
    def add_component(self, reward_fn: Callable, weight: float = 1.0, name: str = "") -> "CompositeReward":
        """
        Add a reward component.
        
        Args:
            reward_fn: Reward function to add
            weight: Weight for this component
            name: Optional name for logging
            
        Returns:
            self for chaining
        """
        self.components.append((name, reward_fn))
        self.weights.append(weight)
        return self
    
    def __call__(self, state: Dict, action: Any, next_state: Dict, info: Dict) -> float:
        """Calculate weighted sum of all reward components."""
        total = 0.0
        for (name, fn), weight in zip(self.components, self.weights):
            component_reward = fn(state, action, next_state, info)
            total += weight * component_reward
        return total
    
    def get_breakdown(self, state: Dict, action: Any, next_state: Dict, info: Dict) -> Dict[str, float]:
        """Get individual reward components for debugging."""
        breakdown = {}
        for (name, fn), weight in zip(self.components, self.weights):
            component_reward = fn(state, action, next_state, info)
            breakdown[name or f"component_{len(breakdown)}"] = {
                "raw": component_reward,
                "weighted": weight * component_reward,
            }
        breakdown["total"] = self(state, action, next_state, info)
        return breakdown

