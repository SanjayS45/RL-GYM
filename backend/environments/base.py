"""
Base environment and physics engine for RL-GYM.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PhysicsEngine:
    """
    Simple 2D physics engine for agent movement.
    
    Handles position, velocity, acceleration, and collision detection.
    """
    
    def __init__(
        self,
        gravity: float = 9.8,
        friction: float = 0.1,
        dt: float = 0.02,
        bounds: Tuple[float, float, float, float] = (0, 0, 100, 100)
    ):
        """
        Initialize physics engine.
        
        Args:
            gravity: Gravity acceleration (pixels/s^2)
            friction: Friction coefficient
            dt: Time step
            bounds: World bounds (x_min, y_min, x_max, y_max)
        """
        self.gravity = gravity
        self.friction = friction
        self.dt = dt
        self.bounds = bounds
    
    def update_position(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        apply_gravity: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update position and velocity using Verlet integration.
        
        Args:
            position: Current position [x, y]
            velocity: Current velocity [vx, vy]
            acceleration: Applied acceleration [ax, ay]
            apply_gravity: Whether to apply gravity
            
        Returns:
            new_position: Updated position
            new_velocity: Updated velocity
        """
        # Apply gravity
        if apply_gravity:
            acceleration = acceleration.copy()
            acceleration[1] += self.gravity
        
        # Apply friction
        velocity = velocity * (1 - self.friction)
        
        # Update velocity
        new_velocity = velocity + acceleration * self.dt
        
        # Update position
        new_position = position + new_velocity * self.dt
        
        # Clamp to bounds
        new_position[0] = np.clip(new_position[0], self.bounds[0], self.bounds[2])
        new_position[1] = np.clip(new_position[1], self.bounds[1], self.bounds[3])
        
        # Stop velocity at bounds
        if new_position[0] == self.bounds[0] or new_position[0] == self.bounds[2]:
            new_velocity[0] = 0
        if new_position[1] == self.bounds[1] or new_position[1] == self.bounds[3]:
            new_velocity[1] = 0
        
        return new_position, new_velocity
    
    def check_collision(
        self,
        pos1: np.ndarray,
        size1: np.ndarray,
        pos2: np.ndarray,
        size2: np.ndarray
    ) -> bool:
        """
        Check AABB collision between two rectangles.
        
        Args:
            pos1: Position of first object (center)
            size1: Size of first object (width, height)
            pos2: Position of second object (center)
            size2: Size of second object (width, height)
            
        Returns:
            True if collision detected
        """
        half1 = size1 / 2
        half2 = size2 / 2
        
        return (abs(pos1[0] - pos2[0]) < half1[0] + half2[0] and
                abs(pos1[1] - pos2[1]) < half1[1] + half2[1])
    
    def resolve_collision(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        agent_size: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_size: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve collision by pushing agent out of obstacle.
        
        Returns:
            new_position: Adjusted position
            new_velocity: Adjusted velocity
        """
        if not self.check_collision(position, agent_size, obstacle_pos, obstacle_size):
            return position, velocity
        
        # Calculate overlap on each axis
        half_agent = agent_size / 2
        half_obs = obstacle_size / 2
        
        dx = position[0] - obstacle_pos[0]
        dy = position[1] - obstacle_pos[1]
        
        overlap_x = half_agent[0] + half_obs[0] - abs(dx)
        overlap_y = half_agent[1] + half_obs[1] - abs(dy)
        
        # Push out along axis with smallest overlap
        new_position = position.copy()
        new_velocity = velocity.copy()
        
        if overlap_x < overlap_y:
            new_position[0] += np.sign(dx) * overlap_x
            new_velocity[0] = 0
        else:
            new_position[1] += np.sign(dy) * overlap_y
            new_velocity[1] = 0
        
        return new_position, new_velocity


class BaseEnvironment(gym.Env, ABC):
    """
    Abstract base class for RL-GYM environments.
    
    Provides standard Gym interface with additional features:
    - Physics simulation
    - Obstacle management
    - Goal tracking
    - Rendering state for visualization
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "state"]}
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize base environment.
        
        Args:
            width: Environment width in pixels
            height: Environment height in pixels
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
            seed: Random seed
        """
        super().__init__()
        
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Physics
        self.physics = PhysicsEngine(
            bounds=(0, 0, width, height)
        )
        
        # State tracking
        self.current_step = 0
        self.total_reward = 0
        
        # Objects in environment
        self.obstacles: List[Any] = []
        self.goals: List[Any] = []
        
        # Agent state
        self.agent_position = np.zeros(2, dtype=np.float32)
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        self.agent_size = np.array([20, 20], dtype=np.float32)
        
        # Set seed
        if seed is not None:
            self.seed(seed)
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        self._np_random = np.random.default_rng(seed)
    
    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        pass
    
    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        pass
    
    def add_obstacle(self, obstacle: Any) -> None:
        """Add an obstacle to the environment."""
        self.obstacles.append(obstacle)
    
    def add_goal(self, goal: Any) -> None:
        """Add a goal to the environment."""
        self.goals.append(goal)
    
    def clear_obstacles(self) -> None:
        """Remove all obstacles."""
        self.obstacles = []
    
    def clear_goals(self) -> None:
        """Remove all goals."""
        self.goals = []
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        self.total_reward = 0
        
        # Reset agent
        self.agent_position = self._get_initial_position()
        self.agent_velocity = np.zeros(2, dtype=np.float32)
        
        # Reset goals
        for goal in self.goals:
            goal.reset()
        
        return self._get_obs(), self._get_info()
    
    def _get_initial_position(self) -> np.ndarray:
        """Get initial agent position."""
        return np.array([self.width / 4, self.height / 2], dtype=np.float32)
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Step reward
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional info
        """
        self.current_step += 1
        
        # Apply action and update physics
        acceleration = self._action_to_acceleration(action)
        self.agent_position, self.agent_velocity = self.physics.update_position(
            self.agent_position,
            self.agent_velocity,
            acceleration,
            apply_gravity=self._use_gravity()
        )
        
        # Check obstacle collisions
        for obstacle in self.obstacles:
            self.agent_position, self.agent_velocity = self.physics.resolve_collision(
                self.agent_position,
                self.agent_velocity,
                self.agent_size,
                obstacle.position,
                obstacle.size
            )
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    @abstractmethod
    def _action_to_acceleration(self, action: Any) -> np.ndarray:
        """Convert action to acceleration vector."""
        pass
    
    def _use_gravity(self) -> bool:
        """Whether to apply gravity."""
        return False
    
    @abstractmethod
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        pass
    
    @abstractmethod
    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        pass
    
    def get_render_state(self) -> Dict[str, Any]:
        """
        Get environment state for rendering.
        
        Returns dict with all info needed for visualization.
        """
        return {
            "width": self.width,
            "height": self.height,
            "agent": {
                "position": self.agent_position.tolist(),
                "velocity": self.agent_velocity.tolist(),
                "size": self.agent_size.tolist(),
            },
            "obstacles": [obs.to_dict() for obs in self.obstacles],
            "goals": [goal.to_dict() for goal in self.goals],
            "step": self.current_step,
            "total_reward": self.total_reward,
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "state":
            return self.get_render_state()
        elif self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray:
        """Render environment as RGB array."""
        # Create blank canvas
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw obstacles (gray)
        for obs in self.obstacles:
            x1 = int(obs.position[0] - obs.size[0] / 2)
            y1 = int(obs.position[1] - obs.size[1] / 2)
            x2 = int(obs.position[0] + obs.size[0] / 2)
            y2 = int(obs.position[1] + obs.size[1] / 2)
            frame[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)] = [100, 100, 100]
        
        # Draw goals (green)
        for goal in self.goals:
            if not goal.achieved:
                x1 = int(goal.position[0] - goal.size[0] / 2)
                y1 = int(goal.position[1] - goal.size[1] / 2)
                x2 = int(goal.position[0] + goal.size[0] / 2)
                y2 = int(goal.position[1] + goal.size[1] / 2)
                frame[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)] = [0, 200, 0]
        
        # Draw agent (blue)
        x1 = int(self.agent_position[0] - self.agent_size[0] / 2)
        y1 = int(self.agent_position[1] - self.agent_size[1] / 2)
        x2 = int(self.agent_position[0] + self.agent_size[0] / 2)
        y2 = int(self.agent_position[1] + self.agent_size[1] / 2)
        frame[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)] = [50, 50, 200]
        
        return frame

