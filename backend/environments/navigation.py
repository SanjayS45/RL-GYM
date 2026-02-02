"""
Navigation Environment for RL-GYM.

A continuous environment where agents navigate around obstacles to reach goals.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEnvironment
from .obstacles import Block, Wall, create_obstacle_from_config
from .goals import PositionGoal, CollectGoal, create_goal_from_config


class NavigationEnv(BaseEnvironment):
    """
    Continuous Navigation Environment.
    
    Agent must navigate through a 2D space with obstacles to reach goals.
    Features continuous state and action spaces.
    
    Observation: [agent_x, agent_y, agent_vx, agent_vy, goal_dx, goal_dy, ...]
    Action: [acceleration_x, acceleration_y] (continuous)
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        max_steps: int = 500,
        max_speed: float = 200.0,
        max_acceleration: float = 100.0,
        goal_positions: Optional[List[Tuple[float, float]]] = None,
        obstacle_configs: Optional[List[Dict]] = None,
        observation_type: str = "state",  # "state" or "lidar"
        lidar_rays: int = 16,
        lidar_range: float = 200.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Navigation Environment.
        
        Args:
            width: Environment width
            height: Environment height
            max_steps: Maximum steps per episode
            max_speed: Maximum agent speed
            max_acceleration: Maximum acceleration
            goal_positions: List of goal positions
            obstacle_configs: List of obstacle configurations
            observation_type: "state" for full state, "lidar" for ray-based
            lidar_rays: Number of lidar rays (if observation_type="lidar")
            lidar_range: Lidar sensing range
            render_mode: Rendering mode
            seed: Random seed
        """
        super().__init__(width, height, max_steps, render_mode, seed)
        
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.observation_type = observation_type
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        
        # Set up physics with no gravity
        self.physics.gravity = 0
        self.physics.friction = 0.05
        
        # Add obstacles
        if obstacle_configs:
            for config in obstacle_configs:
                self.add_obstacle(create_obstacle_from_config(config))
        
        # Add goals
        if goal_positions:
            for pos in goal_positions:
                self.add_goal(PositionGoal(
                    position=np.array(pos),
                    size=np.array([40, 40]),
                    reward=100.0,
                    distance_reward_scale=0.1
                ))
        else:
            # Default goal
            self.add_goal(PositionGoal(
                position=np.array([width * 0.8, height * 0.5]),
                size=np.array([40, 40]),
                reward=100.0,
                distance_reward_scale=0.1
            ))
        
        # Define observation space
        if observation_type == "lidar":
            # Lidar readings + velocity + relative goal position
            obs_dim = lidar_rays + 4  # rays + vx, vy + goal_dx, goal_dy
            self.observation_space = spaces.Box(
                low=-1, high=1,
                shape=(obs_dim,),
                dtype=np.float32
            )
        else:
            # Full state: pos, vel, goal_rel, obstacle_distances (top 3)
            obs_dim = 2 + 2 + 2 + 6  # pos + vel + goal_rel + 3 nearest obstacles
            self.observation_space = spaces.Box(
                low=-1, high=1,
                shape=(obs_dim,),
                dtype=np.float32
            )
        
        # Continuous action space
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(2,),
            dtype=np.float32
        )
    
    def _get_lidar_obs(self) -> np.ndarray:
        """Get lidar-based observation."""
        readings = np.ones(self.lidar_rays, dtype=np.float32)  # 1 = max range
        
        for i in range(self.lidar_rays):
            angle = 2 * np.pi * i / self.lidar_rays
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Cast ray
            min_dist = self.lidar_range
            for obs in self.obstacles:
                dist = self._ray_obstacle_intersection(
                    self.agent_position,
                    direction,
                    obs.position,
                    obs.size
                )
                if dist is not None and dist < min_dist:
                    min_dist = dist
            
            # Also check walls (environment bounds)
            wall_dist = self._ray_wall_intersection(self.agent_position, direction)
            if wall_dist < min_dist:
                min_dist = wall_dist
            
            readings[i] = min_dist / self.lidar_range  # Normalize to [0, 1]
        
        # Add velocity and goal info
        vel_norm = self.agent_velocity / self.max_speed
        goal_rel = (self.goals[0].position - self.agent_position) / np.array([self.width, self.height])
        
        return np.concatenate([readings, vel_norm, goal_rel]).astype(np.float32)
    
    def _ray_obstacle_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        obs_pos: np.ndarray,
        obs_size: np.ndarray
    ) -> Optional[float]:
        """Calculate ray-AABB intersection distance."""
        half = obs_size / 2
        min_corner = obs_pos - half
        max_corner = obs_pos + half
        
        t_min = 0.0
        t_max = self.lidar_range
        
        for i in range(2):
            if abs(direction[i]) < 1e-8:
                if origin[i] < min_corner[i] or origin[i] > max_corner[i]:
                    return None
            else:
                t1 = (min_corner[i] - origin[i]) / direction[i]
                t2 = (max_corner[i] - origin[i]) / direction[i]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                
                if t_min > t_max:
                    return None
        
        return t_min if t_min > 0 else None
    
    def _ray_wall_intersection(self, origin: np.ndarray, direction: np.ndarray) -> float:
        """Calculate ray intersection with environment walls."""
        min_dist = self.lidar_range
        
        # Check each wall
        walls = [
            (np.array([0, 0]), np.array([0, 1])),  # Left
            (np.array([self.width, 0]), np.array([0, 1])),  # Right
            (np.array([0, 0]), np.array([1, 0])),  # Top
            (np.array([0, self.height]), np.array([1, 0])),  # Bottom
        ]
        
        for wall_point, wall_dir in walls:
            # Line-line intersection
            denom = direction[0] * wall_dir[1] - direction[1] * wall_dir[0]
            if abs(denom) < 1e-8:
                continue
            
            t = ((wall_point[0] - origin[0]) * wall_dir[1] - 
                 (wall_point[1] - origin[1]) * wall_dir[0]) / denom
            
            if 0 < t < min_dist:
                min_dist = t
        
        return min_dist
    
    def _get_state_obs(self) -> np.ndarray:
        """Get full state observation."""
        # Normalize positions to [-1, 1]
        pos_norm = 2 * self.agent_position / np.array([self.width, self.height]) - 1
        vel_norm = self.agent_velocity / self.max_speed
        
        # Goal relative position
        goal_rel = (self.goals[0].position - self.agent_position) / np.array([self.width, self.height])
        
        # Nearest obstacles (distance and relative direction)
        obstacle_info = []
        for obs in sorted(self.obstacles, key=lambda o: np.linalg.norm(self.agent_position - o.position))[:3]:
            rel_pos = (obs.position - self.agent_position) / np.array([self.width, self.height])
            obstacle_info.extend(rel_pos.tolist())
        
        # Pad if fewer than 3 obstacles
        while len(obstacle_info) < 6:
            obstacle_info.extend([0, 0])
        
        return np.concatenate([pos_norm, vel_norm, goal_rel, obstacle_info[:6]]).astype(np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation based on observation type."""
        if self.observation_type == "lidar":
            return self._get_lidar_obs()
        else:
            return self._get_state_obs()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "position": self.agent_position.tolist(),
            "velocity": self.agent_velocity.tolist(),
            "distance_to_goal": float(np.linalg.norm(self.agent_position - self.goals[0].position)),
            "step": self.current_step,
            "total_reward": self.total_reward,
        }
    
    def _action_to_acceleration(self, action: np.ndarray) -> np.ndarray:
        """Convert action to acceleration."""
        # Clip and scale action
        action = np.clip(action, -1, 1)
        return action * self.max_acceleration
    
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        reward = 0.0
        
        # Goal rewards
        for goal in self.goals:
            reward += goal.get_reward(self.agent_position)
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        # Penalty for hitting walls
        if (self.agent_position[0] <= 0 or self.agent_position[0] >= self.width or
            self.agent_position[1] <= 0 or self.agent_position[1] >= self.height):
            reward -= 0.5
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if all goals are achieved."""
        return all(goal.achieved for goal in self.goals)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step with velocity clamping."""
        self.current_step += 1
        
        # Apply action
        acceleration = self._action_to_acceleration(action)
        self.agent_position, self.agent_velocity = self.physics.update_position(
            self.agent_position,
            self.agent_velocity,
            acceleration,
            apply_gravity=False
        )
        
        # Clamp velocity
        speed = np.linalg.norm(self.agent_velocity)
        if speed > self.max_speed:
            self.agent_velocity = self.agent_velocity / speed * self.max_speed
        
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


# Predefined navigation configurations
NAVIGATION_CONFIGS = {
    "empty": {
        "goal_positions": [(700, 300)],
        "obstacle_configs": [],
    },
    "simple_obstacles": {
        "goal_positions": [(700, 300)],
        "obstacle_configs": [
            {"type": "block", "x": 300, "y": 200, "width": 100, "height": 200},
            {"type": "block", "x": 500, "y": 400, "width": 150, "height": 100},
        ],
    },
    "maze_like": {
        "goal_positions": [(750, 550)],
        "obstacle_configs": [
            {"type": "wall", "start": (200, 0), "end": (200, 400)},
            {"type": "wall", "start": (400, 200), "end": (400, 600)},
            {"type": "wall", "start": (600, 0), "end": (600, 400)},
            {"type": "block", "x": 300, "y": 500, "width": 50, "height": 100},
        ],
    },
    "cluttered": {
        "goal_positions": [(750, 300)],
        "obstacle_configs": [
            {"type": "block", "x": 200, "y": 150, "width": 60, "height": 60},
            {"type": "block", "x": 350, "y": 250, "width": 80, "height": 40},
            {"type": "block", "x": 250, "y": 400, "width": 50, "height": 80},
            {"type": "block", "x": 450, "y": 350, "width": 70, "height": 70},
            {"type": "block", "x": 550, "y": 150, "width": 60, "height": 100},
            {"type": "block", "x": 500, "y": 450, "width": 90, "height": 50},
            {"type": "block", "x": 650, "y": 250, "width": 40, "height": 120},
        ],
    },
}


def create_navigation_env(config_name: str = "simple_obstacles", **kwargs) -> NavigationEnv:
    """Create a navigation environment from predefined configuration."""
    if config_name in NAVIGATION_CONFIGS:
        config = NAVIGATION_CONFIGS[config_name].copy()
        config.update(kwargs)
        return NavigationEnv(**config)
    else:
        return NavigationEnv(**kwargs)

