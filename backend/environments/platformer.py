"""
Platformer Environment for RL-GYM.

A physics-based platformer where agents jump and navigate platforms.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEnvironment
from .obstacles import Platform, Block, Ramp, MovingPlatform
from .goals import PositionGoal


class PlatformerEnv(BaseEnvironment):
    """
    Platformer Environment with gravity and jumping.
    
    Agent must navigate platforms, jump over gaps, and reach goals.
    Features gravity, jumping, and platform collision physics.
    
    Observation: [x, y, vx, vy, on_ground, goal_dx, goal_dy, nearby_platforms...]
    Action: Discrete [noop, left, right, jump, jump_left, jump_right]
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        max_steps: int = 1000,
        gravity: float = 500.0,
        jump_strength: float = 300.0,
        move_speed: float = 150.0,
        air_control: float = 0.3,
        platform_configs: Optional[List[Dict]] = None,
        goal_position: Optional[Tuple[float, float]] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Platformer Environment.
        
        Args:
            width: Environment width
            height: Environment height
            max_steps: Maximum steps per episode
            gravity: Gravity acceleration
            jump_strength: Initial jump velocity
            move_speed: Horizontal movement speed
            air_control: Control factor while in air (0-1)
            platform_configs: List of platform configurations
            goal_position: Goal position (x, y)
            render_mode: Rendering mode
            seed: Random seed
        """
        super().__init__(width, height, max_steps, render_mode, seed)
        
        self.gravity = gravity
        self.jump_strength = jump_strength
        self.move_speed = move_speed
        self.air_control = air_control
        
        # Physics settings for platformer
        self.physics.gravity = gravity
        self.physics.friction = 0.2
        
        # Agent state
        self.on_ground = False
        self.can_jump = True
        
        # Add ground platform
        self.add_obstacle(Platform(
            x=width / 2,
            y=height - 10,
            width=width,
            height=20,
            one_way=False
        ))
        
        # Add custom platforms
        if platform_configs:
            for config in platform_configs:
                self._add_platform_from_config(config)
        
        # Add goal
        if goal_position:
            goal_pos = np.array(goal_position)
        else:
            goal_pos = np.array([width - 50, height - 100])
        
        self.add_goal(PositionGoal(
            position=goal_pos,
            size=np.array([40, 40]),
            reward=100.0,
            distance_reward_scale=0.05
        ))
        
        # Observation space
        # [x, y, vx, vy, on_ground] + [goal_dx, goal_dy] + [4 nearest platform info (8 values)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(15,),
            dtype=np.float32
        )
        
        # Action space: 0=noop, 1=left, 2=right, 3=jump, 4=jump_left, 5=jump_right
        self.action_space = spaces.Discrete(6)
    
    def _add_platform_from_config(self, config: Dict) -> None:
        """Add a platform from configuration."""
        ptype = config.get("type", "platform")
        
        if ptype == "platform":
            self.add_obstacle(Platform(
                x=config["x"],
                y=config["y"],
                width=config["width"],
                height=config.get("height", 15),
                one_way=config.get("one_way", True)
            ))
        elif ptype == "block":
            self.add_obstacle(Block(
                x=config["x"],
                y=config["y"],
                width=config["width"],
                height=config["height"]
            ))
        elif ptype == "moving":
            self.add_obstacle(MovingPlatform(
                start_pos=(config["start_x"], config["start_y"]),
                end_pos=(config["end_x"], config["end_y"]),
                width=config["width"],
                speed=config.get("speed", 50)
            ))
    
    def _get_initial_position(self) -> np.ndarray:
        """Get initial position on ground."""
        return np.array([100, self.height - 50], dtype=np.float32)
    
    def _check_on_ground(self) -> bool:
        """Check if agent is standing on a platform."""
        agent_bottom = self.agent_position[1] + self.agent_size[1] / 2
        
        for obstacle in self.obstacles:
            if isinstance(obstacle, (Platform, Block)):
                obs_top = obstacle.position[1] - obstacle.size[1] / 2
                
                # Check if agent is above and touching platform
                if (abs(agent_bottom - obs_top) < 5 and
                    abs(self.agent_position[0] - obstacle.position[0]) < 
                    (self.agent_size[0] + obstacle.size[0]) / 2):
                    
                    # For one-way platforms, only count if moving down or stationary
                    if isinstance(obstacle, Platform) and obstacle.one_way:
                        if self.agent_velocity[1] >= 0:
                            return True
                    else:
                        return True
        
        return False
    
    def _get_obs(self) -> np.ndarray:
        """Get platformer observation."""
        # Normalize positions
        pos_x = self.agent_position[0] / self.width
        pos_y = self.agent_position[1] / self.height
        vel_x = self.agent_velocity[0] / self.move_speed
        vel_y = self.agent_velocity[1] / self.jump_strength
        
        # Goal relative
        goal_dx = (self.goals[0].position[0] - self.agent_position[0]) / self.width
        goal_dy = (self.goals[0].position[1] - self.agent_position[1]) / self.height
        
        # Nearest platforms
        platform_info = []
        sorted_platforms = sorted(
            self.obstacles,
            key=lambda p: np.linalg.norm(self.agent_position - p.position)
        )[:4]
        
        for platform in sorted_platforms:
            rel_x = (platform.position[0] - self.agent_position[0]) / self.width
            rel_y = (platform.position[1] - self.agent_position[1]) / self.height
            platform_info.extend([rel_x, rel_y])
        
        # Pad if fewer platforms
        while len(platform_info) < 8:
            platform_info.extend([0, 0])
        
        return np.array([
            pos_x, pos_y, vel_x, vel_y, float(self.on_ground),
            goal_dx, goal_dy,
            *platform_info[:8]
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "position": self.agent_position.tolist(),
            "velocity": self.agent_velocity.tolist(),
            "on_ground": self.on_ground,
            "distance_to_goal": float(np.linalg.norm(self.agent_position - self.goals[0].position)),
            "step": self.current_step,
            "total_reward": self.total_reward,
        }
    
    def _action_to_acceleration(self, action: int) -> np.ndarray:
        """Convert discrete action to acceleration."""
        ax = 0.0
        ay = 0.0
        
        # Horizontal movement
        if action in [1, 4]:  # Left
            ax = -self.move_speed
        elif action in [2, 5]:  # Right
            ax = self.move_speed
        
        # Reduce air control
        if not self.on_ground:
            ax *= self.air_control
        
        return np.array([ax, ay], dtype=np.float32)
    
    def _use_gravity(self) -> bool:
        """Enable gravity for platformer."""
        return True
    
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        reward = 0.0
        
        # Goal progress reward
        for goal in self.goals:
            reward += goal.get_reward(self.agent_position)
        
        # Height bonus (encourage climbing)
        height_progress = (self.height - self.agent_position[1]) / self.height
        reward += 0.001 * height_progress
        
        # Small step penalty
        reward -= 0.01
        
        # Fall penalty
        if self.agent_position[1] >= self.height - 20:
            reward -= 0.1
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if goal reached."""
        return self.goals[0].achieved
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a platformer step."""
        self.current_step += 1
        
        # Update moving platforms
        for obstacle in self.obstacles:
            if isinstance(obstacle, MovingPlatform):
                obstacle.update(self.physics.dt)
        
        # Check ground state before action
        self.on_ground = self._check_on_ground()
        
        # Handle jumping
        if action in [3, 4, 5] and self.on_ground and self.can_jump:
            self.agent_velocity[1] = -self.jump_strength
            self.can_jump = False
        
        # Reset jump ability when landing
        if self.on_ground and not self.can_jump and action not in [3, 4, 5]:
            self.can_jump = True
        
        # Apply horizontal acceleration
        acceleration = self._action_to_acceleration(action)
        
        # Update physics
        self.agent_position, self.agent_velocity = self.physics.update_position(
            self.agent_position,
            self.agent_velocity,
            acceleration,
            apply_gravity=True
        )
        
        # Platform collisions
        for obstacle in self.obstacles:
            if isinstance(obstacle, Platform):
                if obstacle.one_way:
                    if obstacle.should_collide(self.agent_position, self.agent_velocity):
                        self.agent_position, self.agent_velocity = self.physics.resolve_collision(
                            self.agent_position,
                            self.agent_velocity,
                            self.agent_size,
                            obstacle.position,
                            obstacle.size
                        )
                else:
                    self.agent_position, self.agent_velocity = self.physics.resolve_collision(
                        self.agent_position,
                        self.agent_velocity,
                        self.agent_size,
                        obstacle.position,
                        obstacle.size
                    )
            elif isinstance(obstacle, Block):
                self.agent_position, self.agent_velocity = self.physics.resolve_collision(
                    self.agent_position,
                    self.agent_velocity,
                    self.agent_size,
                    obstacle.position,
                    obstacle.size
                )
        
        # Update ground state after physics
        self.on_ground = self._check_on_ground()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # Check termination
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def get_render_state(self) -> Dict[str, Any]:
        """Get state for rendering."""
        state = super().get_render_state()
        state.update({
            "on_ground": self.on_ground,
            "can_jump": self.can_jump,
        })
        return state


# Predefined platformer configurations
PLATFORMER_CONFIGS = {
    "simple": {
        "platform_configs": [
            {"type": "platform", "x": 200, "y": 500, "width": 150},
            {"type": "platform", "x": 400, "y": 400, "width": 150},
            {"type": "platform", "x": 600, "y": 300, "width": 150},
        ],
        "goal_position": (700, 250),
    },
    "climbing": {
        "platform_configs": [
            {"type": "platform", "x": 150, "y": 520, "width": 100},
            {"type": "platform", "x": 300, "y": 450, "width": 100},
            {"type": "platform", "x": 150, "y": 380, "width": 100},
            {"type": "platform", "x": 300, "y": 310, "width": 100},
            {"type": "platform", "x": 150, "y": 240, "width": 100},
            {"type": "platform", "x": 300, "y": 170, "width": 100},
            {"type": "platform", "x": 450, "y": 100, "width": 200},
        ],
        "goal_position": (500, 50),
    },
    "gaps": {
        "platform_configs": [
            {"type": "block", "x": 200, "y": 550, "width": 200, "height": 100},
            {"type": "block", "x": 500, "y": 550, "width": 200, "height": 100},
            {"type": "platform", "x": 350, "y": 450, "width": 80},
        ],
        "goal_position": (600, 480),
    },
    "moving_platforms": {
        "platform_configs": [
            {"type": "platform", "x": 150, "y": 500, "width": 100},
            {"type": "moving", "start_x": 300, "start_y": 400, "end_x": 500, "end_y": 400, "width": 100},
            {"type": "moving", "start_x": 600, "start_y": 300, "end_x": 600, "end_y": 500, "width": 100},
            {"type": "platform", "x": 700, "y": 200, "width": 150},
        ],
        "goal_position": (750, 150),
    },
}


def create_platformer_env(config_name: str = "simple", **kwargs) -> PlatformerEnv:
    """Create a platformer environment from predefined configuration."""
    if config_name in PLATFORMER_CONFIGS:
        config = PLATFORMER_CONFIGS[config_name].copy()
        config.update(kwargs)
        return PlatformerEnv(**config)
    else:
        return PlatformerEnv(**kwargs)

