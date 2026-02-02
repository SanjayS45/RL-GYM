"""
Grid World Environment for RL-GYM.

A simple discrete environment where agents navigate a grid.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEnvironment
from .obstacles import Block
from .goals import PositionGoal


class GridWorldEnv(BaseEnvironment):
    """
    Grid World Environment.
    
    A discrete grid where the agent must navigate to a goal.
    Classic RL environment for testing algorithms.
    
    Observation: One-hot encoded position or grid state
    Action: Discrete (up, down, left, right)
    """
    
    def __init__(
        self,
        grid_size: int = 10,
        cell_size: int = 50,
        max_steps: int = 200,
        obstacle_positions: Optional[List[Tuple[int, int]]] = None,
        goal_position: Optional[Tuple[int, int]] = None,
        start_position: Optional[Tuple[int, int]] = None,
        stochastic: bool = False,
        slip_prob: float = 0.1,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Grid World.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            cell_size: Pixel size of each cell
            max_steps: Maximum steps per episode
            obstacle_positions: List of (row, col) obstacle positions
            goal_position: (row, col) goal position (default: bottom-right)
            start_position: (row, col) start position (default: top-left)
            stochastic: If True, actions may slip
            slip_prob: Probability of slipping to adjacent direction
            render_mode: Rendering mode
            seed: Random seed
        """
        super().__init__(
            width=grid_size * cell_size,
            height=grid_size * cell_size,
            max_steps=max_steps,
            render_mode=render_mode,
            seed=seed
        )
        
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.stochastic = stochastic
        self.slip_prob = slip_prob
        
        # Positions
        self.start_position = start_position or (0, 0)
        self.goal_position = goal_position or (grid_size - 1, grid_size - 1)
        
        # Grid state (0 = empty, 1 = obstacle, 2 = goal, 3 = agent)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        
        # Add obstacles
        if obstacle_positions:
            for row, col in obstacle_positions:
                self.grid[row, col] = 1
                # Create visual obstacle
                self.add_obstacle(Block(
                    x=(col + 0.5) * cell_size,
                    y=(row + 0.5) * cell_size,
                    width=cell_size - 4,
                    height=cell_size - 4
                ))
        
        # Mark goal
        self.grid[self.goal_position[0], self.goal_position[1]] = 2
        
        # Create goal object
        self.add_goal(PositionGoal(
            position=np.array([
                (self.goal_position[1] + 0.5) * cell_size,
                (self.goal_position[0] + 0.5) * cell_size
            ]),
            size=np.array([cell_size - 4, cell_size - 4]),
            reward=100.0,
            distance_reward_scale=0.0,  # Sparse reward
            sparse_reward=True
        ))
        
        # Agent grid position
        self.agent_row = self.start_position[0]
        self.agent_col = self.start_position[1]
        
        # Spaces
        # Observation: flattened one-hot position
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(grid_size * grid_size,),
            dtype=np.float32
        )
        
        # Action: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Action deltas (row, col)
        self._action_deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1),   # Right
        }
    
    def _get_initial_position(self) -> np.ndarray:
        """Get initial agent position in pixels."""
        return np.array([
            (self.start_position[1] + 0.5) * self.cell_size,
            (self.start_position[0] + 0.5) * self.cell_size
        ], dtype=np.float32)
    
    def _grid_to_pixel(self, row: int, col: int) -> np.ndarray:
        """Convert grid position to pixel position."""
        return np.array([
            (col + 0.5) * self.cell_size,
            (row + 0.5) * self.cell_size
        ], dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get one-hot encoded position."""
        obs = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        idx = self.agent_row * self.grid_size + self.agent_col
        obs[idx] = 1.0
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "agent_position": (self.agent_row, self.agent_col),
            "goal_position": self.goal_position,
            "step": self.current_step,
            "total_reward": self.total_reward,
        }
    
    def _action_to_acceleration(self, action: int) -> np.ndarray:
        """Convert discrete action to movement (not used in grid world)."""
        return np.zeros(2, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate step reward."""
        # Check if at goal
        if (self.agent_row, self.agent_col) == self.goal_position:
            return 100.0
        return -0.1  # Small negative reward per step
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate."""
        return (self.agent_row, self.agent_col) == self.goal_position
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is valid (within bounds and not obstacle)."""
        if row < 0 or row >= self.grid_size:
            return False
        if col < 0 or col >= self.grid_size:
            return False
        if self.grid[row, col] == 1:  # Obstacle
            return False
        return True
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        if seed is not None:
            self.seed(seed)
        
        self.current_step = 0
        self.total_reward = 0
        
        # Reset agent position
        self.agent_row = self.start_position[0]
        self.agent_col = self.start_position[1]
        self.agent_position = self._grid_to_pixel(self.agent_row, self.agent_col)
        
        # Reset goals
        for goal in self.goals:
            goal.reset()
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
        """
        self.current_step += 1
        
        # Handle stochastic transitions
        if self.stochastic and np.random.random() < self.slip_prob:
            # Slip to adjacent action
            action = (action + np.random.choice([-1, 1])) % 4
        
        # Get movement delta
        delta = self._action_deltas[action]
        new_row = self.agent_row + delta[0]
        new_col = self.agent_col + delta[1]
        
        # Check if valid
        if self._is_valid_position(new_row, new_col):
            self.agent_row = new_row
            self.agent_col = new_col
        
        # Update pixel position
        self.agent_position = self._grid_to_pixel(self.agent_row, self.agent_col)
        
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
            "grid_size": self.grid_size,
            "cell_size": self.cell_size,
            "grid": self.grid.tolist(),
            "agent_grid_pos": (self.agent_row, self.agent_col),
            "goal_grid_pos": self.goal_position,
        })
        return state
    
    def _render_frame(self) -> np.ndarray:
        """Render environment as RGB array."""
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            # Horizontal lines
            y = i * self.cell_size
            if y < self.height:
                frame[y:y+1, :] = [200, 200, 200]
            # Vertical lines
            x = i * self.cell_size
            if x < self.width:
                frame[:, x:x+1] = [200, 200, 200]
        
        # Draw cells
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * self.cell_size + 2
                y1 = row * self.cell_size + 2
                x2 = (col + 1) * self.cell_size - 2
                y2 = (row + 1) * self.cell_size - 2
                
                if self.grid[row, col] == 1:  # Obstacle
                    frame[y1:y2, x1:x2] = [80, 80, 80]
                elif (row, col) == self.goal_position:  # Goal
                    frame[y1:y2, x1:x2] = [0, 200, 0]
        
        # Draw agent
        x1 = self.agent_col * self.cell_size + 5
        y1 = self.agent_row * self.cell_size + 5
        x2 = (self.agent_col + 1) * self.cell_size - 5
        y2 = (self.agent_row + 1) * self.cell_size - 5
        frame[y1:y2, x1:x2] = [50, 50, 200]
        
        return frame


# Predefined grid world configurations
GRID_WORLD_CONFIGS = {
    "simple": {
        "grid_size": 5,
        "obstacle_positions": [(1, 1), (2, 2), (3, 3)],
        "goal_position": (4, 4),
        "start_position": (0, 0),
    },
    "maze": {
        "grid_size": 8,
        "obstacle_positions": [
            (1, 1), (1, 2), (1, 3), (1, 5), (1, 6),
            (3, 1), (3, 3), (3, 4), (3, 5),
            (5, 2), (5, 3), (5, 5), (5, 6),
            (6, 1), (6, 6),
        ],
        "goal_position": (7, 7),
        "start_position": (0, 0),
    },
    "cliff": {
        "grid_size": 6,
        "obstacle_positions": [(5, 1), (5, 2), (5, 3), (5, 4)],  # Cliff
        "goal_position": (5, 5),
        "start_position": (5, 0),
    },
}


def create_grid_world(config_name: str = "simple", **kwargs) -> GridWorldEnv:
    """Create a grid world from predefined configuration."""
    if config_name in GRID_WORLD_CONFIGS:
        config = GRID_WORLD_CONFIGS[config_name].copy()
        config.update(kwargs)
        return GridWorldEnv(**config)
    else:
        return GridWorldEnv(**kwargs)

