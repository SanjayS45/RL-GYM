"""
Obstacle definitions for RL-GYM environments.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Obstacle:
    """
    Base obstacle class.
    
    Obstacles are static objects that block agent movement.
    """
    
    position: np.ndarray  # Center position [x, y]
    size: np.ndarray  # Size [width, height]
    obstacle_type: str = "generic"
    color: Tuple[int, int, int] = (100, 100, 100)
    solid: bool = True
    
    def __post_init__(self):
        """Convert lists to numpy arrays if needed."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.size, np.ndarray):
            self.size = np.array(self.size, dtype=np.float32)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the obstacle."""
        half_size = self.size / 2
        return (abs(point[0] - self.position[0]) < half_size[0] and
                abs(point[1] - self.position[1]) < half_size[1])
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get obstacle bounds (x_min, y_min, x_max, y_max)."""
        half = self.size / 2
        return (
            self.position[0] - half[0],
            self.position[1] - half[1],
            self.position[0] + half[0],
            self.position[1] + half[1]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.obstacle_type,
            "position": self.position.tolist(),
            "size": self.size.tolist(),
            "color": self.color,
            "solid": self.solid,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Obstacle":
        """Create obstacle from dictionary."""
        return cls(
            position=np.array(data["position"]),
            size=np.array(data["size"]),
            obstacle_type=data.get("type", "generic"),
            color=tuple(data.get("color", (100, 100, 100))),
            solid=data.get("solid", True)
        )


class Wall(Obstacle):
    """
    Wall obstacle - typically longer in one dimension.
    
    Walls are solid barriers that completely block movement.
    """
    
    def __init__(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        thickness: float = 10
    ):
        """
        Create a wall between two points.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            thickness: Wall thickness
        """
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        
        # Calculate center and size
        center = (start + end) / 2
        length = np.linalg.norm(end - start)
        
        # Determine orientation
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx > dy:
            # Horizontal wall
            size = np.array([length, thickness], dtype=np.float32)
        else:
            # Vertical wall
            size = np.array([thickness, length], dtype=np.float32)
        
        super().__init__(
            position=center,
            size=size,
            obstacle_type="wall",
            color=(80, 80, 80)
        )
        
        self.start = start
        self.end = end
        self.thickness = thickness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "start": self.start.tolist(),
            "end": self.end.tolist(),
            "thickness": self.thickness
        })
        return data


class Block(Obstacle):
    """
    Block obstacle - a simple rectangular obstacle.
    """
    
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: Tuple[int, int, int] = (120, 80, 40)
    ):
        """
        Create a block obstacle.
        
        Args:
            x: Center x position
            y: Center y position
            width: Block width
            height: Block height
            color: Block color (RGB)
        """
        super().__init__(
            position=np.array([x, y], dtype=np.float32),
            size=np.array([width, height], dtype=np.float32),
            obstacle_type="block",
            color=color
        )


class Ramp(Obstacle):
    """
    Ramp obstacle - provides a sloped surface.
    
    Ramps can be used by agents to traverse height differences.
    """
    
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        direction: str = "right"  # "left", "right", "up", "down"
    ):
        """
        Create a ramp obstacle.
        
        Args:
            x: Center x position
            y: Bottom y position
            width: Ramp width
            height: Ramp height
            direction: Direction the ramp slopes upward
        """
        super().__init__(
            position=np.array([x, y], dtype=np.float32),
            size=np.array([width, height], dtype=np.float32),
            obstacle_type="ramp",
            color=(100, 70, 30),
            solid=False  # Ramps allow partial passage
        )
        
        self.direction = direction
        self.slope = height / width
    
    def get_height_at(self, x: float) -> float:
        """
        Get the ramp height at a given x position.
        
        Args:
            x: X position
            
        Returns:
            Height of ramp at that position
        """
        bounds = self.get_bounds()
        
        if x < bounds[0] or x > bounds[2]:
            return 0
        
        # Normalize x to [0, 1] across ramp width
        t = (x - bounds[0]) / self.size[0]
        
        if self.direction == "right":
            return self.size[1] * t
        elif self.direction == "left":
            return self.size[1] * (1 - t)
        else:
            return self.size[1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data["direction"] = self.direction
        data["slope"] = self.slope
        return data


class Platform(Obstacle):
    """
    Platform obstacle - can be jumped on from below.
    
    Platforms are solid from above but can be passed through from below.
    """
    
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float = 10,
        one_way: bool = True
    ):
        """
        Create a platform obstacle.
        
        Args:
            x: Center x position
            y: Center y position
            width: Platform width
            height: Platform thickness
            one_way: If True, can pass through from below
        """
        super().__init__(
            position=np.array([x, y], dtype=np.float32),
            size=np.array([width, height], dtype=np.float32),
            obstacle_type="platform",
            color=(60, 60, 60)
        )
        
        self.one_way = one_way
    
    def should_collide(self, agent_position: np.ndarray, agent_velocity: np.ndarray) -> bool:
        """
        Check if collision should occur based on agent state.
        
        For one-way platforms, only collide if agent is moving downward
        and above the platform.
        """
        if not self.one_way:
            return True
        
        # Only collide if moving downward and above platform
        return (agent_velocity[1] > 0 and 
                agent_position[1] < self.position[1])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data["one_way"] = self.one_way
        return data


class MovingPlatform(Platform):
    """
    Moving platform that travels between waypoints.
    """
    
    def __init__(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        width: float,
        speed: float = 50,
        **kwargs
    ):
        """
        Create a moving platform.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            width: Platform width
            speed: Movement speed (pixels/second)
        """
        super().__init__(
            x=start_pos[0],
            y=start_pos[1],
            width=width,
            **kwargs
        )
        
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.end_pos = np.array(end_pos, dtype=np.float32)
        self.speed = speed
        self.t = 0.0  # Progress along path [0, 1]
        self.direction = 1  # 1 = toward end, -1 = toward start
        self.obstacle_type = "moving_platform"
    
    def update(self, dt: float) -> None:
        """Update platform position."""
        # Calculate distance and direction
        total_dist = np.linalg.norm(self.end_pos - self.start_pos)
        
        # Update progress
        self.t += self.direction * (self.speed * dt) / total_dist
        
        # Reverse at endpoints
        if self.t >= 1.0:
            self.t = 1.0
            self.direction = -1
        elif self.t <= 0.0:
            self.t = 0.0
            self.direction = 1
        
        # Update position
        self.position = self.start_pos + self.t * (self.end_pos - self.start_pos)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "start_pos": self.start_pos.tolist(),
            "end_pos": self.end_pos.tolist(),
            "speed": self.speed,
            "t": self.t,
        })
        return data


def create_obstacle_from_config(config: Dict[str, Any]) -> Obstacle:
    """
    Factory function to create obstacles from configuration.
    
    Args:
        config: Obstacle configuration dictionary
        
    Returns:
        Created obstacle instance
    """
    obs_type = config.get("type", "block")
    
    if obs_type == "wall":
        return Wall(
            start=config["start"],
            end=config["end"],
            thickness=config.get("thickness", 10)
        )
    elif obs_type == "block":
        return Block(
            x=config["x"],
            y=config["y"],
            width=config["width"],
            height=config["height"],
            color=tuple(config.get("color", (120, 80, 40)))
        )
    elif obs_type == "ramp":
        return Ramp(
            x=config["x"],
            y=config["y"],
            width=config["width"],
            height=config["height"],
            direction=config.get("direction", "right")
        )
    elif obs_type == "platform":
        return Platform(
            x=config["x"],
            y=config["y"],
            width=config["width"],
            height=config.get("height", 10),
            one_way=config.get("one_way", True)
        )
    elif obs_type == "moving_platform":
        return MovingPlatform(
            start_pos=config["start_pos"],
            end_pos=config["end_pos"],
            width=config["width"],
            speed=config.get("speed", 50)
        )
    else:
        return Obstacle.from_dict(config)

