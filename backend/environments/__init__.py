"""
Environment Engine Module
Contains Gym-style environments with physics, obstacles, and goals.
"""

from .base import BaseEnvironment, PhysicsEngine
from .obstacles import Obstacle, Wall, Block, Ramp, Platform
from .goals import Goal, PositionGoal, CollectGoal, SurvivalGoal
from .grid_world import GridWorldEnv
from .navigation import NavigationEnv
from .platformer import PlatformerEnv

__all__ = [
    "BaseEnvironment",
    "PhysicsEngine",
    "Obstacle",
    "Wall",
    "Block",
    "Ramp",
    "Platform",
    "Goal",
    "PositionGoal",
    "CollectGoal",
    "SurvivalGoal",
    "GridWorldEnv",
    "NavigationEnv",
    "PlatformerEnv",
]

