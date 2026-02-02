"""
Environment API Routes

Endpoints for managing and configuring environments.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class EnvironmentConfig(BaseModel):
    """Environment configuration model."""
    
    name: str
    type: str  # grid_world, navigation, platformer
    width: int = 800
    height: int = 600
    max_steps: int = 1000
    
    # Type-specific config
    obstacles: List[Dict[str, Any]] = Field(default_factory=list)
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Physics settings
    gravity: float = 0.0
    friction: float = 0.1


class EnvironmentInfo(BaseModel):
    """Environment information response."""
    
    name: str
    type: str
    description: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    configs: List[str] = []


# Available environments
ENVIRONMENTS = {
    "grid_world": {
        "name": "Grid World",
        "type": "grid_world",
        "description": "Classic grid navigation environment with discrete actions",
        "observation_space": {"type": "Box", "shape": [100], "dtype": "float32"},
        "action_space": {"type": "Discrete", "n": 4},
        "configs": ["simple", "maze", "cliff"],
    },
    "navigation": {
        "name": "Navigation",
        "type": "navigation",
        "description": "Continuous 2D navigation with obstacles",
        "observation_space": {"type": "Box", "shape": [12], "dtype": "float32"},
        "action_space": {"type": "Box", "shape": [2], "dtype": "float32"},
        "configs": ["empty", "simple_obstacles", "maze_like", "cluttered"],
    },
    "platformer": {
        "name": "Platformer",
        "type": "platformer",
        "description": "Physics-based platformer with jumping",
        "observation_space": {"type": "Box", "shape": [15], "dtype": "float32"},
        "action_space": {"type": "Discrete", "n": 6},
        "configs": ["simple", "climbing", "gaps", "moving_platforms"],
    },
}


@router.get("/", response_model=List[EnvironmentInfo])
async def list_environments():
    """List all available environments."""
    return [
        EnvironmentInfo(**env_data)
        for env_data in ENVIRONMENTS.values()
    ]


@router.get("/{env_type}", response_model=EnvironmentInfo)
async def get_environment(env_type: str):
    """Get information about a specific environment type."""
    if env_type not in ENVIRONMENTS:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    return EnvironmentInfo(**ENVIRONMENTS[env_type])


@router.get("/{env_type}/configs")
async def get_environment_configs(env_type: str):
    """Get available configurations for an environment type."""
    if env_type not in ENVIRONMENTS:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    configs = ENVIRONMENTS[env_type]["configs"]
    return {"configs": configs}


@router.post("/create")
async def create_environment(config: EnvironmentConfig):
    """
    Create a new environment instance.
    
    Returns environment ID and initial state.
    """
    import uuid
    
    env_id = str(uuid.uuid4())[:8]
    
    # In production, this would create an actual environment instance
    return {
        "env_id": env_id,
        "config": config.dict(),
        "initial_state": {
            "width": config.width,
            "height": config.height,
            "agent": {"position": [100, 300], "velocity": [0, 0]},
            "obstacles": config.obstacles,
            "goals": config.goals,
        }
    }


@router.post("/{env_id}/step")
async def step_environment(env_id: str, action: Dict[str, Any]):
    """
    Take a step in an environment.
    
    Returns next state, reward, and done flag.
    """
    import numpy as np
    
    # Simulated step response
    return {
        "state": {
            "position": [np.random.uniform(0, 800), np.random.uniform(0, 600)],
            "velocity": [np.random.normal(0, 10), np.random.normal(0, 10)],
        },
        "reward": np.random.normal(0, 1),
        "terminated": False,
        "truncated": False,
        "info": {},
    }


@router.post("/{env_id}/reset")
async def reset_environment(env_id: str):
    """Reset an environment to initial state."""
    return {
        "state": {
            "position": [100, 300],
            "velocity": [0, 0],
        },
        "info": {},
    }


@router.get("/{env_id}/state")
async def get_environment_state(env_id: str):
    """Get current environment state for visualization."""
    return {
        "width": 800,
        "height": 600,
        "agent": {
            "position": [100, 300],
            "velocity": [0, 0],
            "size": [20, 20],
        },
        "obstacles": [],
        "goals": [{"position": [700, 300], "size": [40, 40], "achieved": False}],
        "step": 0,
    }

