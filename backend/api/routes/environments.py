"""Environment routes for RL-GYM API."""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from environments import GridWorldEnv, NavigationEnv, PlatformerEnv
from nlp import GoalParser

router = APIRouter(prefix="/environments", tags=["environments"])

# Goal parser for natural language goals
goal_parser = GoalParser()


class ObstacleConfig(BaseModel):
    """Obstacle configuration model."""
    x: float
    y: float
    width: float
    height: float
    obstacle_type: str = "static"


class GoalConfig(BaseModel):
    """Goal configuration model."""
    x: float
    y: float
    radius: float = 0.5
    reward: float = 10.0


class EnvironmentConfig(BaseModel):
    """Environment configuration request model."""
    env_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    obstacles: Optional[List[ObstacleConfig]] = None
    goals: Optional[List[GoalConfig]] = None
    natural_language_goal: Optional[str] = None
    seed: Optional[int] = None
    render_mode: Optional[str] = None


class EnvironmentResponse(BaseModel):
    """Environment response model."""
    env_id: str
    env_type: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]


# Store created environments
_environments: Dict[str, Any] = {}
_env_counter = 0


def _create_environment(config: EnvironmentConfig) -> Any:
    """Create an environment from configuration."""
    env_type = config.env_type.lower()
    
    if env_type == "gridworld":
        env_kwargs = {
            "size": config.width or 10,
            "seed": config.seed,
            "render_mode": config.render_mode
        }
        return GridWorldEnv(**{k: v for k, v in env_kwargs.items() if v is not None})
    
    elif env_type == "navigation":
        env_kwargs = {
            "width": float(config.width or 20),
            "height": float(config.height or 20),
            "seed": config.seed,
            "render_mode": config.render_mode
        }
        return NavigationEnv(**{k: v for k, v in env_kwargs.items() if v is not None})
    
    elif env_type == "platformer":
        env_kwargs = {
            "width": float(config.width or 30),
            "height": float(config.height or 15),
            "seed": config.seed,
            "render_mode": config.render_mode
        }
        return PlatformerEnv(**{k: v for k, v in env_kwargs.items() if v is not None})
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


@router.get("/list")
async def list_environments():
    """List all available environment types."""
    return {
        "environments": [
            {
                "id": "gridworld",
                "name": "Grid World",
                "description": "A simple grid-based navigation environment where an agent must reach a goal",
                "observation_type": "discrete",
                "action_type": "discrete",
                "parameters": {
                    "size": {"type": "int", "default": 10, "min": 5, "max": 50, "description": "Grid size"},
                    "obstacle_density": {"type": "float", "default": 0.1, "min": 0, "max": 0.5, "description": "Density of obstacles"}
                },
                "preview": {
                    "type": "grid",
                    "defaultSize": [10, 10]
                }
            },
            {
                "id": "navigation",
                "name": "Continuous Navigation",
                "description": "A continuous 2D navigation environment with physics and lidar sensors",
                "observation_type": "continuous",
                "action_type": "continuous",
                "parameters": {
                    "width": {"type": "float", "default": 20, "min": 10, "max": 100, "description": "Environment width"},
                    "height": {"type": "float", "default": 20, "min": 10, "max": 100, "description": "Environment height"},
                    "num_rays": {"type": "int", "default": 16, "min": 4, "max": 32, "description": "Number of lidar rays"},
                    "max_speed": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "description": "Maximum agent speed"}
                },
                "preview": {
                    "type": "continuous",
                    "defaultSize": [20, 20]
                }
            },
            {
                "id": "platformer",
                "name": "Platformer",
                "description": "A 2D platformer environment with jumping mechanics and gravity",
                "observation_type": "continuous",
                "action_type": "discrete",
                "parameters": {
                    "width": {"type": "float", "default": 30, "min": 15, "max": 100, "description": "Level width"},
                    "height": {"type": "float", "default": 15, "min": 10, "max": 50, "description": "Level height"},
                    "gravity": {"type": "float", "default": 0.5, "min": 0.1, "max": 1.0, "description": "Gravity strength"},
                    "jump_force": {"type": "float", "default": 3.0, "min": 1.0, "max": 5.0, "description": "Jump force"}
                },
                "preview": {
                    "type": "platformer",
                    "defaultSize": [30, 15]
                }
            }
        ]
    }


@router.post("/create", response_model=EnvironmentResponse)
async def create_environment(config: EnvironmentConfig):
    """Create a new environment instance."""
    global _env_counter
    
    try:
        env = _create_environment(config)
        env_id = f"env_{_env_counter}"
        _env_counter += 1
        _environments[env_id] = env
        
        # Get space info
        obs_space = env.observation_space
        act_space = env.action_space
        
        return EnvironmentResponse(
            env_id=env_id,
            env_type=config.env_type,
            observation_space=_space_to_dict(obs_space),
            action_space=_space_to_dict(act_space)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _space_to_dict(space) -> Dict[str, Any]:
    """Convert a Gymnasium space to a dictionary representation."""
    space_type = type(space).__name__
    
    if space_type == "Discrete":
        return {"type": "discrete", "n": space.n}
    elif space_type == "Box":
        return {
            "type": "box",
            "shape": list(space.shape),
            "low": space.low.tolist() if hasattr(space.low, 'tolist') else float(space.low),
            "high": space.high.tolist() if hasattr(space.high, 'tolist') else float(space.high)
        }
    elif space_type == "MultiDiscrete":
        return {"type": "multi_discrete", "nvec": space.nvec.tolist()}
    else:
        return {"type": space_type}


@router.get("/{env_id}")
async def get_environment(env_id: str):
    """Get information about a specific environment instance."""
    if env_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = _environments[env_id]
    return {
        "env_id": env_id,
        "observation_space": _space_to_dict(env.observation_space),
        "action_space": _space_to_dict(env.action_space)
    }


@router.post("/{env_id}/reset")
async def reset_environment(env_id: str, seed: Optional[int] = None):
    """Reset an environment instance."""
    if env_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = _environments[env_id]
    obs, info = env.reset(seed=seed)
    
    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "info": info
    }


@router.post("/{env_id}/step")
async def step_environment(env_id: str, action: Any):
    """Take a step in an environment instance."""
    if env_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = _environments[env_id]
    obs, reward, terminated, truncated, info = env.step(action)
    
    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }


@router.delete("/{env_id}")
async def delete_environment(env_id: str):
    """Delete an environment instance."""
    if env_id not in _environments:
        raise HTTPException(status_code=404, detail="Environment not found")
    
    env = _environments.pop(env_id)
    env.close()
    
    return {"status": "deleted", "env_id": env_id}


@router.post("/parse-goal")
async def parse_natural_language_goal(goal: str):
    """Parse a natural language goal into a structured format."""
    try:
        parsed = goal_parser.parse(goal)
        return {
            "original": goal,
            "parsed": parsed.to_dict() if hasattr(parsed, 'to_dict') else {
                "goal_type": parsed.goal_type,
                "target_description": parsed.target_description,
                "constraints": parsed.constraints,
                "metrics": parsed.metrics,
                "termination_conditions": parsed.termination_conditions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/templates/obstacles")
async def get_obstacle_templates():
    """Get predefined obstacle templates."""
    return {
        "templates": [
            {
                "id": "wall_vertical",
                "name": "Vertical Wall",
                "width": 0.5,
                "height": 3.0,
                "obstacle_type": "static"
            },
            {
                "id": "wall_horizontal",
                "name": "Horizontal Wall",
                "width": 3.0,
                "height": 0.5,
                "obstacle_type": "static"
            },
            {
                "id": "box",
                "name": "Box",
                "width": 1.0,
                "height": 1.0,
                "obstacle_type": "static"
            },
            {
                "id": "platform",
                "name": "Platform",
                "width": 4.0,
                "height": 0.5,
                "obstacle_type": "platform"
            },
            {
                "id": "moving_platform",
                "name": "Moving Platform",
                "width": 3.0,
                "height": 0.5,
                "obstacle_type": "moving"
            }
        ]
    }


@router.get("/active")
async def list_active_environments():
    """List all active environment instances."""
    return {
        "environments": [
            {
                "env_id": env_id,
                "type": type(env).__name__
            }
            for env_id, env in _environments.items()
        ]
    }
