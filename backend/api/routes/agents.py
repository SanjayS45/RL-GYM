"""Agent routes for RL-GYM API."""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import os
import uuid
import torch

from rl_core import set_seed
from rl_core.algorithms import DQN, PPO, SAC, A2C
from config import Config

router = APIRouter(prefix="/agents", tags=["agents"])

# Store created agents
_agents: Dict[str, Dict[str, Any]] = {}
config = Config()


class AgentConfig(BaseModel):
    """Agent configuration model."""
    algorithm: str
    state_dim: int
    action_dim: int
    hidden_dims: List[int] = [256, 256]
    learning_rate: float = 0.001
    gamma: float = 0.99
    seed: Optional[int] = None
    additional_params: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Agent creation response model."""
    agent_id: str
    algorithm: str
    status: str


@router.get("/algorithms")
async def list_algorithms():
    """List available RL algorithms with detailed configuration options."""
    return {
        "algorithms": [
            {
                "id": "dqn",
                "name": "Deep Q-Network (DQN)",
                "description": "Value-based RL algorithm that uses a neural network to approximate the Q-function",
                "type": "value_based",
                "action_space": "discrete",
                "observation_space": "any",
                "parameters": {
                    "learning_rate": {
                        "type": "float",
                        "default": 0.001,
                        "min": 0.0001,
                        "max": 0.1,
                        "description": "Learning rate for the optimizer"
                    },
                    "gamma": {
                        "type": "float",
                        "default": 0.99,
                        "min": 0.9,
                        "max": 0.999,
                        "description": "Discount factor for future rewards"
                    },
                    "epsilon_start": {
                        "type": "float",
                        "default": 1.0,
                        "description": "Initial exploration rate"
                    },
                    "epsilon_end": {
                        "type": "float",
                        "default": 0.01,
                        "description": "Final exploration rate"
                    },
                    "epsilon_decay": {
                        "type": "float",
                        "default": 0.995,
                        "description": "Exploration decay rate per episode"
                    },
                    "buffer_size": {
                        "type": "int",
                        "default": 10000,
                        "min": 1000,
                        "max": 1000000,
                        "description": "Replay buffer size"
                    },
                    "batch_size": {
                        "type": "int",
                        "default": 64,
                        "min": 16,
                        "max": 512,
                        "description": "Training batch size"
                    },
                    "target_update_freq": {
                        "type": "int",
                        "default": 100,
                        "description": "Steps between target network updates"
                    },
                    "dueling": {
                        "type": "bool",
                        "default": False,
                        "description": "Use dueling network architecture"
                    }
                }
            },
            {
                "id": "ppo",
                "name": "Proximal Policy Optimization (PPO)",
                "description": "Policy gradient algorithm with clipped surrogate objective for stable training",
                "type": "policy_gradient",
                "action_space": "both",
                "observation_space": "any",
                "parameters": {
                    "learning_rate": {
                        "type": "float",
                        "default": 0.0003,
                        "min": 0.00001,
                        "max": 0.01,
                        "description": "Learning rate for the optimizer"
                    },
                    "gamma": {
                        "type": "float",
                        "default": 0.99,
                        "description": "Discount factor"
                    },
                    "gae_lambda": {
                        "type": "float",
                        "default": 0.95,
                        "description": "GAE lambda parameter"
                    },
                    "clip_epsilon": {
                        "type": "float",
                        "default": 0.2,
                        "min": 0.1,
                        "max": 0.4,
                        "description": "PPO clipping parameter"
                    },
                    "value_coef": {
                        "type": "float",
                        "default": 0.5,
                        "description": "Value loss coefficient"
                    },
                    "entropy_coef": {
                        "type": "float",
                        "default": 0.01,
                        "description": "Entropy bonus coefficient"
                    },
                    "n_steps": {
                        "type": "int",
                        "default": 2048,
                        "description": "Steps to collect per update"
                    },
                    "n_epochs": {
                        "type": "int",
                        "default": 10,
                        "description": "Optimization epochs per update"
                    },
                    "batch_size": {
                        "type": "int",
                        "default": 64,
                        "description": "Mini-batch size"
                    },
                    "max_grad_norm": {
                        "type": "float",
                        "default": 0.5,
                        "description": "Gradient clipping threshold"
                    }
                }
            },
            {
                "id": "sac",
                "name": "Soft Actor-Critic (SAC)",
                "description": "Off-policy actor-critic with entropy regularization for maximum entropy RL",
                "type": "actor_critic",
                "action_space": "continuous",
                "observation_space": "any",
                "parameters": {
                    "learning_rate": {
                        "type": "float",
                        "default": 0.0003,
                        "description": "Learning rate"
                    },
                    "gamma": {
                        "type": "float",
                        "default": 0.99,
                        "description": "Discount factor"
                    },
                    "tau": {
                        "type": "float",
                        "default": 0.005,
                        "min": 0.001,
                        "max": 0.1,
                        "description": "Soft target update rate"
                    },
                    "alpha": {
                        "type": "float",
                        "default": 0.2,
                        "description": "Entropy regularization coefficient"
                    },
                    "auto_alpha": {
                        "type": "bool",
                        "default": True,
                        "description": "Automatically tune alpha"
                    },
                    "buffer_size": {
                        "type": "int",
                        "default": 100000,
                        "description": "Replay buffer size"
                    },
                    "batch_size": {
                        "type": "int",
                        "default": 256,
                        "description": "Training batch size"
                    }
                }
            },
            {
                "id": "a2c",
                "name": "Advantage Actor-Critic (A2C)",
                "description": "Synchronous actor-critic algorithm for efficient on-policy learning",
                "type": "actor_critic",
                "action_space": "both",
                "observation_space": "any",
                "parameters": {
                    "learning_rate": {
                        "type": "float",
                        "default": 0.0007,
                        "description": "Learning rate"
                    },
                    "gamma": {
                        "type": "float",
                        "default": 0.99,
                        "description": "Discount factor"
                    },
                    "value_coef": {
                        "type": "float",
                        "default": 0.5,
                        "description": "Value loss coefficient"
                    },
                    "entropy_coef": {
                        "type": "float",
                        "default": 0.01,
                        "description": "Entropy coefficient"
                    },
                    "n_steps": {
                        "type": "int",
                        "default": 5,
                        "description": "Steps per update"
                    },
                    "max_grad_norm": {
                        "type": "float",
                        "default": 0.5,
                        "description": "Gradient clipping"
                    }
                }
            }
        ]
    }


def _create_agent(config: AgentConfig) -> Any:
    """Create an agent from configuration."""
    algorithm = config.algorithm.lower()
    
    if config.seed is not None:
        set_seed(config.seed)
    
    additional = config.additional_params or {}
    
    if algorithm == "dqn":
        return DQN(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.learning_rate,
            gamma=config.gamma,
            **additional
        )
    elif algorithm == "ppo":
        return PPO(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.learning_rate,
            gamma=config.gamma,
            **additional
        )
    elif algorithm == "sac":
        return SAC(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.learning_rate,
            gamma=config.gamma,
            **additional
        )
    elif algorithm == "a2c":
        return A2C(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            lr=config.learning_rate,
            gamma=config.gamma,
            **additional
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


@router.post("/create", response_model=AgentResponse)
async def create_agent(agent_config: AgentConfig):
    """Create a new agent instance."""
    try:
        agent = _create_agent(agent_config)
        agent_id = str(uuid.uuid4())[:8]
        
        _agents[agent_id] = {
            "agent": agent,
            "config": agent_config.model_dump(),
            "training_steps": 0
        }
        
        return AgentResponse(
            agent_id=agent_id,
            algorithm=agent_config.algorithm,
            status="created"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent information."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_info = _agents[agent_id]
    return {
        "agent_id": agent_id,
        "algorithm": agent_info["config"]["algorithm"],
        "training_steps": agent_info["training_steps"],
        "config": agent_info["config"]
    }


@router.post("/{agent_id}/save")
async def save_agent(agent_id: str, filename: Optional[str] = None):
    """Save agent to disk."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_info = _agents[agent_id]
    agent = agent_info["agent"]
    
    # Create models directory
    models_dir = config.models_dir
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate filename
    save_filename = filename or f"{agent_id}_{agent_info['config']['algorithm']}.pt"
    save_path = os.path.join(models_dir, save_filename)
    
    # Save agent
    agent.save(save_path)
    
    return {
        "status": "saved",
        "agent_id": agent_id,
        "path": save_path
    }


@router.post("/{agent_id}/load")
async def load_agent_weights(agent_id: str, filepath: str):
    """Load agent weights from file."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Weight file not found")
    
    agent_info = _agents[agent_id]
    agent = agent_info["agent"]
    agent.load(filepath)
    
    return {
        "status": "loaded",
        "agent_id": agent_id,
        "path": filepath
    }


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent instance."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    del _agents[agent_id]
    
    return {"status": "deleted", "agent_id": agent_id}


@router.get("/list/active")
async def list_active_agents():
    """List all active agent instances."""
    return {
        "agents": [
            {
                "agent_id": agent_id,
                "algorithm": info["config"]["algorithm"],
                "training_steps": info["training_steps"]
            }
            for agent_id, info in _agents.items()
        ]
    }


@router.get("/saved/list")
async def list_saved_agents():
    """List all saved agent models."""
    models_dir = config.models_dir
    
    if not os.path.exists(models_dir):
        return {"saved_agents": []}
    
    saved = []
    for filename in os.listdir(models_dir):
        if filename.endswith(('.pt', '.pth')):
            filepath = os.path.join(models_dir, filename)
            saved.append({
                "filename": filename,
                "path": filepath,
                "size": os.path.getsize(filepath)
            })
    
    return {"saved_agents": saved}


@router.post("/upload")
async def upload_agent_weights(file: UploadFile = File(...), agent_id: Optional[str] = None):
    """Upload agent weights file."""
    try:
        contents = await file.read()
        
        models_dir = config.models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        filename = file.filename or f"uploaded_{uuid.uuid4()[:8]}.pt"
        save_path = os.path.join(models_dir, filename)
        
        with open(save_path, 'wb') as f:
            f.write(contents)
        
        # If agent_id provided, load into that agent
        if agent_id and agent_id in _agents:
            agent_info = _agents[agent_id]
            agent = agent_info["agent"]
            agent.load(save_path)
        
        return {
            "status": "uploaded",
            "path": save_path,
            "agent_id": agent_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{agent_id}/act")
async def get_action(agent_id: str, state: List[float], deterministic: bool = False):
    """Get an action from the agent for a given state."""
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_info = _agents[agent_id]
    agent = agent_info["agent"]
    
    import numpy as np
    state_array = np.array(state, dtype=np.float32)
    
    action = agent.act(state_array, deterministic=deterministic)
    
    return {
        "action": action.tolist() if hasattr(action, 'tolist') else action
    }
