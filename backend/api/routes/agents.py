"""
Agent API Routes

Endpoints for managing and configuring RL agents.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class AgentConfig(BaseModel):
    """Agent configuration model."""
    
    algorithm: str = Field(default="PPO", description="RL algorithm")
    
    # Common hyperparameters
    learning_rate: float = Field(default=3e-4, ge=1e-6, le=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    
    # Network architecture
    hidden_dims: List[int] = Field(default=[256, 256])
    
    # Algorithm-specific
    extra_config: Dict[str, Any] = Field(default_factory=dict)


class AlgorithmInfo(BaseModel):
    """Algorithm information response."""
    
    name: str
    description: str
    type: str  # on_policy, off_policy
    continuous_action: bool
    discrete_action: bool
    hyperparameters: List[Dict[str, Any]]


# Available algorithms
ALGORITHMS = {
    "DQN": {
        "name": "Deep Q-Network",
        "description": "Value-based algorithm for discrete actions using Q-learning",
        "type": "off_policy",
        "continuous_action": False,
        "discrete_action": True,
        "hyperparameters": [
            {"name": "learning_rate", "type": "float", "default": 1e-4, "range": [1e-6, 1e-2]},
            {"name": "gamma", "type": "float", "default": 0.99, "range": [0.9, 0.999]},
            {"name": "buffer_size", "type": "int", "default": 100000, "range": [10000, 1000000]},
            {"name": "batch_size", "type": "int", "default": 64, "range": [16, 512]},
            {"name": "exploration_fraction", "type": "float", "default": 0.1, "range": [0.01, 0.5]},
            {"name": "exploration_final_eps", "type": "float", "default": 0.05, "range": [0.01, 0.2]},
            {"name": "target_update_interval", "type": "int", "default": 1000, "range": [100, 10000]},
            {"name": "use_double_dqn", "type": "bool", "default": True},
            {"name": "use_dueling", "type": "bool", "default": False},
        ],
    },
    "PPO": {
        "name": "Proximal Policy Optimization",
        "description": "Stable policy gradient algorithm with clipped surrogate objective",
        "type": "on_policy",
        "continuous_action": True,
        "discrete_action": True,
        "hyperparameters": [
            {"name": "learning_rate", "type": "float", "default": 3e-4, "range": [1e-5, 1e-2]},
            {"name": "gamma", "type": "float", "default": 0.99, "range": [0.9, 0.999]},
            {"name": "n_steps", "type": "int", "default": 2048, "range": [64, 8192]},
            {"name": "batch_size", "type": "int", "default": 64, "range": [16, 512]},
            {"name": "n_epochs", "type": "int", "default": 10, "range": [1, 30]},
            {"name": "clip_range", "type": "float", "default": 0.2, "range": [0.1, 0.4]},
            {"name": "ent_coef", "type": "float", "default": 0.01, "range": [0.0, 0.1]},
            {"name": "vf_coef", "type": "float", "default": 0.5, "range": [0.1, 1.0]},
            {"name": "gae_lambda", "type": "float", "default": 0.95, "range": [0.8, 1.0]},
        ],
    },
    "SAC": {
        "name": "Soft Actor-Critic",
        "description": "Maximum entropy algorithm for continuous control",
        "type": "off_policy",
        "continuous_action": True,
        "discrete_action": False,
        "hyperparameters": [
            {"name": "learning_rate", "type": "float", "default": 3e-4, "range": [1e-5, 1e-2]},
            {"name": "gamma", "type": "float", "default": 0.99, "range": [0.9, 0.999]},
            {"name": "buffer_size", "type": "int", "default": 100000, "range": [10000, 1000000]},
            {"name": "batch_size", "type": "int", "default": 256, "range": [32, 512]},
            {"name": "tau", "type": "float", "default": 0.005, "range": [0.001, 0.1]},
            {"name": "ent_coef", "type": "string", "default": "auto", "options": ["auto", "0.1", "0.2"]},
        ],
    },
    "A2C": {
        "name": "Advantage Actor-Critic",
        "description": "Synchronous actor-critic with advantage estimation",
        "type": "on_policy",
        "continuous_action": True,
        "discrete_action": True,
        "hyperparameters": [
            {"name": "learning_rate", "type": "float", "default": 7e-4, "range": [1e-5, 1e-2]},
            {"name": "gamma", "type": "float", "default": 0.99, "range": [0.9, 0.999]},
            {"name": "n_steps", "type": "int", "default": 5, "range": [1, 20]},
            {"name": "ent_coef", "type": "float", "default": 0.01, "range": [0.0, 0.1]},
            {"name": "vf_coef", "type": "float", "default": 0.5, "range": [0.1, 1.0]},
            {"name": "gae_lambda", "type": "float", "default": 1.0, "range": [0.8, 1.0]},
        ],
    },
}


@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_algorithms():
    """List all available RL algorithms."""
    return [
        AlgorithmInfo(**algo_data)
        for algo_data in ALGORITHMS.values()
    ]


@router.get("/algorithms/{algorithm}", response_model=AlgorithmInfo)
async def get_algorithm(algorithm: str):
    """Get information about a specific algorithm."""
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    return AlgorithmInfo(**ALGORITHMS[algorithm])


@router.get("/algorithms/{algorithm}/hyperparameters")
async def get_algorithm_hyperparameters(algorithm: str):
    """Get hyperparameter definitions for an algorithm."""
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=404, detail="Algorithm not found")
    
    return {
        "algorithm": algorithm,
        "hyperparameters": ALGORITHMS[algorithm]["hyperparameters"],
    }


@router.post("/create")
async def create_agent(config: AgentConfig):
    """
    Create a new agent instance.
    
    Returns agent ID and configuration.
    """
    import uuid
    
    agent_id = str(uuid.uuid4())[:8]
    
    return {
        "agent_id": agent_id,
        "algorithm": config.algorithm,
        "config": config.dict(),
    }


@router.post("/{agent_id}/predict")
async def predict_action(agent_id: str, observation: Dict[str, Any]):
    """
    Get action prediction from an agent.
    
    Returns predicted action and optional metadata.
    """
    import numpy as np
    
    # Simulated prediction
    return {
        "action": np.random.uniform(-1, 1, size=2).tolist(),
        "value": np.random.uniform(-10, 10),
        "log_prob": np.random.uniform(-5, 0),
    }


@router.post("/{agent_id}/save")
async def save_agent(agent_id: str, path: str):
    """Save agent to disk."""
    return {"status": "saved", "path": path}


@router.post("/{agent_id}/load")
async def load_agent(agent_id: str, path: str):
    """Load agent from disk."""
    return {"status": "loaded", "path": path}

