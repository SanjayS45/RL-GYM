"""
Configuration Module
Centralized configuration for RL-GYM.
"""

from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # CORS
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    
    # Training Settings
    default_algorithm: str = Field(default="PPO", env="DEFAULT_ALGORITHM")
    default_total_timesteps: int = Field(default=100000, env="DEFAULT_TIMESTEPS")
    max_concurrent_sessions: int = Field(default=5, env="MAX_SESSIONS")
    
    # Storage
    models_dir: str = Field(default="models", env="MODELS_DIR")
    datasets_dir: str = Field(default="datasets", env="DATASETS_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    
    # Device
    device: str = Field(default="auto", env="DEVICE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Default hyperparameters for each algorithm
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "DQN": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 64,
        "learning_starts": 1000,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "use_double_dqn": True,
        "use_dueling": False,
        "hidden_dims": [256, 256],
    },
    "PPO": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "hidden_dims": [256, 256],
    },
    "SAC": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "buffer_size": 100000,
        "batch_size": 256,
        "tau": 0.005,
        "ent_coef": "auto",
        "learning_starts": 1000,
        "hidden_dims": [256, 256],
    },
    "A2C": {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "n_steps": 5,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "gae_lambda": 1.0,
        "hidden_dims": [64, 64],
    },
}


# Environment configurations
ENVIRONMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "grid_world": {
        "simple": {
            "grid_size": 5,
            "obstacle_positions": [(1, 1), (2, 2), (3, 3)],
            "max_steps": 200,
        },
        "maze": {
            "grid_size": 8,
            "obstacle_positions": [
                (1, 1), (1, 2), (1, 3), (1, 5), (1, 6),
                (3, 1), (3, 3), (3, 4), (3, 5),
                (5, 2), (5, 3), (5, 5), (5, 6),
            ],
            "max_steps": 500,
        },
    },
    "navigation": {
        "empty": {
            "width": 800,
            "height": 600,
            "obstacles": [],
            "max_steps": 500,
        },
        "simple_obstacles": {
            "width": 800,
            "height": 600,
            "obstacles": [
                {"type": "block", "x": 300, "y": 200, "width": 100, "height": 200},
                {"type": "block", "x": 500, "y": 400, "width": 150, "height": 100},
            ],
            "max_steps": 500,
        },
    },
    "platformer": {
        "simple": {
            "width": 800,
            "height": 600,
            "platforms": [
                {"x": 200, "y": 500, "width": 150},
                {"x": 400, "y": 400, "width": 150},
                {"x": 600, "y": 300, "width": 150},
            ],
            "max_steps": 1000,
        },
    },
}


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_algorithm_defaults(algorithm: str) -> Dict[str, Any]:
    """Get default hyperparameters for an algorithm."""
    return DEFAULT_HYPERPARAMS.get(algorithm, DEFAULT_HYPERPARAMS["PPO"]).copy()


def get_environment_config(env_type: str, config_name: str) -> Dict[str, Any]:
    """Get environment configuration."""
    env_configs = ENVIRONMENT_CONFIGS.get(env_type, {})
    return env_configs.get(config_name, {}).copy()

