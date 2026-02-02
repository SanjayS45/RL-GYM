"""
API Module
FastAPI server for RL-GYM training and visualization.
"""

from .main import app
from .routes import training, environments, agents, datasets

__all__ = ["app"]

