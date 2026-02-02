"""
RL Core Module
Contains base classes and utilities for reinforcement learning algorithms.
"""

from .base import BaseAgent, BasePolicy, BaseBuffer
from .networks import MLP, CNN, ActorCritic
from .utils import set_seed, explained_variance, polyak_update
from .buffers import ReplayBuffer, PrioritizedReplayBuffer, RolloutBuffer
from .schedulers import (
    Scheduler,
    LinearSchedule,
    ExponentialSchedule,
    CosineAnnealingSchedule,
    WarmupSchedule,
    ConstantSchedule,
    EpsilonGreedy,
    create_scheduler,
)
from .metrics import (
    MetricsCollector,
    EpisodeMetrics,
    StepMetrics,
    RewardTracker,
    compute_discounted_returns,
    compute_gae,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "BasePolicy", 
    "BaseBuffer",
    # Networks
    "MLP",
    "CNN",
    "ActorCritic",
    # Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "RolloutBuffer",
    # Schedulers
    "Scheduler",
    "LinearSchedule",
    "ExponentialSchedule",
    "CosineAnnealingSchedule",
    "WarmupSchedule",
    "ConstantSchedule",
    "EpsilonGreedy",
    "create_scheduler",
    # Metrics
    "MetricsCollector",
    "EpisodeMetrics",
    "StepMetrics",
    "RewardTracker",
    "compute_discounted_returns",
    "compute_gae",
    # Utilities
    "set_seed",
    "explained_variance",
    "polyak_update",
]

