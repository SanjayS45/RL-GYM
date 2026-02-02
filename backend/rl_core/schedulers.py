"""Learning rate and exploration schedulers for RL algorithms."""
from abc import ABC, abstractmethod
from typing import Callable
import math


class Scheduler(ABC):
    """Abstract base class for schedulers."""
    
    @abstractmethod
    def value(self, step: int) -> float:
        """Get the scheduled value at a given step."""
        pass
    
    @abstractmethod
    def step(self) -> float:
        """Advance the scheduler and return the new value."""
        pass


class LinearSchedule(Scheduler):
    """Linear interpolation between initial and final values."""
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int
    ):
        """
        Initialize linear scheduler.
        
        Args:
            initial_value: Starting value
            final_value: Ending value
            total_steps: Number of steps for the schedule
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self._step = 0
    
    def value(self, step: int) -> float:
        """Get value at a specific step."""
        progress = min(1.0, step / self.total_steps)
        return self.initial_value + progress * (self.final_value - self.initial_value)
    
    def step(self) -> float:
        """Advance scheduler and return new value."""
        value = self.value(self._step)
        self._step += 1
        return value


class ExponentialSchedule(Scheduler):
    """Exponential decay schedule."""
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        decay_rate: float
    ):
        """
        Initialize exponential scheduler.
        
        Args:
            initial_value: Starting value
            final_value: Minimum value (floor)
            decay_rate: Decay rate per step (0-1)
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_rate = decay_rate
        self._step = 0
        self._current_value = initial_value
    
    def value(self, step: int) -> float:
        """Get value at a specific step."""
        return max(
            self.final_value,
            self.initial_value * (self.decay_rate ** step)
        )
    
    def step(self) -> float:
        """Advance scheduler and return new value."""
        self._current_value = max(
            self.final_value,
            self._current_value * self.decay_rate
        )
        self._step += 1
        return self._current_value


class CosineAnnealingSchedule(Scheduler):
    """Cosine annealing schedule with warm restarts."""
    
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        num_cycles: int = 1
    ):
        """
        Initialize cosine annealing scheduler.
        
        Args:
            initial_value: Maximum value
            final_value: Minimum value
            total_steps: Total number of steps
            num_cycles: Number of cosine cycles
        """
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.num_cycles = num_cycles
        self._step = 0
    
    def value(self, step: int) -> float:
        """Get value at a specific step."""
        steps_per_cycle = self.total_steps / self.num_cycles
        cycle_position = (step % steps_per_cycle) / steps_per_cycle
        
        cosine_value = (1 + math.cos(math.pi * cycle_position)) / 2
        return self.final_value + (self.initial_value - self.final_value) * cosine_value
    
    def step(self) -> float:
        """Advance scheduler and return new value."""
        value = self.value(self._step)
        self._step += 1
        return value


class WarmupSchedule(Scheduler):
    """Linear warmup followed by decay."""
    
    def __init__(
        self,
        initial_value: float,
        peak_value: float,
        final_value: float,
        warmup_steps: int,
        total_steps: int
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            initial_value: Starting value (before warmup)
            peak_value: Peak value (after warmup)
            final_value: Final value (after decay)
            warmup_steps: Number of warmup steps
            total_steps: Total number of steps
        """
        self.initial_value = initial_value
        self.peak_value = peak_value
        self.final_value = final_value
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
    
    def value(self, step: int) -> float:
        """Get value at a specific step."""
        if step < self.warmup_steps:
            # Linear warmup
            progress = step / self.warmup_steps
            return self.initial_value + progress * (self.peak_value - self.initial_value)
        else:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            if decay_steps <= 0:
                return self.peak_value
            progress = (step - self.warmup_steps) / decay_steps
            return self.peak_value + progress * (self.final_value - self.peak_value)
    
    def step(self) -> float:
        """Advance scheduler and return new value."""
        value = self.value(self._step)
        self._step += 1
        return value


class ConstantSchedule(Scheduler):
    """Constant value scheduler."""
    
    def __init__(self, value: float):
        """Initialize constant scheduler."""
        self._value = value
    
    def value(self, step: int) -> float:
        """Get value (always constant)."""
        return self._value
    
    def step(self) -> float:
        """Return constant value."""
        return self._value


class StepSchedule(Scheduler):
    """Step-wise schedule with predefined milestones."""
    
    def __init__(
        self,
        initial_value: float,
        milestones: dict[int, float]
    ):
        """
        Initialize step scheduler.
        
        Args:
            initial_value: Starting value
            milestones: Dictionary mapping step numbers to values
        """
        self.initial_value = initial_value
        self.milestones = sorted(milestones.items())
        self._step = 0
    
    def value(self, step: int) -> float:
        """Get value at a specific step."""
        current_value = self.initial_value
        
        for milestone_step, milestone_value in self.milestones:
            if step >= milestone_step:
                current_value = milestone_value
            else:
                break
        
        return current_value
    
    def step(self) -> float:
        """Advance scheduler and return new value."""
        value = self.value(self._step)
        self._step += 1
        return value


def create_scheduler(
    schedule_type: str,
    **kwargs
) -> Scheduler:
    """
    Factory function to create schedulers.
    
    Args:
        schedule_type: Type of scheduler ('linear', 'exponential', 'cosine', 'warmup', 'constant', 'step')
        **kwargs: Arguments for the specific scheduler
        
    Returns:
        Scheduler instance
    """
    schedulers = {
        'linear': LinearSchedule,
        'exponential': ExponentialSchedule,
        'cosine': CosineAnnealingSchedule,
        'warmup': WarmupSchedule,
        'constant': ConstantSchedule,
        'step': StepSchedule,
    }
    
    if schedule_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {schedule_type}")
    
    return schedulers[schedule_type](**kwargs)


class EpsilonGreedy:
    """Epsilon-greedy exploration strategy."""
    
    def __init__(
        self,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.01,
        decay_steps: int = 10000,
        decay_type: str = 'linear'
    ):
        """
        Initialize epsilon-greedy strategy.
        
        Args:
            initial_epsilon: Starting exploration rate
            final_epsilon: Final exploration rate
            decay_steps: Number of steps for decay
            decay_type: Type of decay ('linear', 'exponential')
        """
        if decay_type == 'linear':
            self.scheduler = LinearSchedule(initial_epsilon, final_epsilon, decay_steps)
        elif decay_type == 'exponential':
            decay_rate = (final_epsilon / initial_epsilon) ** (1.0 / decay_steps)
            self.scheduler = ExponentialSchedule(initial_epsilon, final_epsilon, decay_rate)
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
    
    @property
    def epsilon(self) -> float:
        """Get current epsilon value."""
        return self.scheduler._current_value if hasattr(self.scheduler, '_current_value') else self.scheduler.value(self.scheduler._step)
    
    def step(self) -> float:
        """Decay epsilon and return new value."""
        return self.scheduler.step()
    
    def should_explore(self, rng=None) -> bool:
        """
        Determine if agent should explore.
        
        Args:
            rng: Optional random number generator
            
        Returns:
            True if should explore, False if should exploit
        """
        import random
        rand = rng.random() if rng else random.random()
        return rand < self.epsilon

