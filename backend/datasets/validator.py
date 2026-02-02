"""
Dataset Validator
Validates dataset compatibility with environments.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class ValidationResult:
    """Result of a validation check."""
    
    def __init__(self, valid: bool, message: str = "", warnings: List[str] = None):
        self.valid = valid
        self.message = message
        self.warnings = warnings or []
    
    def __bool__(self):
        return self.valid
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "message": self.message,
            "warnings": self.warnings,
        }


class DatasetValidator:
    """
    Validate datasets for RL training compatibility.
    
    Checks:
    - Required fields presence
    - Shape compatibility
    - Data type correctness
    - Value range validation
    """
    
    REQUIRED_FIELDS = ['observations', 'actions']
    OPTIONAL_FIELDS = ['rewards', 'next_observations', 'dones', 'infos']
    
    def __init__(self):
        """Initialize validator."""
        pass
    
    def validate(
        self,
        data: Dict[str, Any],
        observation_space: Optional[Tuple] = None,
        action_space: Optional[Tuple] = None
    ) -> ValidationResult:
        """
        Validate a dataset.
        
        Args:
            data: Dataset dictionary
            observation_space: Expected observation shape
            action_space: Expected action space info
            
        Returns:
            ValidationResult with validation status and messages
        """
        warnings = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                return ValidationResult(False, f"Missing required field: {field}")
        
        observations = data['observations']
        actions = data['actions']
        
        # Check array types
        if not isinstance(observations, np.ndarray):
            observations = np.array(observations)
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # Check consistent lengths
        if len(observations) != len(actions):
            return ValidationResult(
                False,
                f"Inconsistent lengths: observations ({len(observations)}) != actions ({len(actions)})"
            )
        
        if len(observations) == 0:
            return ValidationResult(False, "Dataset is empty")
        
        # Validate observation space
        if observation_space is not None:
            obs_shape = observations.shape[1:]
            if obs_shape != observation_space:
                return ValidationResult(
                    False,
                    f"Observation shape mismatch: expected {observation_space}, got {obs_shape}"
                )
        
        # Validate action space
        if action_space is not None:
            if isinstance(action_space, int):
                # Discrete action space
                if not np.issubdtype(actions.dtype, np.integer):
                    warnings.append("Actions should be integers for discrete action space")
                    
                max_action = np.max(actions)
                if max_action >= action_space:
                    return ValidationResult(
                        False,
                        f"Action value {max_action} exceeds action space size {action_space}"
                    )
            else:
                # Continuous action space
                action_shape = actions.shape[1:] if len(actions.shape) > 1 else (1,)
                if action_shape != action_space:
                    return ValidationResult(
                        False,
                        f"Action shape mismatch: expected {action_space}, got {action_shape}"
                    )
        
        # Validate optional fields if present
        if 'rewards' in data:
            rewards = data['rewards']
            if len(rewards) != len(observations):
                return ValidationResult(
                    False,
                    f"Rewards length mismatch: expected {len(observations)}, got {len(rewards)}"
                )
            
            # Check for NaN/Inf
            if np.any(~np.isfinite(rewards)):
                warnings.append("Rewards contain NaN or Inf values")
        
        if 'next_observations' in data:
            next_obs = data['next_observations']
            if len(next_obs) != len(observations):
                return ValidationResult(
                    False,
                    f"Next observations length mismatch"
                )
            if next_obs.shape[1:] != observations.shape[1:]:
                return ValidationResult(
                    False,
                    f"Next observation shape mismatch"
                )
        
        if 'dones' in data:
            dones = data['dones']
            if len(dones) != len(observations):
                return ValidationResult(
                    False,
                    f"Dones length mismatch"
                )
        
        # Check for NaN/Inf in observations
        if np.any(~np.isfinite(observations)):
            warnings.append("Observations contain NaN or Inf values")
        
        # Check for NaN/Inf in actions (for continuous)
        if np.issubdtype(actions.dtype, np.floating):
            if np.any(~np.isfinite(actions)):
                warnings.append("Actions contain NaN or Inf values")
        
        return ValidationResult(
            True,
            f"Dataset valid: {len(observations)} samples",
            warnings
        )
    
    def validate_for_algorithm(
        self,
        data: Dict[str, Any],
        algorithm: str
    ) -> ValidationResult:
        """
        Validate dataset compatibility with specific algorithm.
        
        Args:
            data: Dataset dictionary
            algorithm: Algorithm name (DQN, PPO, SAC, etc.)
            
        Returns:
            ValidationResult
        """
        warnings = []
        
        # Base validation
        base_result = self.validate(data)
        if not base_result:
            return base_result
        
        observations = data['observations']
        actions = data['actions']
        
        # Algorithm-specific checks
        if algorithm == 'DQN':
            # DQN requires discrete actions
            if not np.issubdtype(actions.dtype, np.integer):
                return ValidationResult(
                    False,
                    "DQN requires discrete (integer) actions"
                )
        
        elif algorithm == 'SAC':
            # SAC requires continuous actions
            if np.issubdtype(actions.dtype, np.integer):
                return ValidationResult(
                    False,
                    "SAC requires continuous actions"
                )
        
        elif algorithm in ['PPO', 'A2C']:
            # PPO/A2C work with both, but prefer having returns
            if 'rewards' not in data:
                warnings.append(f"{algorithm} benefits from reward data for return estimation")
        
        # Check for offline RL requirements
        if 'next_observations' not in data:
            if algorithm in ['DQN', 'SAC']:
                return ValidationResult(
                    False,
                    f"{algorithm} requires next_observations for TD learning"
                )
        
        return ValidationResult(
            True,
            f"Dataset compatible with {algorithm}",
            base_result.warnings + warnings
        )
    
    def check_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check dataset quality metrics.
        
        Args:
            data: Dataset dictionary
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {}
        
        observations = np.array(data['observations'])
        actions = np.array(data['actions'])
        
        # Size metrics
        metrics['num_samples'] = len(observations)
        metrics['observation_dim'] = observations.shape[1:] if len(observations.shape) > 1 else (1,)
        
        # Observation statistics
        metrics['observation_stats'] = {
            'mean': float(np.mean(observations)),
            'std': float(np.std(observations)),
            'min': float(np.min(observations)),
            'max': float(np.max(observations)),
        }
        
        # Action statistics
        if np.issubdtype(actions.dtype, np.integer):
            unique, counts = np.unique(actions, return_counts=True)
            metrics['action_distribution'] = {
                int(a): int(c) for a, c in zip(unique, counts)
            }
            metrics['action_entropy'] = float(self._entropy(counts / counts.sum()))
        else:
            metrics['action_stats'] = {
                'mean': float(np.mean(actions)),
                'std': float(np.std(actions)),
                'min': float(np.min(actions)),
                'max': float(np.max(actions)),
            }
        
        # Reward statistics if available
        if 'rewards' in data:
            rewards = np.array(data['rewards'])
            metrics['reward_stats'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'positive_ratio': float(np.mean(rewards > 0)),
            }
        
        # Episode statistics if dones available
        if 'dones' in data:
            dones = np.array(data['dones'])
            episode_ends = np.where(dones)[0]
            if len(episode_ends) > 0:
                episode_lengths = np.diff(np.concatenate([[-1], episode_ends]))
                metrics['episode_stats'] = {
                    'num_episodes': len(episode_ends),
                    'mean_length': float(np.mean(episode_lengths)),
                    'std_length': float(np.std(episode_lengths)),
                    'min_length': int(np.min(episode_lengths)),
                    'max_length': int(np.max(episode_lengths)),
                }
        
        return metrics
    
    def _entropy(self, probs: np.ndarray) -> float:
        """Calculate entropy of probability distribution."""
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))

