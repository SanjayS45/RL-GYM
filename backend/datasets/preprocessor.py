"""
Dataset Preprocessor
Preprocesses datasets for training.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class DatasetPreprocessor:
    """
    Preprocess datasets for RL training.
    
    Features:
    - Normalization (observation, action, reward)
    - Data augmentation
    - Trajectory splitting
    - Reward computation
    """
    
    def __init__(self):
        """Initialize preprocessor."""
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None
        self.reward_mean = None
        self.reward_std = None
    
    def fit(self, data: Dict[str, Any]) -> "DatasetPreprocessor":
        """
        Compute statistics from dataset.
        
        Args:
            data: Dataset dictionary
            
        Returns:
            self
        """
        observations = np.array(data['observations'])
        self.obs_mean = np.mean(observations, axis=0)
        self.obs_std = np.std(observations, axis=0) + 1e-8
        
        actions = np.array(data['actions'])
        if np.issubdtype(actions.dtype, np.floating):
            self.action_mean = np.mean(actions, axis=0)
            self.action_std = np.std(actions, axis=0) + 1e-8
        
        if 'rewards' in data:
            rewards = np.array(data['rewards'])
            self.reward_mean = np.mean(rewards)
            self.reward_std = np.std(rewards) + 1e-8
        
        return self
    
    def transform(
        self,
        data: Dict[str, Any],
        normalize_obs: bool = True,
        normalize_actions: bool = False,
        normalize_rewards: bool = False
    ) -> Dict[str, Any]:
        """
        Transform dataset.
        
        Args:
            data: Dataset dictionary
            normalize_obs: Normalize observations
            normalize_actions: Normalize actions
            normalize_rewards: Normalize rewards
            
        Returns:
            Transformed dataset
        """
        result = {}
        
        # Transform observations
        observations = np.array(data['observations'], dtype=np.float32)
        if normalize_obs and self.obs_mean is not None:
            observations = (observations - self.obs_mean) / self.obs_std
        result['observations'] = observations
        
        # Transform next observations if present
        if 'next_observations' in data:
            next_obs = np.array(data['next_observations'], dtype=np.float32)
            if normalize_obs and self.obs_mean is not None:
                next_obs = (next_obs - self.obs_mean) / self.obs_std
            result['next_observations'] = next_obs
        
        # Transform actions
        actions = np.array(data['actions'])
        if normalize_actions and self.action_mean is not None:
            if np.issubdtype(actions.dtype, np.floating):
                actions = (actions - self.action_mean) / self.action_std
        result['actions'] = actions
        
        # Transform rewards
        if 'rewards' in data:
            rewards = np.array(data['rewards'], dtype=np.float32)
            if normalize_rewards and self.reward_mean is not None:
                rewards = (rewards - self.reward_mean) / self.reward_std
            result['rewards'] = rewards
        
        # Copy other fields
        if 'dones' in data:
            result['dones'] = np.array(data['dones'], dtype=bool)
        
        if 'infos' in data:
            result['infos'] = data['infos']
        
        return result
    
    def fit_transform(
        self,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data, **kwargs)
    
    def compute_returns(
        self,
        data: Dict[str, Any],
        gamma: float = 0.99
    ) -> Dict[str, Any]:
        """
        Compute discounted returns from rewards.
        
        Args:
            data: Dataset with rewards and dones
            gamma: Discount factor
            
        Returns:
            Dataset with 'returns' field added
        """
        if 'rewards' not in data:
            raise ValueError("Dataset must contain rewards")
        
        rewards = np.array(data['rewards'])
        dones = np.array(data.get('dones', np.zeros(len(rewards), dtype=bool)))
        
        # Compute returns using backward pass
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        result = dict(data)
        result['returns'] = returns
        return result
    
    def compute_advantages(
        self,
        data: Dict[str, Any],
        values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute GAE advantages.
        
        Args:
            data: Dataset with rewards and dones
            values: Value function estimates
            gamma: Discount factor
            gae_lambda: GAE lambda
            
        Returns:
            Dataset with 'advantages' and 'returns' fields added
        """
        if 'rewards' not in data:
            raise ValueError("Dataset must contain rewards")
        
        rewards = np.array(data['rewards'])
        dones = np.array(data.get('dones', np.zeros(len(rewards), dtype=bool)))
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        
        result = dict(data)
        result['advantages'] = advantages
        result['returns'] = returns
        return result
    
    def split_into_trajectories(
        self,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split dataset into individual trajectories.
        
        Args:
            data: Dataset with dones field
            
        Returns:
            List of trajectory dictionaries
        """
        if 'dones' not in data:
            # Return single trajectory
            return [data]
        
        dones = np.array(data['dones'])
        episode_ends = np.where(dones)[0]
        
        if len(episode_ends) == 0:
            return [data]
        
        trajectories = []
        start = 0
        
        for end in episode_ends:
            traj = {}
            for key, value in data.items():
                arr = np.array(value)
                traj[key] = arr[start:end + 1]
            trajectories.append(traj)
            start = end + 1
        
        # Handle remaining data if last transition wasn't done
        if start < len(dones):
            traj = {}
            for key, value in data.items():
                arr = np.array(value)
                traj[key] = arr[start:]
            trajectories.append(traj)
        
        return trajectories
    
    def merge_trajectories(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge multiple trajectories into single dataset.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Merged dataset
        """
        if len(trajectories) == 0:
            return {}
        
        result = {}
        keys = trajectories[0].keys()
        
        for key in keys:
            arrays = [np.array(traj[key]) for traj in trajectories]
            result[key] = np.concatenate(arrays, axis=0)
        
        return result
    
    def subsample(
        self,
        data: Dict[str, Any],
        ratio: float = 0.5,
        preserve_episodes: bool = True
    ) -> Dict[str, Any]:
        """
        Subsample dataset.
        
        Args:
            data: Dataset dictionary
            ratio: Fraction of data to keep
            preserve_episodes: Keep full episodes when subsampling
            
        Returns:
            Subsampled dataset
        """
        if preserve_episodes and 'dones' in data:
            trajectories = self.split_into_trajectories(data)
            num_keep = max(1, int(len(trajectories) * ratio))
            indices = np.random.choice(len(trajectories), num_keep, replace=False)
            selected = [trajectories[i] for i in indices]
            return self.merge_trajectories(selected)
        else:
            n = len(data['observations'])
            num_keep = max(1, int(n * ratio))
            indices = np.random.choice(n, num_keep, replace=False)
            indices = np.sort(indices)
            
            result = {}
            for key, value in data.items():
                arr = np.array(value)
                result[key] = arr[indices]
            
            return result
    
    def augment(
        self,
        data: Dict[str, Any],
        noise_scale: float = 0.01
    ) -> Dict[str, Any]:
        """
        Augment dataset with noise.
        
        Args:
            data: Dataset dictionary
            noise_scale: Scale of Gaussian noise to add
            
        Returns:
            Augmented dataset
        """
        result = dict(data)
        
        # Add noise to observations
        observations = np.array(data['observations'], dtype=np.float32)
        noise = np.random.randn(*observations.shape).astype(np.float32) * noise_scale
        result['observations'] = observations + noise
        
        # Add noise to next observations if present
        if 'next_observations' in data:
            next_obs = np.array(data['next_observations'], dtype=np.float32)
            noise = np.random.randn(*next_obs.shape).astype(np.float32) * noise_scale
            result['next_observations'] = next_obs + noise
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get computed statistics."""
        return {
            'obs_mean': self.obs_mean.tolist() if self.obs_mean is not None else None,
            'obs_std': self.obs_std.tolist() if self.obs_std is not None else None,
            'action_mean': self.action_mean.tolist() if self.action_mean is not None else None,
            'action_std': self.action_std.tolist() if self.action_std is not None else None,
            'reward_mean': self.reward_mean if self.reward_mean is not None else None,
            'reward_std': self.reward_std if self.reward_std is not None else None,
        }

