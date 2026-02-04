"""
Dataset Loader
Loads datasets from various formats.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np


class DatasetLoader:
    """
    Load datasets from various file formats.
    
    Supports:
    - JSON trajectories
    - CSV state-action logs
    - HDF5 compressed datasets
    """
    
    SUPPORTED_FORMATS = ['json', 'csv', 'h5', 'hdf5', 'npz']
    
    def __init__(self, base_path: Optional[Union[Path, str]] = None):
        """
        Initialize dataset loader.
        
        Args:
            base_path: Base directory for dataset files
        """
        if base_path is None:
            self.base_path = Path("datasets")
        elif isinstance(base_path, str):
            self.base_path = Path(base_path)
        else:
            self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load a dataset from file.
        
        Args:
            path: Path to dataset file
            
        Returns:
            Dictionary containing loaded data
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        suffix = file_path.suffix.lower().lstrip('.')
        
        if suffix == 'json':
            return self._load_json(file_path)
        elif suffix == 'csv':
            return self._load_csv(file_path)
        elif suffix in ['h5', 'hdf5']:
            return self._load_hdf5(file_path)
        elif suffix == 'npz':
            return self._load_npz(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON dataset."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # List of transitions
            return self._parse_transitions(data)
        elif isinstance(data, dict):
            if 'trajectories' in data:
                # Dictionary with trajectories key
                return self._parse_trajectories(data['trajectories'])
            elif 'observations' in data:
                # Flat arrays
                return data
            else:
                raise ValueError("Unknown JSON structure")
        else:
            raise ValueError("Invalid JSON data type")
    
    def _load_csv(self, path: Path) -> Dict[str, Any]:
        """Load CSV dataset."""
        import csv
        
        observations = []
        actions = []
        rewards = []
        dones = []
        
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'observation' in row:
                    obs = json.loads(row['observation'])
                    observations.append(obs)
                if 'action' in row:
                    action = json.loads(row['action']) if row['action'].startswith('[') else float(row['action'])
                    actions.append(action)
                if 'reward' in row:
                    rewards.append(float(row['reward']))
                if 'done' in row:
                    dones.append(row['done'].lower() == 'true')
        
        return {
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
        }
    
    def _load_hdf5(self, path: Path) -> Dict[str, Any]:
        """Load HDF5 dataset."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for HDF5 support")
        
        data = {}
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
        
        return data
    
    def _load_npz(self, path: Path) -> Dict[str, Any]:
        """Load NPZ dataset."""
        loaded = np.load(path, allow_pickle=True)
        return {key: loaded[key] for key in loaded.files}
    
    def _parse_transitions(self, transitions: List[Dict]) -> Dict[str, Any]:
        """Parse list of transition dictionaries."""
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        
        for t in transitions:
            observations.append(t.get('observation', t.get('obs', [])))
            actions.append(t.get('action', 0))
            rewards.append(t.get('reward', 0))
            next_observations.append(t.get('next_observation', t.get('next_obs', [])))
            dones.append(t.get('done', t.get('terminal', False)))
        
        return {
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions),
            'rewards': np.array(rewards, dtype=np.float32),
            'next_observations': np.array(next_observations, dtype=np.float32),
            'dones': np.array(dones, dtype=bool),
        }
    
    def _parse_trajectories(self, trajectories: List[List[Dict]]) -> Dict[str, Any]:
        """Parse list of episode trajectories."""
        all_transitions = []
        for trajectory in trajectories:
            all_transitions.extend(trajectory)
        return self._parse_transitions(all_transitions)
    
    def save(self, data: Dict[str, Any], path: str, format: str = 'npz') -> str:
        """
        Save dataset to file.
        
        Args:
            data: Dataset dictionary
            path: Output file path
            format: Output format
            
        Returns:
            Path to saved file
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'npz':
            np.savez(file_path, **data)
        elif format == 'json':
            # Convert numpy arrays to lists
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
            with open(file_path, 'w') as f:
                json.dump(json_data, f)
        elif format in ['h5', 'hdf5']:
            try:
                import h5py
            except ImportError:
                raise ImportError("h5py required for HDF5 support")
            
            with h5py.File(file_path, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(file_path)
    
    def get_dataset_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a loaded dataset.
        
        Args:
            data: Loaded dataset
            
        Returns:
            Dictionary with dataset info
        """
        info = {
            'keys': list(data.keys()),
            'num_samples': 0,
        }
        
        if 'observations' in data:
            obs = data['observations']
            info['num_samples'] = len(obs)
            info['observation_shape'] = obs.shape[1:] if len(obs.shape) > 1 else (1,)
            info['observation_dtype'] = str(obs.dtype)
        
        if 'actions' in data:
            actions = data['actions']
            info['action_shape'] = actions.shape[1:] if len(actions.shape) > 1 else (1,)
            info['action_dtype'] = str(actions.dtype)
            
            # Detect discrete vs continuous
            if np.issubdtype(actions.dtype, np.integer):
                info['action_type'] = 'discrete'
                info['num_actions'] = int(np.max(actions)) + 1
            else:
                info['action_type'] = 'continuous'
        
        if 'rewards' in data:
            rewards = data['rewards']
            info['reward_stats'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            }
        
        return info

