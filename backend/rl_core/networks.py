"""
Neural network architectures for RL policies and value functions.
"""

from typing import List, Tuple, Optional, Type
import torch
import torch.nn as nn
import numpy as np


def get_activation(activation: str) -> Type[nn.Module]:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    return activations.get(activation.lower(), nn.ReLU)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        output_activation: Output activation (None for linear)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation)())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(get_activation(output_activation)())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network for image-based observations.
    
    Uses the Nature DQN architecture by default.
    
    Args:
        input_channels: Number of input channels
        output_dim: Output dimension (typically fed to MLP)
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 512,
        input_height: int = 84,
        input_width: int = 84,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Nature DQN architecture
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv(dummy_input)
            conv_out_size = conv_out.shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.ReLU(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        """
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        return self.fc(self.conv(x))


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared or separate feature extractors.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dims: Hidden layer dimensions
        shared_layers: Number of shared layers (0 = separate networks)
        continuous: Whether action space is continuous
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        shared_layers: int = 0,
        continuous: bool = False,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Shared feature extractor (optional)
        if shared_layers > 0:
            shared_hidden = hidden_dims[:shared_layers]
            remaining_hidden = hidden_dims[shared_layers:]
            
            self.shared = MLP(obs_dim, shared_hidden[-1], shared_hidden[:-1])
            feature_dim = shared_hidden[-1]
        else:
            self.shared = None
            feature_dim = obs_dim
            remaining_hidden = hidden_dims
        
        # Actor network
        if continuous:
            # Output mean and log_std for Gaussian policy
            self.actor_mean = MLP(feature_dim, action_dim, remaining_hidden)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            # Output action logits for categorical policy
            self.actor = MLP(feature_dim, action_dim, remaining_hidden)
        
        # Critic network (value function)
        self.critic = MLP(feature_dim, 1, remaining_hidden)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy output and value.
        
        Returns:
            policy_output: Action logits (discrete) or (mean, log_std) (continuous)
            value: State value estimate
        """
        if self.shared is not None:
            features = self.shared(obs)
        else:
            features = obs
        
        value = self.critic(features)
        
        if self.continuous:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std.expand_as(mean)
            return (mean, log_std), value
        else:
            logits = self.actor(features)
            return logits, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        if self.shared is not None:
            features = self.shared(obs)
        else:
            features = obs
        return self.critic(features)
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation tensor
            action: Optional action to evaluate (if None, sample new action)
            
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value estimate
        """
        policy_output, value = self.forward(obs)
        
        if self.continuous:
            mean, log_std = policy_output
            std = log_std.exp()
            
            # Create normal distribution
            dist = torch.distributions.Normal(mean, std)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = policy_output
            dist = torch.distributions.Categorical(logits=logits)
            
            if action is None:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)

