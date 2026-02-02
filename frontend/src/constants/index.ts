/**
 * RL-GYM Constants
 */

// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

// Training Configuration
export const DEFAULT_TIMESTEPS = 100000;
export const DEFAULT_EVAL_FREQ = 1000;
export const DEFAULT_LOG_FREQ = 100;
export const DEFAULT_SAVE_FREQ = 10000;

// Visualization Configuration
export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 600;
export const DEFAULT_FPS = 30;
export const MAX_TRAJECTORY_LENGTH = 500;
export const MAX_METRICS_HISTORY = 500;

// WebSocket Configuration
export const WS_RECONNECT_INTERVAL = 3000;
export const WS_MAX_RECONNECT_ATTEMPTS = 5;
export const WS_PING_INTERVAL = 30000;

// UI Configuration
export const SIDEBAR_WIDTH = 280;
export const SIDEBAR_COLLAPSED_WIDTH = 64;
export const HEADER_HEIGHT = 64;

// Chart Configuration
export const CHART_COLORS = {
  reward: '#64ffda',
  loss: '#ff6b6b',
  epsilon: '#ffd93d',
  entropy: '#6bcb77',
  length: '#4ecdc4',
} as const;

// Environment Types
export const ENVIRONMENTS = [
  {
    id: 'gridworld',
    name: 'Grid World',
    description: 'Discrete navigation on a grid',
    icon: '🎯',
  },
  {
    id: 'navigation',
    name: 'Navigation',
    description: 'Continuous 2D navigation with lidar',
    icon: '🧭',
  },
  {
    id: 'platformer',
    name: 'Platformer',
    description: '2D platformer with jumping',
    icon: '🎮',
  },
] as const;

// Algorithm Types
export const ALGORITHMS = [
  {
    id: 'dqn',
    name: 'DQN',
    fullName: 'Deep Q-Network',
    description: 'Value-based algorithm with experience replay',
    type: 'value_based',
    actionSpace: 'discrete',
  },
  {
    id: 'ppo',
    name: 'PPO',
    fullName: 'Proximal Policy Optimization',
    description: 'Policy gradient with clipped objective',
    type: 'policy_gradient',
    actionSpace: 'both',
  },
  {
    id: 'sac',
    name: 'SAC',
    fullName: 'Soft Actor-Critic',
    description: 'Off-policy with entropy regularization',
    type: 'actor_critic',
    actionSpace: 'continuous',
  },
  {
    id: 'a2c',
    name: 'A2C',
    fullName: 'Advantage Actor-Critic',
    description: 'Synchronous actor-critic algorithm',
    type: 'actor_critic',
    actionSpace: 'both',
  },
] as const;

// Default Hyperparameters
export const DEFAULT_HYPERPARAMETERS = {
  dqn: {
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_end: 0.01,
    epsilon_decay: 0.995,
    buffer_size: 10000,
    batch_size: 64,
    target_update_freq: 100,
  },
  ppo: {
    learning_rate: 0.0003,
    gamma: 0.99,
    gae_lambda: 0.95,
    clip_epsilon: 0.2,
    value_coef: 0.5,
    entropy_coef: 0.01,
    n_steps: 2048,
    n_epochs: 10,
    batch_size: 64,
  },
  sac: {
    learning_rate: 0.0003,
    gamma: 0.99,
    tau: 0.005,
    alpha: 0.2,
    auto_alpha: true,
    buffer_size: 100000,
    batch_size: 256,
  },
  a2c: {
    learning_rate: 0.0007,
    gamma: 0.99,
    value_coef: 0.5,
    entropy_coef: 0.01,
    n_steps: 5,
    max_grad_norm: 0.5,
  },
} as const;

// Training Status Colors
export const STATUS_COLORS = {
  idle: '#888888',
  running: '#64ffda',
  paused: '#ffd93d',
  completed: '#6bcb77',
  error: '#ff6b6b',
} as const;

// Playback Speeds
export const PLAYBACK_SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

// Keyboard Shortcuts
export const KEYBOARD_SHORTCUTS = {
  startTraining: 'Space',
  pauseTraining: 'Space',
  stopTraining: 'Escape',
  toggleSidebar: 'b',
  nextTab: 'Tab',
  speedUp: ']',
  speedDown: '[',
} as const;

// Notification Durations
export const NOTIFICATION_DURATION = {
  info: 3000,
  success: 3000,
  warning: 5000,
  error: 10000,
} as const;

// File Upload Limits
export const MAX_DATASET_SIZE = 100 * 1024 * 1024; // 100MB
export const ALLOWED_DATASET_TYPES = ['.json', '.npz', '.pkl', '.h5'];

// Local Storage Keys
export const STORAGE_KEYS = {
  theme: 'rl-gym-theme',
  sidebarCollapsed: 'rl-gym-sidebar-collapsed',
  recentSessions: 'rl-gym-recent-sessions',
  preferences: 'rl-gym-preferences',
} as const;

