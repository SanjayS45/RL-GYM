/**
 * RL-GYM Type Definitions
 */

// Environment Types
export interface Environment {
  id: string;
  name: string;
  description: string;
  observationType: 'discrete' | 'continuous';
  actionType: 'discrete' | 'continuous';
  parameters: Record<string, ParameterConfig>;
}

export interface ParameterConfig {
  type: 'int' | 'float' | 'bool' | 'string';
  default: number | boolean | string;
  min?: number;
  max?: number;
  description?: string;
  options?: string[];
}

export interface EnvironmentState {
  type: string;
  width: number;
  height: number;
  agentPosition: [number, number];
  goalPosition: [number, number];
  obstacles: Obstacle[];
}

export interface Obstacle {
  x: number;
  y: number;
  width: number;
  height: number;
  type?: 'static' | 'moving' | 'platform';
}

// Agent Types
export interface Algorithm {
  id: string;
  name: string;
  description: string;
  type: 'value_based' | 'policy_gradient' | 'actor_critic';
  actionSpace: 'discrete' | 'continuous' | 'both';
  parameters: Record<string, ParameterConfig>;
}

export interface AgentConfig {
  algorithm: string;
  learningRate: number;
  gamma: number;
  batchSize: number;
  hiddenDims: number[];
  [key: string]: unknown;
}

// Training Types
export type TrainingStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

export interface TrainingSession {
  sessionId: string;
  status: TrainingStatus;
  currentStep: number;
  currentEpisode: number;
  totalReward: number;
  startTime: string;
  elapsedTime: number;
}

export interface TrainingConfig {
  environment: string;
  algorithm: string;
  envConfig: Record<string, unknown>;
  agentConfig: Record<string, unknown>;
  trainingConfig: {
    totalTimesteps: number;
    evalFreq: number;
    logFreq: number;
    saveFreq: number;
  };
  naturalLanguageGoal?: string;
  datasetId?: string;
}

// Metrics Types
export interface MetricsDataPoint {
  step: number;
  episode: number;
  reward: number;
  loss?: number;
  epsilon?: number;
  entropy?: number;
  fps?: number;
  timestamp: number;
}

export interface TrainingMetrics {
  meanReward: number;
  maxReward: number;
  minReward: number;
  meanLoss: number;
  episodeLength: number;
  successRate?: number;
}

// Visualization Types
export interface VisualizationFrame {
  agentPosition: [number, number];
  goalPosition: [number, number];
  obstacles?: Obstacle[];
  trajectory?: [number, number][];
  reward?: number;
  step?: number;
  action?: number | number[];
}

export interface PlaybackState {
  isPlaying: boolean;
  speed: number;
  currentFrame: number;
  totalFrames: number;
}

// Dataset Types
export interface Dataset {
  id: string;
  name: string;
  type: 'demonstrations' | 'trajectories' | 'replay_buffer';
  size: number;
  numTrajectories?: number;
  numTransitions?: number;
  environment?: string;
  createdAt: string;
}

export interface DatasetUploadResponse {
  id: string;
  status: 'success' | 'error';
  message: string;
  validationResults?: {
    isValid: boolean;
    errors: string[];
    warnings: string[];
  };
}

// NLP Types
export interface ParsedGoal {
  goalType: 'reach' | 'avoid' | 'collect' | 'survive' | 'custom';
  targetDescription: string;
  constraints: string[];
  metrics: string[];
  terminationConditions: string[];
}

// WebSocket Types
export type WebSocketMessageType = 
  | 'training_update'
  | 'training_complete'
  | 'training_error'
  | 'subscribe'
  | 'ping'
  | 'pong';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  sessionId?: string;
  data?: unknown;
}

export interface TrainingUpdate {
  type: 'training_update';
  sessionId: string;
  step: number;
  episode: number;
  reward: number;
  loss?: number;
  metrics?: Record<string, number>;
  visualization?: VisualizationFrame;
}

// API Response Types
export interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
}

// UI Types
export type TabId = 'environment' | 'agent' | 'training' | 'visualization' | 'metrics' | 'datasets';

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  duration?: number;
}

