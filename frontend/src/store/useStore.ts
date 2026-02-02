import { create } from 'zustand'

interface TrainingState {
  status: 'idle' | 'running' | 'paused' | 'completed' | 'error'
  sessionId: string | null
  currentStep: number
  currentEpisode: number
  totalReward: number
  metrics: {
    meanReward: number
    maxReward: number
    loss: number
    episodeLength: number
  }
  history: {
    rewards: number[]
    losses: number[]
    steps: number[]
  }
}

interface MetricsDataPoint {
  step: number
  episode: number
  reward: number
  loss?: number
  epsilon?: number
  fps?: number
  timestamp: number
  [key: string]: number | undefined
}

interface VisualizationState {
  agentPosition: [number, number]
  goalPosition: [number, number]
  obstacles?: Array<{ x: number; y: number; width: number; height: number }>
  trajectory?: Array<[number, number]>
}

interface EnvironmentState {
  type: string
  config: string
  width: number
  height: number
  obstacles: any[]
  goals: any[]
  agentPosition: [number, number]
  agentVelocity: [number, number]
}

interface AgentState {
  algorithm: string
  learningRate: number
  gamma: number
  batchSize: number
  nSteps: number
  nEpochs: number
  clipRange: number
  entCoef: number
  hiddenDims: number[]
}

interface AppState {
  // UI state
  activeTab: string
  setActiveTab: (tab: string) => void
  sidebarCollapsed: boolean
  setSidebarCollapsed: (collapsed: boolean) => void

  // Training state
  training: TrainingState
  isTraining: boolean
  setIsTraining: (isTraining: boolean) => void
  setTrainingStatus: (status: TrainingState['status']) => void
  setSessionId: (id: string | null) => void
  updateTrainingMetrics: (metrics: Partial<TrainingState['metrics']>) => void
  updateTrainingProgress: (step: number, episode: number, reward: number) => void
  addToHistory: (reward: number, loss: number) => void
  resetTraining: () => void

  // Metrics data for charts
  metricsData: MetricsDataPoint[]
  addMetricsData: (data: MetricsDataPoint) => void
  clearMetricsData: () => void

  // Visualization state
  visualization: VisualizationState
  setVisualizationState: (state: Partial<VisualizationState>) => void

  // Environment state
  environment: EnvironmentState
  setEnvironmentType: (type: string) => void
  setEnvironmentConfig: (config: string) => void
  updateEnvironmentState: (state: Partial<EnvironmentState>) => void
  addObstacle: (obstacle: any) => void
  removeObstacle: (index: number) => void
  clearObstacles: () => void

  // Agent state
  agent: AgentState
  setAlgorithm: (algorithm: string) => void
  setHyperparameter: (key: keyof AgentState, value: any) => void
  resetAgentToDefaults: (algorithm: string) => void

  // WebSocket connection
  wsConnected: boolean
  setWsConnected: (connected: boolean) => void

  // Goal text
  goalText: string
  setGoalText: (text: string) => void

  // Playback controls
  playbackSpeed: number
  setPlaybackSpeed: (speed: number) => void
  isPlaying: boolean
  setIsPlaying: (playing: boolean) => void
}

const initialTrainingState: TrainingState = {
  status: 'idle',
  sessionId: null,
  currentStep: 0,
  currentEpisode: 0,
  totalReward: 0,
  metrics: {
    meanReward: 0,
    maxReward: 0,
    loss: 0,
    episodeLength: 0,
  },
  history: {
    rewards: [],
    losses: [],
    steps: [],
  },
}

const initialVisualizationState: VisualizationState = {
  agentPosition: [100, 300],
  goalPosition: [700, 300],
  obstacles: [],
  trajectory: [],
}

const initialEnvironmentState: EnvironmentState = {
  type: 'navigation',
  config: 'simple_obstacles',
  width: 800,
  height: 600,
  obstacles: [],
  goals: [{ position: [700, 300], size: [40, 40] }],
  agentPosition: [100, 300],
  agentVelocity: [0, 0],
}

const algorithmDefaults: Record<string, Partial<AgentState>> = {
  PPO: {
    learningRate: 3e-4,
    gamma: 0.99,
    batchSize: 64,
    nSteps: 2048,
    nEpochs: 10,
    clipRange: 0.2,
    entCoef: 0.01,
  },
  DQN: {
    learningRate: 1e-4,
    gamma: 0.99,
    batchSize: 64,
    nSteps: 4,
    nEpochs: 1,
    clipRange: 0,
    entCoef: 0,
  },
  SAC: {
    learningRate: 3e-4,
    gamma: 0.99,
    batchSize: 256,
    nSteps: 1,
    nEpochs: 1,
    clipRange: 0,
    entCoef: 0,
  },
  A2C: {
    learningRate: 7e-4,
    gamma: 0.99,
    batchSize: 64,
    nSteps: 5,
    nEpochs: 1,
    clipRange: 0,
    entCoef: 0.01,
  },
}

const initialAgentState: AgentState = {
  algorithm: 'PPO',
  learningRate: 3e-4,
  gamma: 0.99,
  batchSize: 64,
  nSteps: 2048,
  nEpochs: 10,
  clipRange: 0.2,
  entCoef: 0.01,
  hiddenDims: [256, 256],
}

export const useStore = create<AppState>((set, get) => ({
  // UI state
  activeTab: 'environment',
  setActiveTab: (activeTab) => set({ activeTab }),
  sidebarCollapsed: false,
  setSidebarCollapsed: (sidebarCollapsed) => set({ sidebarCollapsed }),

  // Training state
  training: initialTrainingState,
  isTraining: false,
  setIsTraining: (isTraining) => set({ isTraining }),
  setTrainingStatus: (status) =>
    set((state) => ({
      training: { ...state.training, status },
      isTraining: status === 'running',
    })),
  setSessionId: (sessionId) =>
    set((state) => ({ training: { ...state.training, sessionId } })),
  updateTrainingMetrics: (metrics) =>
    set((state) => ({
      training: {
        ...state.training,
        metrics: { ...state.training.metrics, ...metrics },
      },
    })),
  updateTrainingProgress: (currentStep, currentEpisode, totalReward) =>
    set((state) => ({
      training: {
        ...state.training,
        currentStep,
        currentEpisode,
        totalReward,
      },
    })),
  addToHistory: (reward, loss) =>
    set((state) => ({
      training: {
        ...state.training,
        history: {
          rewards: [...state.training.history.rewards.slice(-499), reward],
          losses: [...state.training.history.losses.slice(-499), loss],
          steps: [...state.training.history.steps.slice(-499), state.training.currentStep],
        },
      },
    })),
  resetTraining: () =>
    set({
      training: initialTrainingState,
      isTraining: false,
      metricsData: [],
    }),

  // Metrics data
  metricsData: [],
  addMetricsData: (data) =>
    set((state) => ({
      metricsData: [...state.metricsData.slice(-499), data],
      training: {
        ...state.training,
        currentStep: data.step,
        currentEpisode: data.episode,
        totalReward: state.training.totalReward + data.reward,
      },
    })),
  clearMetricsData: () => set({ metricsData: [] }),

  // Visualization state
  visualization: initialVisualizationState,
  setVisualizationState: (newState) =>
    set((state) => ({
      visualization: { ...state.visualization, ...newState },
    })),

  // Environment state
  environment: initialEnvironmentState,
  setEnvironmentType: (type) =>
    set((state) => ({ environment: { ...state.environment, type } })),
  setEnvironmentConfig: (config) =>
    set((state) => ({ environment: { ...state.environment, config } })),
  updateEnvironmentState: (newState) =>
    set((state) => ({ environment: { ...state.environment, ...newState } })),
  addObstacle: (obstacle) =>
    set((state) => ({
      environment: {
        ...state.environment,
        obstacles: [...state.environment.obstacles, obstacle],
      },
    })),
  removeObstacle: (index) =>
    set((state) => ({
      environment: {
        ...state.environment,
        obstacles: state.environment.obstacles.filter((_, i) => i !== index),
      },
    })),
  clearObstacles: () =>
    set((state) => ({ environment: { ...state.environment, obstacles: [] } })),

  // Agent state
  agent: initialAgentState,
  setAlgorithm: (algorithm) =>
    set((state) => ({
      agent: {
        ...state.agent,
        algorithm,
        ...algorithmDefaults[algorithm],
      },
    })),
  setHyperparameter: (key, value) =>
    set((state) => ({ agent: { ...state.agent, [key]: value } })),
  resetAgentToDefaults: (algorithm) =>
    set((state) => ({
      agent: {
        ...initialAgentState,
        algorithm,
        ...algorithmDefaults[algorithm],
      },
    })),

  // WebSocket
  wsConnected: false,
  setWsConnected: (connected) => set({ wsConnected: connected }),

  // Goal
  goalText: '',
  setGoalText: (goalText) => set({ goalText }),

  // Playback controls
  playbackSpeed: 1,
  setPlaybackSpeed: (playbackSpeed) => set({ playbackSpeed }),
  isPlaying: true,
  setIsPlaying: (isPlaying) => set({ isPlaying }),
}))
