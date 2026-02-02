/**
 * API hooks for RL-GYM frontend
 * Provides React hooks for interacting with the backend API
 */

import { useState, useCallback } from 'react';

// API base URL - configurable via environment variable
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Generic API response type
 */
interface ApiResponse<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/**
 * Environment configuration
 */
export interface EnvironmentConfig {
  env_type: string;
  width?: number;
  height?: number;
  seed?: number;
  obstacles?: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    obstacle_type: string;
  }>;
  natural_language_goal?: string;
}

/**
 * Agent configuration
 */
export interface AgentConfig {
  algorithm: string;
  state_dim: number;
  action_dim: number;
  hidden_dims?: number[];
  learning_rate?: number;
  gamma?: number;
  additional_params?: Record<string, unknown>;
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  environment: string;
  algorithm: string;
  env_config?: Record<string, unknown>;
  agent_config?: Record<string, unknown>;
  training_config?: Record<string, unknown>;
  natural_language_goal?: string;
  dataset_id?: string;
}

/**
 * Training session info
 */
export interface TrainingSession {
  session_id: string;
  status: string;
  current_step: number;
  current_episode: number;
  metrics?: Array<Record<string, unknown>>;
}

/**
 * Algorithm info
 */
export interface AlgorithmInfo {
  id: string;
  name: string;
  description: string;
  type: string;
  action_space: string;
  parameters: Record<string, {
    type: string;
    default: unknown;
    min?: number;
    max?: number;
    description?: string;
  }>;
}

/**
 * Environment info
 */
export interface EnvironmentInfo {
  id: string;
  name: string;
  description: string;
  observation_type: string;
  action_type: string;
  parameters: Record<string, {
    type: string;
    default: unknown;
    min?: number;
    max?: number;
    description?: string;
  }>;
}

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return response.json();
}

/**
 * Hook for fetching available environments
 */
export function useEnvironments() {
  const [state, setState] = useState<ApiResponse<EnvironmentInfo[]>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const fetchEnvironments = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<{ environments: EnvironmentInfo[] }>('/environments/list');
      setState({ data: response.environments, error: null, loading: false });
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
    }
  }, []);
  
  return { ...state, fetchEnvironments };
}

/**
 * Hook for fetching available algorithms
 */
export function useAlgorithms() {
  const [state, setState] = useState<ApiResponse<AlgorithmInfo[]>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const fetchAlgorithms = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<{ algorithms: AlgorithmInfo[] }>('/agents/algorithms');
      setState({ data: response.algorithms, error: null, loading: false });
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
    }
  }, []);
  
  return { ...state, fetchAlgorithms };
}

/**
 * Hook for managing training sessions
 */
export function useTraining() {
  const [state, setState] = useState<ApiResponse<TrainingSession>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const startTraining = useCallback(async (config: TrainingConfig) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<TrainingSession>('/training/start', {
        method: 'POST',
        body: JSON.stringify(config),
      });
      setState({ data: response, error: null, loading: false });
      return response;
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
      throw error;
    }
  }, []);
  
  const stopTraining = useCallback(async (sessionId: string) => {
    try {
      await apiFetch(`/training/stop/${sessionId}`, { method: 'POST' });
      setState(prev => ({
        ...prev,
        data: prev.data ? { ...prev.data, status: 'stopped' } : null,
      }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    }
  }, []);
  
  const pauseTraining = useCallback(async (sessionId: string) => {
    try {
      await apiFetch(`/training/pause/${sessionId}`, { method: 'POST' });
      setState(prev => ({
        ...prev,
        data: prev.data ? { ...prev.data, status: 'paused' } : null,
      }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    }
  }, []);
  
  const resumeTraining = useCallback(async (sessionId: string) => {
    try {
      await apiFetch(`/training/resume/${sessionId}`, { method: 'POST' });
      setState(prev => ({
        ...prev,
        data: prev.data ? { ...prev.data, status: 'running' } : null,
      }));
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
    }
  }, []);
  
  const getStatus = useCallback(async (sessionId: string) => {
    try {
      const response = await apiFetch<TrainingSession>(`/training/status/${sessionId}`);
      setState({ data: response, error: null, loading: false });
      return response;
    } catch (error) {
      setState(prev => ({ ...prev, error: (error as Error).message }));
      throw error;
    }
  }, []);
  
  return {
    ...state,
    startTraining,
    stopTraining,
    pauseTraining,
    resumeTraining,
    getStatus,
  };
}

/**
 * Hook for managing datasets
 */
export function useDatasets() {
  const [state, setState] = useState<ApiResponse<Array<{
    id: string;
    name: string;
    type: string;
    size: number;
  }>>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const fetchDatasets = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<{ datasets: Array<{
        id: string;
        name: string;
        type: string;
        size: number;
      }> }>('/datasets/list');
      setState({ data: response.datasets, error: null, loading: false });
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
    }
  }, []);
  
  const uploadDataset = useCallback(async (file: File, metadata?: {
    name?: string;
    type?: string;
    environment?: string;
    description?: string;
  }) => {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata?.name) formData.append('name', metadata.name);
    if (metadata?.type) formData.append('dataset_type', metadata.type);
    if (metadata?.environment) formData.append('environment', metadata.environment);
    if (metadata?.description) formData.append('description', metadata.description);
    
    const response = await fetch(`${API_BASE_URL}/datasets/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail);
    }
    
    return response.json();
  }, []);
  
  const deleteDataset = useCallback(async (datasetId: string) => {
    await apiFetch(`/datasets/${datasetId}`, { method: 'DELETE' });
    setState(prev => ({
      ...prev,
      data: prev.data?.filter(d => d.id !== datasetId) || null,
    }));
  }, []);
  
  return { ...state, fetchDatasets, uploadDataset, deleteDataset };
}

/**
 * Hook for parsing natural language goals
 */
export function useGoalParser() {
  const [state, setState] = useState<ApiResponse<{
    original: string;
    parsed: Record<string, unknown>;
  }>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const parseGoal = useCallback(async (goal: string) => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<{
        original: string;
        parsed: Record<string, unknown>;
      }>(`/environments/parse-goal?goal=${encodeURIComponent(goal)}`, {
        method: 'POST',
      });
      setState({ data: response, error: null, loading: false });
      return response;
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
      throw error;
    }
  }, []);
  
  return { ...state, parseGoal };
}

/**
 * Hook for fetching training metrics
 */
export function useMetrics(sessionId: string | null) {
  const [state, setState] = useState<ApiResponse<{
    metrics: Array<Record<string, unknown>>;
    total_steps: number;
    total_episodes: number;
  }>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const fetchMetrics = useCallback(async (limit = 100) => {
    if (!sessionId) return;
    
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<{
        metrics: Array<Record<string, unknown>>;
        total_steps: number;
        total_episodes: number;
      }>(`/training/metrics/${sessionId}?limit=${limit}`);
      setState({ data: response, error: null, loading: false });
      return response;
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
      throw error;
    }
  }, [sessionId]);
  
  return { ...state, fetchMetrics };
}

/**
 * Hook for managing agents
 */
export function useAgents() {
  const [state, setState] = useState<ApiResponse<Array<{
    agent_id: string;
    algorithm: string;
    training_steps: number;
  }>>>({
    data: null,
    error: null,
    loading: false,
  });
  
  const fetchAgents = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const response = await apiFetch<{ agents: Array<{
        agent_id: string;
        algorithm: string;
        training_steps: number;
      }> }>('/agents/list/active');
      setState({ data: response.agents, error: null, loading: false });
    } catch (error) {
      setState({ data: null, error: (error as Error).message, loading: false });
    }
  }, []);
  
  const createAgent = useCallback(async (config: AgentConfig) => {
    const response = await apiFetch<{
      agent_id: string;
      algorithm: string;
      status: string;
    }>('/agents/create', {
      method: 'POST',
      body: JSON.stringify(config),
    });
    return response;
  }, []);
  
  const deleteAgent = useCallback(async (agentId: string) => {
    await apiFetch(`/agents/${agentId}`, { method: 'DELETE' });
    setState(prev => ({
      ...prev,
      data: prev.data?.filter(a => a.agent_id !== agentId) || null,
    }));
  }, []);
  
  return { ...state, fetchAgents, createAgent, deleteAgent };
}

/**
 * Combined hook for all API operations
 */
export function useApi() {
  const environments = useEnvironments();
  const algorithms = useAlgorithms();
  const training = useTraining();
  const datasets = useDatasets();
  const goalParser = useGoalParser();
  const agents = useAgents();
  
  return {
    environments,
    algorithms,
    training,
    datasets,
    goalParser,
    agents,
  };
}

export default useApi;
