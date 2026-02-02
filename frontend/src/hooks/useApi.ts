import { useState, useCallback } from 'react'

const API_BASE = '/api'

interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
  body?: any
  headers?: Record<string, string>
}

export function useApi<T>() {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  })

  const request = useCallback(async (
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T | null> => {
    setState((prev) => ({ ...prev, loading: true, error: null }))

    try {
      const { method = 'GET', body, headers = {} } = options

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
        body: body ? JSON.stringify(body) : undefined,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Request failed: ${response.status}`)
      }

      const data = await response.json()
      setState({ data, loading: false, error: null })
      return data
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }))
      return null
    }
  }, [])

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null })
  }, [])

  return { ...state, request, reset }
}

// Specific API hooks
export function useTrainingApi() {
  const api = useApi<any>()

  const startTraining = useCallback(async (config: any) => {
    return api.request('/training/start', {
      method: 'POST',
      body: config,
    })
  }, [api])

  const getStatus = useCallback(async (sessionId: string) => {
    return api.request(`/training/${sessionId}`)
  }, [api])

  const pauseTraining = useCallback(async (sessionId: string) => {
    return api.request(`/training/${sessionId}/pause`, { method: 'POST' })
  }, [api])

  const resumeTraining = useCallback(async (sessionId: string) => {
    return api.request(`/training/${sessionId}/resume`, { method: 'POST' })
  }, [api])

  const stopTraining = useCallback(async (sessionId: string) => {
    return api.request(`/training/${sessionId}/stop`, { method: 'POST' })
  }, [api])

  return {
    ...api,
    startTraining,
    getStatus,
    pauseTraining,
    resumeTraining,
    stopTraining,
  }
}

export function useEnvironmentsApi() {
  const api = useApi<any>()

  const listEnvironments = useCallback(async () => {
    return api.request('/environments')
  }, [api])

  const getEnvironment = useCallback(async (envType: string) => {
    return api.request(`/environments/${envType}`)
  }, [api])

  const createEnvironment = useCallback(async (config: any) => {
    return api.request('/environments/create', {
      method: 'POST',
      body: config,
    })
  }, [api])

  return {
    ...api,
    listEnvironments,
    getEnvironment,
    createEnvironment,
  }
}

export function useAgentsApi() {
  const api = useApi<any>()

  const listAlgorithms = useCallback(async () => {
    return api.request('/agents/algorithms')
  }, [api])

  const getAlgorithm = useCallback(async (algorithm: string) => {
    return api.request(`/agents/algorithms/${algorithm}`)
  }, [api])

  const createAgent = useCallback(async (config: any) => {
    return api.request('/agents/create', {
      method: 'POST',
      body: config,
    })
  }, [api])

  return {
    ...api,
    listAlgorithms,
    getAlgorithm,
    createAgent,
  }
}

export function useDatasetsApi() {
  const api = useApi<any>()

  const listDatasets = useCallback(async () => {
    return api.request('/datasets')
  }, [api])

  const uploadDataset = useCallback(async (file: File, name?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (name) formData.append('name', name)

    const response = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error('Upload failed')
    }

    return response.json()
  }, [])

  const deleteDataset = useCallback(async (datasetId: string) => {
    return api.request(`/datasets/${datasetId}`, { method: 'DELETE' })
  }, [api])

  return {
    ...api,
    listDatasets,
    uploadDataset,
    deleteDataset,
  }
}

