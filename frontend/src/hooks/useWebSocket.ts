import { useEffect, useRef, useCallback } from 'react'
import { useStore } from '../store/useStore'

interface WebSocketMessage {
  type: string
  data: any
}

export function useWebSocket(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  
  const {
    setWsConnected,
    updateTrainingProgress,
    updateTrainingMetrics,
    addToHistory,
    setTrainingStatus,
  } = useStore()

  const connect = useCallback(() => {
    if (!sessionId) return

    const wsUrl = `ws://localhost:8000/api/training/ws/${sessionId}`
    
    try {
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setWsConnected(true)
      }

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected')
        setWsConnected(false)
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connect()
        }, 3000)
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          handleMessage(message)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket:', error)
    }
  }, [sessionId, setWsConnected])

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'training_update':
        updateTrainingProgress(
          message.data.step,
          message.data.episode,
          message.data.reward
        )
        if (message.data.metrics) {
          updateTrainingMetrics(message.data.metrics)
        }
        break

      case 'episode_complete':
        addToHistory(message.data.total_reward, message.data.metrics?.loss || 0)
        updateTrainingMetrics({
          meanReward: message.data.metrics?.mean_reward || 0,
          maxReward: message.data.metrics?.max_reward || 0,
          episodeLength: message.data.length,
        })
        break

      case 'training_complete':
        setTrainingStatus('completed')
        updateTrainingMetrics(message.data.metrics)
        break

      case 'pong':
        // Heartbeat response
        break

      default:
        console.log('Unknown message type:', message.type)
    }
  }, [updateTrainingProgress, updateTrainingMetrics, addToHistory, setTrainingStatus])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setWsConnected(false)
  }, [setWsConnected])

  useEffect(() => {
    if (sessionId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [sessionId, connect, disconnect])

  // Heartbeat to keep connection alive
  useEffect(() => {
    const interval = setInterval(() => {
      send('ping')
    }, 30000)

    return () => clearInterval(interval)
  }, [send])

  return { send, disconnect }
}

