/**
 * WebSocket hook for real-time training updates
 * Connects to the backend WebSocket server for live training data
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { useStore } from '../store/useStore';

// WebSocket URL - configurable via environment variable
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

/**
 * Training update from WebSocket
 */
export interface TrainingUpdate {
  type: 'training_update' | 'training_error' | 'training_complete';
  session_id: string;
  step?: number;
  episode?: number;
  reward?: number;
  loss?: number;
  epsilon?: number;
  fps?: number;
  metrics?: Record<string, number>;
  visualization?: {
    agent_position: [number, number];
    goal_position: [number, number];
    obstacles?: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
    }>;
    trajectory?: Array<[number, number]>;
  };
  error?: string;
}

/**
 * WebSocket connection state
 */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

/**
 * WebSocket hook options
 */
interface UseWebSocketOptions {
  /** Auto-reconnect on disconnect */
  reconnect?: boolean;
  /** Reconnect interval in ms */
  reconnectInterval?: number;
  /** Max reconnect attempts */
  maxReconnectAttempts?: number;
  /** Session ID to subscribe to */
  sessionId?: string;
}

/**
 * WebSocket hook for training updates
 */
export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    reconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    sessionId,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [lastUpdate, setLastUpdate] = useState<TrainingUpdate | null>(null);
  const [updateHistory, setUpdateHistory] = useState<TrainingUpdate[]>([]);

  // Get store actions
  const {
    setIsTraining,
    addMetricsData,
    setVisualizationState,
  } = useStore();

  /**
   * Process incoming training update
   */
  const processUpdate = useCallback((update: TrainingUpdate) => {
    setLastUpdate(update);
    setUpdateHistory(prev => [...prev.slice(-99), update]);

    // Update global store based on update type
    switch (update.type) {
      case 'training_update':
        // Add metrics to store
        if (update.metrics || update.reward !== undefined) {
          addMetricsData({
            step: update.step || 0,
            episode: update.episode || 0,
            reward: update.reward || 0,
            loss: update.loss,
            epsilon: update.epsilon,
            fps: update.fps,
            timestamp: Date.now(),
            ...update.metrics,
          });
        }

        // Update visualization state
        if (update.visualization) {
          setVisualizationState({
            agentPosition: update.visualization.agent_position,
            goalPosition: update.visualization.goal_position,
            obstacles: update.visualization.obstacles,
            trajectory: update.visualization.trajectory,
          });
        }
        break;

      case 'training_complete':
        setIsTraining(false);
        break;

      case 'training_error':
        setIsTraining(false);
        console.error('Training error:', update.error);
        break;
    }
  }, [addMetricsData, setVisualizationState, setIsTraining]);

  /**
   * Connect to WebSocket server
   */
  const connect = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    setConnectionState('connecting');

    const ws = new WebSocket(`${WS_BASE_URL}/training/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionState('connected');
      reconnectAttemptsRef.current = 0;

      // Subscribe to session if provided
      if (sessionId) {
        ws.send(JSON.stringify({
          type: 'subscribe',
          session_id: sessionId,
        }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TrainingUpdate;
        
        // Only process updates for our session (if subscribed)
        if (!sessionId || data.session_id === sessionId) {
          processUpdate(data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionState('error');
    };

    ws.onclose = () => {
      setConnectionState('disconnected');
      wsRef.current = null;

      // Attempt reconnect if enabled
      if (reconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current++;
        reconnectTimeoutRef.current = setTimeout(() => {
          connect();
        }, reconnectInterval);
      }
    };
  }, [sessionId, reconnect, reconnectInterval, maxReconnectAttempts, processUpdate]);

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionState('disconnected');
  }, []);

  /**
   * Send a message through WebSocket
   */
  const send = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  /**
   * Subscribe to a training session
   */
  const subscribe = useCallback((newSessionId: string) => {
    send({
      type: 'subscribe',
      session_id: newSessionId,
    });
  }, [send]);

  /**
   * Send a ping to keep connection alive
   */
  const ping = useCallback(() => {
    send({ type: 'ping' });
  }, [send]);

  /**
   * Clear update history
   */
  const clearHistory = useCallback(() => {
    setUpdateHistory([]);
    setLastUpdate(null);
  }, []);

  // Connect on mount
  useEffect(() => {
    connect();

    // Set up ping interval to keep connection alive
    const pingInterval = setInterval(ping, 30000);

    return () => {
      clearInterval(pingInterval);
      disconnect();
    };
  }, [connect, disconnect, ping]);

  // Reconnect when session ID changes
  useEffect(() => {
    if (sessionId && wsRef.current?.readyState === WebSocket.OPEN) {
      subscribe(sessionId);
    }
  }, [sessionId, subscribe]);

  return {
    connectionState,
    isConnected: connectionState === 'connected',
    lastUpdate,
    updateHistory,
    connect,
    disconnect,
    send,
    subscribe,
    ping,
    clearHistory,
  };
}

/**
 * Hook for managing training visualization playback
 */
export function useVisualizationPlayback() {
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frames, setFrames] = useState<Array<{
    agentPosition: [number, number];
    goalPosition: [number, number];
    obstacles?: Array<{ x: number; y: number; width: number; height: number }>;
    reward?: number;
    step?: number;
  }>>([]);

  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);

  /**
   * Add a new frame to the playback buffer
   */
  const addFrame = useCallback((frame: typeof frames[0]) => {
    setFrames(prev => [...prev.slice(-500), frame]); // Keep last 500 frames
  }, []);

  /**
   * Start playback
   */
  const play = useCallback(() => {
    setIsPlaying(true);
  }, []);

  /**
   * Pause playback
   */
  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  /**
   * Toggle playback
   */
  const toggle = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  /**
   * Set playback speed (0.25x to 4x)
   */
  const setSpeed = useCallback((speed: number) => {
    setPlaybackSpeed(Math.max(0.25, Math.min(4, speed)));
  }, []);

  /**
   * Jump to a specific frame
   */
  const seekTo = useCallback((frame: number) => {
    setCurrentFrame(Math.max(0, Math.min(frames.length - 1, frame)));
  }, [frames.length]);

  /**
   * Clear all frames
   */
  const clear = useCallback(() => {
    setFrames([]);
    setCurrentFrame(0);
  }, []);

  // Handle playback
  useEffect(() => {
    if (isPlaying && frames.length > 0) {
      const interval = 100 / playbackSpeed; // Base interval of 100ms

      frameIntervalRef.current = setInterval(() => {
        setCurrentFrame(prev => {
          // Loop back to start if at end
          if (prev >= frames.length - 1) {
            return frames.length - 1; // Stay at last frame
          }
          return prev + 1;
        });
      }, interval);

      return () => {
        if (frameIntervalRef.current) {
          clearInterval(frameIntervalRef.current);
        }
      };
    }
  }, [isPlaying, frames.length, playbackSpeed]);

  return {
    isPlaying,
    playbackSpeed,
    currentFrame,
    totalFrames: frames.length,
    currentFrameData: frames[currentFrame] || null,
    frames,
    addFrame,
    play,
    pause,
    toggle,
    setSpeed,
    seekTo,
    clear,
  };
}

export default useWebSocket;
