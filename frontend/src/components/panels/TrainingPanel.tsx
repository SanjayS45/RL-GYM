import { useState, useEffect, useCallback } from 'react'
import { Play, Pause, Square, Settings, Terminal, Cpu, Clock, Zap } from 'lucide-react'
import { useStore } from '../../store/useStore'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

export default function TrainingPanel() {
  const { 
    training, 
    agent, 
    environment, 
    goalText, 
    setTrainingStatus, 
    setSessionId,
    updateTrainingProgress,
    updateTrainingMetrics,
    addToHistory,
    resetTraining
  } = useStore()
  
  const [totalTimesteps, setTotalTimesteps] = useState(100000)
  const [evalFrequency, setEvalFrequency] = useState(1000)
  const [logFrequency, setLogFrequency] = useState(100)
  const [ws, setWs] = useState<WebSocket | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  const [logs, setLogs] = useState<string[]>([])

  const isTraining = training.status === 'running'
  const isPaused = training.status === 'paused'
  const canStart = training.status === 'idle' || training.status === 'completed'

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    setConnectionStatus('connecting')
    const socket = new WebSocket(`${WS_URL}/training/ws`)
    
    socket.onopen = () => {
      setConnectionStatus('connected')
      addLog('[SYSTEM] Connected to training server')
    }
    
    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        if (data.type === 'training_update') {
          updateTrainingProgress(data.step || 0, data.episode || 0, data.reward || 0)
          if (data.loss !== undefined) {
            updateTrainingMetrics({ loss: data.loss })
          }
          if (data.metrics) {
            updateTrainingMetrics(data.metrics)
          }
          if (data.reward !== undefined && data.loss !== undefined) {
            addToHistory(data.reward, data.loss)
          }
        } else if (data.type === 'training_complete') {
          setTrainingStatus('completed')
          addLog('[SYSTEM] Training completed')
        } else if (data.type === 'training_error') {
          setTrainingStatus('error')
          addLog(`[ERROR] ${data.error}`)
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }
    
    socket.onclose = () => {
      setConnectionStatus('disconnected')
      addLog('[SYSTEM] Disconnected from server')
    }
    
    socket.onerror = () => {
      setConnectionStatus('disconnected')
      addLog('[ERROR] Connection failed')
    }
    
    setWs(socket)
    return socket
  }, [updateTrainingProgress, updateTrainingMetrics, addToHistory, setTrainingStatus])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (ws) {
        ws.close()
      }
    }
  }, [ws])

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs(prev => [...prev.slice(-100), `[${timestamp}] ${message}`])
  }

  const handleStart = async () => {
    try {
      // Connect WebSocket if not connected
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket()
      }
      
      // Reset previous training data
      resetTraining()
      setLogs([])
      
      addLog(`[CONFIG] Environment: ${environment.type} (${environment.config})`)
      addLog(`[CONFIG] Algorithm: ${agent.algorithm}`)
      addLog(`[CONFIG] Total timesteps: ${totalTimesteps.toLocaleString()}`)
      addLog('[SYSTEM] Starting training...')
      
      const config = {
        environment: environment.type,
        algorithm: agent.algorithm.toLowerCase(),
        env_config: {
          width: environment.width,
          height: environment.height,
          config: environment.config,
        },
        agent_config: {
          learning_rate: agent.learningRate,
          gamma: agent.gamma,
          batch_size: agent.batchSize,
          hidden_dims: agent.hiddenDims,
          ...(agent.algorithm === 'PPO' && {
            n_steps: agent.nSteps,
            n_epochs: agent.nEpochs,
            clip_range: agent.clipRange,
            ent_coef: agent.entCoef,
          }),
          ...(agent.algorithm === 'A2C' && {
            n_steps: agent.nSteps,
            ent_coef: agent.entCoef,
          }),
        },
        training_config: {
          total_timesteps: totalTimesteps,
          eval_freq: evalFrequency,
          log_freq: logFrequency,
        },
        natural_language_goal: goalText || undefined,
      }
      
      const response = await fetch(`${API_URL}/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      
      if (response.ok) {
        const data = await response.json()
        setSessionId(data.session_id)
        setTrainingStatus('running')
        addLog(`[SYSTEM] Training session started: ${data.session_id}`)
      } else {
        const error = await response.json()
        addLog(`[ERROR] Failed to start: ${error.detail || 'Unknown error'}`)
        // Run local simulation for demo
        runLocalSimulation()
      }
    } catch (error) {
      addLog('[WARN] Backend unavailable, running local simulation...')
      runLocalSimulation()
    }
  }

  // Local simulation for demo when backend is not available
  const runLocalSimulation = () => {
    setTrainingStatus('running')
    addLog('[SYSTEM] Running local simulation mode')
    
    let step = 0
    let episode = 0
    let totalReward = 0
    
    const interval = setInterval(() => {
      if (step >= totalTimesteps) {
        clearInterval(interval)
        setTrainingStatus('completed')
        addLog('[SYSTEM] Training simulation completed')
        return
      }
      
      // Simulate training progress
      step += Math.floor(Math.random() * 100) + 50
      const reward = Math.random() * 10 - 2 + (step / totalTimesteps) * 5 // Improving rewards over time
      const loss = 0.5 * Math.exp(-step / (totalTimesteps * 0.3)) + Math.random() * 0.05
      
      if (Math.random() < 0.1) {
        episode += 1
        totalReward = reward
      } else {
        totalReward += reward * 0.1
      }
      
      updateTrainingProgress(step, episode, totalReward)
      updateTrainingMetrics({
        meanReward: reward,
        maxReward: Math.max(reward, training.metrics.maxReward),
        loss: loss,
        episodeLength: Math.floor(Math.random() * 500) + 100,
      })
      
      if (episode > 0 && Math.random() < 0.3) {
        addToHistory(reward, loss)
      }
      
      if (step % 5000 < 100) {
        addLog(`[TRAIN] Step ${step.toLocaleString()} | Episode ${episode} | Reward: ${reward.toFixed(2)} | Loss: ${loss.toExponential(2)}`)
      }
    }, 50)
    
    // Store interval for cleanup
    return () => clearInterval(interval)
  }

  const handlePause = async () => {
    if (isPaused) {
      setTrainingStatus('running')
      addLog('[SYSTEM] Training resumed')
      if (training.sessionId) {
        try {
          await fetch(`${API_URL}/training/resume/${training.sessionId}`, { method: 'POST' })
        } catch {}
      }
    } else {
      setTrainingStatus('paused')
      addLog('[SYSTEM] Training paused')
      if (training.sessionId) {
        try {
          await fetch(`${API_URL}/training/pause/${training.sessionId}`, { method: 'POST' })
        } catch {}
      }
    }
  }

  const handleStop = async () => {
    setTrainingStatus('idle')
    addLog('[SYSTEM] Training stopped')
    if (training.sessionId) {
      try {
        await fetch(`${API_URL}/training/stop/${training.sessionId}`, { method: 'POST' })
      } catch {}
    }
  }

  const progress = totalTimesteps > 0 ? (training.currentStep / totalTimesteps) * 100 : 0

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Top row - Configuration and Controls */}
      <div className="flex gap-4">
        {/* Training Configuration */}
        <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Settings className="w-4 h-4 text-[#8b949e]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Training Configuration</span>
          </div>
          <div className="p-4 grid grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-[#8b949e] mb-1.5">Total Timesteps</label>
              <input
                type="number"
                value={totalTimesteps}
                onChange={(e) => setTotalTimesteps(parseInt(e.target.value) || 100000)}
                className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-sm text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
                disabled={isTraining}
              />
            </div>
            <div>
              <label className="block text-xs text-[#8b949e] mb-1.5">Eval Frequency</label>
              <input
                type="number"
                value={evalFrequency}
                onChange={(e) => setEvalFrequency(parseInt(e.target.value) || 1000)}
                className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-sm text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
                disabled={isTraining}
              />
            </div>
            <div>
              <label className="block text-xs text-[#8b949e] mb-1.5">Log Frequency</label>
              <input
                type="number"
                value={logFrequency}
                onChange={(e) => setLogFrequency(parseInt(e.target.value) || 100)}
                className="w-full bg-[#161b22] border border-[#30363d] rounded px-3 py-1.5 text-sm text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
                disabled={isTraining}
              />
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="w-80 bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-4 py-2 border-b border-[#30363d] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#8b949e]" />
              <span className="text-sm font-medium text-[#c9d1d9]">Control</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-[#3fb950]' :
                connectionStatus === 'connecting' ? 'bg-[#d29922] animate-pulse' :
                'bg-[#8b949e]'
              }`} />
              <span className="text-xs text-[#8b949e]">{connectionStatus}</span>
            </div>
          </div>
          <div className="p-4 space-y-4">
            {/* Status */}
            <div className="flex items-center justify-between">
              <span className="text-xs text-[#8b949e]">Status</span>
              <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                isTraining ? 'bg-[#238636]/20 text-[#3fb950]' :
                isPaused ? 'bg-[#9e6a03]/20 text-[#d29922]' :
                training.status === 'completed' ? 'bg-[#1f6feb]/20 text-[#58a6ff]' :
                'bg-[#30363d] text-[#8b949e]'
              }`}>
                {training.status.toUpperCase()}
              </span>
            </div>

            {/* Progress */}
            <div>
              <div className="flex justify-between text-xs text-[#8b949e] mb-1">
                <span>Progress</span>
                <span className="font-mono">{progress.toFixed(1)}%</span>
              </div>
              <div className="h-1.5 bg-[#21262d] rounded-full overflow-hidden">
                <div 
                  className="h-full bg-[#238636] transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Buttons */}
            <div className="flex gap-2">
              {canStart ? (
                <button
                  onClick={handleStart}
                  className="flex-1 bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium py-2 px-4 rounded flex items-center justify-center gap-2 transition-colors"
                >
                  <Play className="w-4 h-4" />
                  Start
                </button>
              ) : (
                <>
                  <button
                    onClick={handlePause}
                    className="flex-1 bg-[#30363d] hover:bg-[#484f58] text-[#c9d1d9] text-sm font-medium py-2 px-4 rounded flex items-center justify-center gap-2 transition-colors"
                  >
                    {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                    {isPaused ? 'Resume' : 'Pause'}
                  </button>
                  <button
                    onClick={handleStop}
                    className="bg-[#da3633] hover:bg-[#f85149] text-white text-sm font-medium py-2 px-3 rounded flex items-center justify-center transition-colors"
                  >
                    <Square className="w-4 h-4" />
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Middle row - Live Stats */}
      <div className="grid grid-cols-5 gap-3">
        <StatCard label="Steps" value={training.currentStep.toLocaleString()} icon={<Zap className="w-3.5 h-3.5" />} />
        <StatCard label="Episodes" value={training.currentEpisode.toString()} icon={<Clock className="w-3.5 h-3.5" />} />
        <StatCard 
          label="Mean Reward" 
          value={training.history.rewards.length > 0 ? training.metrics.meanReward.toFixed(3) : '--'} 
          trend={training.metrics.meanReward > 0 ? 'up' : undefined}
        />
        <StatCard 
          label="Loss" 
          value={training.history.losses.length > 0 ? training.metrics.loss.toExponential(2) : '--'} 
        />
        <StatCard 
          label="Ep. Length" 
          value={training.metrics.episodeLength > 0 ? training.metrics.episodeLength.toFixed(0) : '--'} 
        />
      </div>

      {/* Bottom row - Training Log */}
      <div className="flex-1 min-h-0 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
        <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#8b949e]" />
          <span className="text-sm font-medium text-[#c9d1d9]">Training Log</span>
          <span className="text-xs text-[#8b949e] ml-auto">{logs.length} entries</span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 font-mono text-xs">
          {logs.length === 0 ? (
            <div className="text-[#8b949e] text-center py-8">
              No training logs yet. Start a training run to see output.
            </div>
          ) : (
            <div className="space-y-0.5">
              {logs.map((log, i) => (
                <div 
                  key={i} 
                  className={`${
                    log.includes('[ERROR]') ? 'text-[#f85149]' :
                    log.includes('[WARN]') ? 'text-[#d29922]' :
                    log.includes('[SYSTEM]') ? 'text-[#58a6ff]' :
                    log.includes('[CONFIG]') ? 'text-[#a371f7]' :
                    'text-[#8b949e]'
                  }`}
                >
                  {log}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function StatCard({ 
  label, 
  value, 
  icon,
  trend 
}: { 
  label: string
  value: string
  icon?: React.ReactNode
  trend?: 'up' | 'down' 
}) {
  return (
    <div className="bg-[#0d1117] border border-[#30363d] rounded-md px-3 py-2">
      <div className="flex items-center gap-1.5 text-[#8b949e] mb-1">
        {icon}
        <span className="text-xs">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-lg font-mono font-semibold text-[#c9d1d9]">{value}</span>
        {trend && (
          <span className={trend === 'up' ? 'text-[#3fb950] text-xs' : 'text-[#f85149] text-xs'}>
            {trend === 'up' ? '↑' : '↓'}
          </span>
        )}
      </div>
    </div>
  )
}
