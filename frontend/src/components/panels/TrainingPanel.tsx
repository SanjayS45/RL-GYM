import { useState, useEffect, useCallback, useRef } from 'react'
import { Play, Pause, Square, Settings, Terminal, Cpu, Clock, Zap, Eye, Target } from 'lucide-react'
import { useStore } from '../../store/useStore'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

interface AgentState {
  x: number
  y: number
  vx: number
  vy: number
  trajectory: Array<{ x: number; y: number }>
}

interface SimulationState {
  agent: AgentState
  goal: { x: number; y: number }
  obstacles: Array<{ x: number; y: number; w: number; h: number }>
  episodeReward: number
  episodeStep: number
  reachedGoal: boolean
}

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
  
  // Simulation state for visualization
  const [simulation, setSimulation] = useState<SimulationState>({
    agent: { x: 80, y: 300, vx: 0, vy: 0, trajectory: [] },
    goal: { x: 700, y: 300 },
    obstacles: [],
    episodeReward: 0,
    episodeStep: 0,
    reachedGoal: false,
  })
  const [visualizationSpeed, setVisualizationSpeed] = useState(1)
  const simulationRef = useRef<number | null>(null)
  const stepRef = useRef(0)
  const episodeRef = useRef(0)

  const isTraining = training.status === 'running'
  const isPaused = training.status === 'paused'
  const canStart = training.status === 'idle' || training.status === 'completed'

  // Initialize obstacles based on environment config
  useEffect(() => {
    const obstacleConfigs: Record<string, Array<{ x: number; y: number; w: number; h: number }>> = {
      simple_obstacles: [
        { x: 280, y: 150, w: 80, h: 150 },
        { x: 450, y: 350, w: 120, h: 80 },
        { x: 350, y: 50, w: 60, h: 100 },
      ],
      maze_like: [
        { x: 150, y: 100, w: 200, h: 15 },
        { x: 450, y: 100, w: 200, h: 15 },
        { x: 200, y: 200, w: 150, h: 15 },
        { x: 450, y: 200, w: 150, h: 15 },
        { x: 150, y: 300, w: 200, h: 15 },
        { x: 450, y: 300, w: 200, h: 15 },
        { x: 200, y: 400, w: 150, h: 15 },
        { x: 450, y: 400, w: 150, h: 15 },
      ],
      cluttered: Array.from({ length: 15 }, (_, i) => ({
        x: 150 + (i % 5) * 100 + Math.random() * 50,
        y: 80 + Math.floor(i / 5) * 150 + Math.random() * 50,
        w: 30 + Math.random() * 40,
        h: 30 + Math.random() * 40,
      })),
      empty: [],
    }
    
    setSimulation(prev => ({
      ...prev,
      obstacles: obstacleConfigs[environment.config] || obstacleConfigs.simple_obstacles,
    }))
  }, [environment.config])

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
          // Update visualization if agent position is provided
          if (data.agent_position) {
            setSimulation(prev => ({
              ...prev,
              agent: {
                ...prev.agent,
                x: data.agent_position[0],
                y: data.agent_position[1],
                trajectory: [...prev.agent.trajectory.slice(-50), { x: data.agent_position[0], y: data.agent_position[1] }],
              },
            }))
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
      if (ws) ws.close()
      if (simulationRef.current) cancelAnimationFrame(simulationRef.current)
    }
  }, [ws])

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false })
    setLogs(prev => [...prev.slice(-100), `[${timestamp}] ${message}`])
  }

  // Run visual simulation
  const runVisualSimulation = useCallback(() => {
    stepRef.current = 0
    episodeRef.current = 0
    
    // Reset agent position
    setSimulation(prev => ({
      ...prev,
      agent: { x: 80, y: 300, vx: 0, vy: 0, trajectory: [] },
      episodeReward: 0,
      episodeStep: 0,
      reachedGoal: false,
    }))
    
    const simulate = () => {
      if (training.status !== 'running') {
        return
      }
      
      setSimulation(prev => {
        // Calculate direction to goal
        const dx = prev.goal.x - prev.agent.x
        const dy = prev.goal.y - prev.agent.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        
        // Simulated policy output (improves over time based on training progress)
        const trainingProgress = stepRef.current / totalTimesteps
        const policyQuality = 0.3 + trainingProgress * 0.6 // Policy improves from 0.3 to 0.9
        const noise = (1 - policyQuality) * (Math.random() - 0.5) * 2
        
        // Calculate desired velocity with noise
        let nvx = (dx / dist) * 3 * policyQuality + noise * 3
        let nvy = (dy / dist) * 3 * policyQuality + noise * 3
        
        // Add some smoothing
        const vx = prev.agent.vx * 0.7 + nvx * 0.3
        const vy = prev.agent.vy * 0.7 + nvy * 0.3
        
        let newX = prev.agent.x + vx * visualizationSpeed
        let newY = prev.agent.y + vy * visualizationSpeed
        
        // Clamp to bounds
        newX = Math.max(20, Math.min(780, newX))
        newY = Math.max(20, Math.min(580, newY))
        
        // Check collision with obstacles
        let collision = false
        for (const obs of prev.obstacles) {
          if (newX > obs.x - 15 && newX < obs.x + obs.w + 15 &&
              newY > obs.y - 15 && newY < obs.y + obs.h + 15) {
            collision = true
            break
          }
        }
        
        // Calculate reward
        let reward = -0.01 // Small step penalty
        const newDist = Math.sqrt((prev.goal.x - newX) ** 2 + (prev.goal.y - newY) ** 2)
        reward += (dist - newDist) * 0.1 // Reward for getting closer
        
        if (collision) {
          reward -= 0.5 // Collision penalty
          newX = prev.agent.x
          newY = prev.agent.y
        }
        
        // Check if reached goal
        const reachedGoal = newDist < 40
        if (reachedGoal) {
          reward += 10 // Goal bonus
        }
        
        const newEpisodeStep = prev.episodeStep + 1
        const newEpisodeReward = prev.episodeReward + reward
        
        // Update trajectory (keep last 100 points)
        const newTrajectory = [...prev.agent.trajectory.slice(-100), { x: newX, y: newY }]
        
        // Episode end conditions
        const episodeEnd = reachedGoal || newEpisodeStep > 500 || collision
        
        if (episodeEnd) {
          episodeRef.current += 1
          
          // Update training metrics
          updateTrainingProgress(stepRef.current, episodeRef.current, newEpisodeReward)
          updateTrainingMetrics({
            meanReward: newEpisodeReward,
            maxReward: Math.max(newEpisodeReward, training.metrics.maxReward),
            episodeLength: newEpisodeStep,
          })
          
          if (episodeRef.current % 5 === 0) {
            const loss = 0.5 * Math.exp(-stepRef.current / (totalTimesteps * 0.3)) + Math.random() * 0.05
            addToHistory(newEpisodeReward, loss)
            updateTrainingMetrics({ loss })
            addLog(`[TRAIN] Episode ${episodeRef.current} | Steps: ${newEpisodeStep} | Reward: ${newEpisodeReward.toFixed(2)} | ${reachedGoal ? '✓ GOAL' : collision ? '✗ Collision' : '⏱ Timeout'}`)
          }
          
          // Reset for new episode
          return {
            ...prev,
            agent: { 
              x: 80, 
              y: 250 + Math.random() * 100, 
              vx: 0, 
              vy: 0, 
              trajectory: [] 
            },
            episodeReward: 0,
            episodeStep: 0,
            reachedGoal: false,
          }
        }
        
        stepRef.current += 1
        
        return {
          ...prev,
          agent: { x: newX, y: newY, vx, vy, trajectory: newTrajectory },
          episodeReward: newEpisodeReward,
          episodeStep: newEpisodeStep,
          reachedGoal,
        }
      })
      
      // Check if training should end
      if (stepRef.current >= totalTimesteps) {
        setTrainingStatus('completed')
        addLog('[SYSTEM] Training simulation completed')
        return
      }
      
      simulationRef.current = requestAnimationFrame(simulate)
    }
    
    simulationRef.current = requestAnimationFrame(simulate)
  }, [totalTimesteps, visualizationSpeed, updateTrainingProgress, updateTrainingMetrics, addToHistory, setTrainingStatus, training.status, training.metrics.maxReward])

  const handleStart = async () => {
    try {
      // Connect WebSocket if not connected
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket()
      }
      
      // Reset previous training data
      resetTraining()
      setLogs([])
      stepRef.current = 0
      episodeRef.current = 0
      
      addLog(`[CONFIG] Environment: ${environment.type} (${environment.config})`)
      addLog(`[CONFIG] Algorithm: ${agent.algorithm}`)
      addLog(`[CONFIG] Goal: ${goalText || 'Navigate to target'}`)
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
        addLog('[WARN] Backend unavailable, running local simulation...')
        setTrainingStatus('running')
        runVisualSimulation()
      }
    } catch (error) {
      addLog('[WARN] Backend unavailable, running local simulation...')
      setTrainingStatus('running')
      runVisualSimulation()
    }
  }

  // Effect to start/stop simulation based on training status
  useEffect(() => {
    if (training.status === 'running' && !simulationRef.current) {
      runVisualSimulation()
    } else if (training.status !== 'running' && simulationRef.current) {
      cancelAnimationFrame(simulationRef.current)
      simulationRef.current = null
    }
  }, [training.status, runVisualSimulation])

  const handlePause = async () => {
    if (isPaused) {
      setTrainingStatus('running')
      addLog('[SYSTEM] Training resumed')
    } else {
      setTrainingStatus('paused')
      addLog('[SYSTEM] Training paused')
      if (simulationRef.current) {
        cancelAnimationFrame(simulationRef.current)
        simulationRef.current = null
      }
    }
  }

  const handleStop = async () => {
    setTrainingStatus('idle')
    addLog('[SYSTEM] Training stopped')
    if (simulationRef.current) {
      cancelAnimationFrame(simulationRef.current)
      simulationRef.current = null
    }
  }

  const progress = totalTimesteps > 0 ? (training.currentStep / totalTimesteps) * 100 : 0

  return (
    <div className="h-full flex gap-4">
      {/* Left column - Controls and Config */}
      <div className="w-80 flex flex-col gap-3">
        {/* Control Panel */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center justify-between">
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
              <span className="text-[10px] text-[#8b949e]">{connectionStatus}</span>
            </div>
          </div>
          <div className="p-3 space-y-3">
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
                  className="flex-1 bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium py-2 px-3 rounded flex items-center justify-center gap-2 transition-colors"
                >
                  <Play className="w-4 h-4" />
                  Start Training
                </button>
              ) : (
                <>
                  <button
                    onClick={handlePause}
                    className="flex-1 bg-[#30363d] hover:bg-[#484f58] text-[#c9d1d9] text-sm font-medium py-2 px-3 rounded flex items-center justify-center gap-2 transition-colors"
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

        {/* Training Config */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Settings className="w-4 h-4 text-[#8b949e]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Configuration</span>
          </div>
          <div className="p-3 space-y-3">
            <div>
              <label className="block text-[10px] text-[#8b949e] mb-1">Total Timesteps</label>
              <input
                type="number"
                value={totalTimesteps}
                onChange={(e) => setTotalTimesteps(parseInt(e.target.value) || 100000)}
                className="w-full bg-[#161b22] border border-[#30363d] rounded px-2 py-1.5 text-xs text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
                disabled={isTraining}
              />
            </div>
            <div>
              <label className="block text-[10px] text-[#8b949e] mb-1">Visualization Speed</label>
              <input
                type="range"
                min="0.5"
                max="3"
                step="0.5"
                value={visualizationSpeed}
                onChange={(e) => setVisualizationSpeed(parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] text-[#484f58]">
                <span>0.5x</span>
                <span className="text-[#58a6ff]">{visualizationSpeed}x</span>
                <span>3x</span>
              </div>
            </div>
          </div>
        </div>

        {/* Live Stats */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Zap className="w-4 h-4 text-[#8b949e]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Live Stats</span>
          </div>
          <div className="p-3 space-y-2">
            <StatRow label="Steps" value={training.currentStep.toLocaleString()} />
            <StatRow label="Episodes" value={training.currentEpisode.toString()} />
            <StatRow label="Episode Reward" value={simulation.episodeReward.toFixed(2)} highlight={simulation.episodeReward > 0} />
            <StatRow label="Mean Reward" value={training.metrics.meanReward > 0 ? training.metrics.meanReward.toFixed(2) : '--'} />
            <StatRow label="Best Reward" value={training.metrics.maxReward > 0 ? training.metrics.maxReward.toFixed(2) : '--'} />
          </div>
        </div>

        {/* Goal Display */}
        {goalText && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-md p-3">
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4 text-[#3fb950]" />
              <span className="text-xs font-medium text-[#c9d1d9]">Active Goal</span>
            </div>
            <p className="text-[10px] text-[#8b949e]">{goalText}</p>
          </div>
        )}
      </div>

      {/* Center - Visualization */}
      <div className="flex-1 flex flex-col gap-3">
        <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-[#8b949e]" />
              <span className="text-sm font-medium text-[#c9d1d9]">Live Training Visualization</span>
            </div>
            <div className="flex items-center gap-3 text-[10px] text-[#8b949e]">
              <span>Episode: <span className="text-[#c9d1d9] font-mono">{training.currentEpisode}</span></span>
              <span>Step: <span className="text-[#c9d1d9] font-mono">{simulation.episodeStep}</span></span>
            </div>
          </div>
          <div className="flex-1 bg-[#010409] relative overflow-hidden">
            <svg viewBox="0 0 800 600" className="w-full h-full">
              {/* Background grid */}
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#21262d" strokeWidth="0.5"/>
                </pattern>
              </defs>
              <rect width="800" height="600" fill="url(#grid)" />
              
              {/* Trajectory */}
              {simulation.agent.trajectory.length > 1 && (
                <polyline
                  points={simulation.agent.trajectory.map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="#58a6ff"
                  strokeWidth="1.5"
                  strokeOpacity="0.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              
              {/* Obstacles */}
              {simulation.obstacles.map((obs, i) => (
                <rect
                  key={i}
                  x={obs.x}
                  y={obs.y}
                  width={obs.w}
                  height={obs.h}
                  fill="#21262d"
                  stroke="#30363d"
                  strokeWidth="1"
                />
              ))}
              
              {/* Goal */}
              <g>
                <circle
                  cx={simulation.goal.x}
                  cy={simulation.goal.y}
                  r="35"
                  fill="none"
                  stroke="#238636"
                  strokeWidth="2"
                  strokeDasharray="4 2"
                  opacity="0.5"
                />
                <circle
                  cx={simulation.goal.x}
                  cy={simulation.goal.y}
                  r="20"
                  fill="#238636"
                  fillOpacity="0.3"
                />
                <circle
                  cx={simulation.goal.x}
                  cy={simulation.goal.y}
                  r="8"
                  fill="#3fb950"
                />
                <text
                  x={simulation.goal.x}
                  y={simulation.goal.y + 55}
                  textAnchor="middle"
                  fill="#3fb950"
                  fontSize="10"
                  fontFamily="monospace"
                >
                  GOAL
                </text>
              </g>
              
              {/* Agent */}
              <g>
                {/* Agent body */}
                <circle
                  cx={simulation.agent.x}
                  cy={simulation.agent.y}
                  r="12"
                  fill="#58a6ff"
                  fillOpacity="0.3"
                  stroke="#58a6ff"
                  strokeWidth="2"
                />
                <circle
                  cx={simulation.agent.x}
                  cy={simulation.agent.y}
                  r="5"
                  fill="#58a6ff"
                />
                {/* Velocity indicator */}
                {(simulation.agent.vx !== 0 || simulation.agent.vy !== 0) && (
                  <line
                    x1={simulation.agent.x}
                    y1={simulation.agent.y}
                    x2={simulation.agent.x + simulation.agent.vx * 5}
                    y2={simulation.agent.y + simulation.agent.vy * 5}
                    stroke="#58a6ff"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                )}
              </g>
              
              {/* Status overlay */}
              {!isTraining && !isPaused && (
                <g>
                  <rect x="0" y="0" width="800" height="600" fill="#010409" fillOpacity="0.7" />
                  <text x="400" y="290" textAnchor="middle" fill="#8b949e" fontSize="14" fontFamily="sans-serif">
                    Click "Start Training" to begin
                  </text>
                  <text x="400" y="315" textAnchor="middle" fill="#484f58" fontSize="11" fontFamily="sans-serif">
                    Agent will learn to reach the goal
                  </text>
                </g>
              )}
              
              {/* Training info overlay */}
              {isTraining && (
                <g>
                  <rect x="10" y="10" width="140" height="50" rx="4" fill="#0d1117" fillOpacity="0.9" stroke="#30363d" />
                  <text x="20" y="30" fill="#8b949e" fontSize="10" fontFamily="monospace">
                    Episode: <tspan fill="#c9d1d9">{training.currentEpisode}</tspan>
                  </text>
                  <text x="20" y="48" fill="#8b949e" fontSize="10" fontFamily="monospace">
                    Reward: <tspan fill={simulation.episodeReward > 0 ? '#3fb950' : '#f85149'}>{simulation.episodeReward.toFixed(2)}</tspan>
                  </text>
                </g>
              )}
            </svg>
          </div>
        </div>

        {/* Training Log */}
        <div className="h-40 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Terminal className="w-4 h-4 text-[#8b949e]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Training Log</span>
            <span className="text-[10px] text-[#8b949e] ml-auto">{logs.length} entries</span>
          </div>
          <div className="flex-1 overflow-y-auto p-2 font-mono text-[10px]">
            {logs.length === 0 ? (
              <div className="text-[#8b949e] text-center py-4">
                Logs will appear here during training
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
                      log.includes('✓ GOAL') ? 'text-[#3fb950]' :
                      log.includes('✗') ? 'text-[#f85149]' :
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
    </div>
  )
}

function StatRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[10px] text-[#8b949e]">{label}</span>
      <span className={`text-xs font-mono ${highlight ? 'text-[#3fb950]' : 'text-[#c9d1d9]'}`}>{value}</span>
    </div>
  )
}
