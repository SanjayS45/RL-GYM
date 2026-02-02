import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, FastForward, Rewind, Maximize2, RefreshCw } from 'lucide-react'
import { useStore } from '../../store/useStore'

interface AgentState {
  position: [number, number]
  velocity: [number, number]
  size: [number, number]
}

interface EnvironmentState {
  width: number
  height: number
  agent: AgentState
  obstacles: any[]
  goals: any[]
}

export default function VisualizationCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { training, environment } = useStore()
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [isPaused, setIsPaused] = useState(false)
  const [showTrails, setShowTrails] = useState(true)

  // Animation state
  const [agentPos, setAgentPos] = useState<[number, number]>([100, 300])
  const [trails, setTrails] = useState<[number, number][]>([])
  const animationRef = useRef<number>()

  // Simulate agent movement
  useEffect(() => {
    if (isPaused || training.status !== 'running') return

    const animate = () => {
      setAgentPos((prev) => {
        // Simple wandering behavior for demo
        const newX = prev[0] + (Math.random() - 0.4) * 5 * playbackSpeed
        const newY = prev[1] + (Math.random() - 0.5) * 5 * playbackSpeed

        // Clamp to bounds
        const clampedX = Math.max(20, Math.min(780, newX))
        const clampedY = Math.max(20, Math.min(580, newY))

        // Add to trails
        if (showTrails) {
          setTrails((t) => [...t.slice(-50), [clampedX, clampedY]])
        }

        return [clampedX, clampedY]
      })

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPaused, playbackSpeed, showTrails, training.status])

  // Draw on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.fillStyle = 'rgba(15, 15, 35, 1)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)'
    ctx.lineWidth = 1
    for (let x = 0; x <= canvas.width; x += 30) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, canvas.height)
      ctx.stroke()
    }
    for (let y = 0; y <= canvas.height; y += 30) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(canvas.width, y)
      ctx.stroke()
    }

    // Draw obstacles
    ctx.fillStyle = 'rgba(100, 100, 100, 0.5)'
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.8)'
    ctx.lineWidth = 2
    
    // Sample obstacles
    const obstacles = [
      { x: 280, y: 150, w: 100, h: 200 },
      { x: 450, y: 350, w: 150, h: 100 },
    ]
    obstacles.forEach((obs) => {
      ctx.fillRect(obs.x, obs.y, obs.w, obs.h)
      ctx.strokeRect(obs.x, obs.y, obs.w, obs.h)
    })

    // Draw goal
    ctx.fillStyle = 'rgba(16, 185, 129, 0.3)'
    ctx.strokeStyle = '#10b981'
    ctx.lineWidth = 2
    ctx.fillRect(680, 260, 60, 60)
    ctx.strokeRect(680, 260, 60, 60)
    
    // Goal pulse effect
    ctx.beginPath()
    ctx.arc(710, 290, 20 + Math.sin(Date.now() / 200) * 5, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)'
    ctx.stroke()

    // Draw trails
    if (showTrails && trails.length > 1) {
      ctx.beginPath()
      ctx.moveTo(trails[0][0], trails[0][1])
      trails.forEach((point, i) => {
        ctx.lineTo(point[0], point[1])
      })
      ctx.strokeStyle = 'rgba(0, 217, 255, 0.3)'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // Draw agent
    ctx.beginPath()
    ctx.arc(agentPos[0], agentPos[1], 15, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(0, 217, 255, 0.5)'
    ctx.fill()
    ctx.strokeStyle = '#00d9ff'
    ctx.lineWidth = 3
    ctx.stroke()

    // Agent glow
    const gradient = ctx.createRadialGradient(
      agentPos[0], agentPos[1], 0,
      agentPos[0], agentPos[1], 30
    )
    gradient.addColorStop(0, 'rgba(0, 217, 255, 0.3)')
    gradient.addColorStop(1, 'rgba(0, 217, 255, 0)')
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(agentPos[0], agentPos[1], 30, 0, Math.PI * 2)
    ctx.fill()

  }, [agentPos, trails, showTrails])

  const handleReset = () => {
    setAgentPos([100, 300])
    setTrails([])
  }

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPaused(!isPaused)}
            className="p-2 bg-surface-200 hover:bg-surface-100 rounded-lg transition-colors"
          >
            {isPaused ? (
              <Play className="w-5 h-5 text-accent-cyan" />
            ) : (
              <Pause className="w-5 h-5 text-accent-cyan" />
            )}
          </button>
          
          <button
            onClick={handleReset}
            className="p-2 bg-surface-200 hover:bg-surface-100 rounded-lg transition-colors"
          >
            <RefreshCw className="w-5 h-5 text-gray-400" />
          </button>

          <div className="h-8 w-px bg-gray-700 mx-2" />

          <div className="flex items-center gap-2 bg-surface-200 rounded-lg p-1">
            <button
              onClick={() => setPlaybackSpeed(Math.max(0.25, playbackSpeed - 0.25))}
              className="p-1 hover:bg-surface-100 rounded"
            >
              <Rewind className="w-4 h-4 text-gray-400" />
            </button>
            <span className="text-sm font-mono text-white w-12 text-center">
              {playbackSpeed}x
            </span>
            <button
              onClick={() => setPlaybackSpeed(Math.min(4, playbackSpeed + 0.25))}
              className="p-1 hover:bg-surface-100 rounded"
            >
              <FastForward className="w-4 h-4 text-gray-400" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={(e) => setShowTrails(e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-surface-300 text-accent-cyan focus:ring-accent-cyan/50"
            />
            <span className="text-sm text-gray-400">Show trails</span>
          </label>

          <button className="p-2 bg-surface-200 hover:bg-surface-100 rounded-lg transition-colors">
            <Maximize2 className="w-5 h-5 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 panel p-0 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={800}
          height={600}
          className="w-full h-full object-contain"
          style={{ imageRendering: 'pixelated' }}
        />
      </div>

      {/* Info bar */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-4">
          <span className="text-gray-400">
            Position:{' '}
            <span className="font-mono text-accent-cyan">
              ({agentPos[0].toFixed(0)}, {agentPos[1].toFixed(0)})
            </span>
          </span>
          <span className="text-gray-400">
            Step:{' '}
            <span className="font-mono text-accent-purple">
              {training.currentStep}
            </span>
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-accent-cyan" />
          <span className="text-gray-400">Agent</span>
          <div className="w-3 h-3 rounded-full bg-accent-green ml-4" />
          <span className="text-gray-400">Goal</span>
          <div className="w-3 h-3 rounded-full bg-gray-500 ml-4" />
          <span className="text-gray-400">Obstacle</span>
        </div>
      </div>
    </div>
  )
}

