import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, Square, RefreshCw, Settings, Zap } from 'lucide-react'
import { useStore } from '../../store/useStore'

export default function TrainingPanel() {
  const { training, agent, environment, goalText, setTrainingStatus } = useStore()
  const [totalTimesteps, setTotalTimesteps] = useState(100000)
  const [renderFrequency, setRenderFrequency] = useState(100)

  const isTraining = training.status === 'running'
  const isPaused = training.status === 'paused'
  const canStart = training.status === 'idle' || training.status === 'completed'

  const handleStart = async () => {
    setTrainingStatus('running')
    // API call would go here
    console.log('Starting training with:', {
      algorithm: agent.algorithm,
      environment: environment.type,
      config: environment.config,
      goal: goalText,
      totalTimesteps,
      hyperparameters: agent,
    })
  }

  const handlePause = () => {
    setTrainingStatus(isPaused ? 'running' : 'paused')
  }

  const handleStop = () => {
    setTrainingStatus('idle')
  }

  const progress = training.currentStep / totalTimesteps * 100

  return (
    <div className="h-full flex gap-6">
      {/* Training controls */}
      <div className="w-96 flex flex-col gap-4">
        {/* Status panel */}
        <div className="panel">
          <h2 className="panel-header">
            <Zap className="w-5 h-5" />
            Training Status
          </h2>

          <div className="flex items-center gap-3 mb-4">
            <div
              className={`w-3 h-3 rounded-full ${
                isTraining
                  ? 'bg-accent-green animate-pulse'
                  : isPaused
                  ? 'bg-accent-orange'
                  : 'bg-gray-500'
              }`}
            />
            <span className="font-medium capitalize">{training.status}</span>
          </div>

          {/* Progress bar */}
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <span>Progress</span>
              <span>{progress.toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-surface-300 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-neural-light to-accent-cyan"
                initial={{ width: 0 }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
          </div>

          {/* Quick stats */}
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Steps" value={training.currentStep.toLocaleString()} />
            <StatCard label="Episodes" value={training.currentEpisode.toString()} />
            <StatCard
              label="Mean Reward"
              value={training.metrics.meanReward.toFixed(2)}
              trend={training.metrics.meanReward > 0 ? 'up' : 'down'}
            />
            <StatCard label="Loss" value={training.metrics.loss.toFixed(4)} />
          </div>
        </div>

        {/* Controls */}
        <div className="panel">
          <h2 className="panel-header">
            <Settings className="w-5 h-5" />
            Training Settings
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-400 mb-2">Total Timesteps</label>
              <input
                type="number"
                value={totalTimesteps}
                onChange={(e) => setTotalTimesteps(parseInt(e.target.value))}
                className="input-field font-mono"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-2">Render Frequency</label>
              <input
                type="number"
                value={renderFrequency}
                onChange={(e) => setRenderFrequency(parseInt(e.target.value))}
                className="input-field font-mono"
                disabled={isTraining}
              />
              <p className="text-xs text-gray-500 mt-1">Update visualization every N steps</p>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3">
          {canStart ? (
            <motion.button
              onClick={handleStart}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex-1 btn-primary flex items-center justify-center gap-2 py-3"
            >
              <Play className="w-5 h-5" />
              Start Training
            </motion.button>
          ) : (
            <>
              <motion.button
                onClick={handlePause}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="flex-1 btn-secondary flex items-center justify-center gap-2 py-3"
              >
                {isPaused ? <Play className="w-5 h-5" /> : <Pause className="w-5 h-5" />}
                {isPaused ? 'Resume' : 'Pause'}
              </motion.button>
              <motion.button
                onClick={handleStop}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="btn-secondary flex items-center justify-center gap-2 py-3 px-4"
              >
                <Square className="w-5 h-5" />
              </motion.button>
            </>
          )}
        </div>
      </div>

      {/* Live visualization */}
      <div className="flex-1 panel">
        <h2 className="panel-header">
          <span className="text-accent-purple">🎬</span>
          Live Training View
        </h2>

        <div
          className="relative bg-surface-300/50 rounded-lg overflow-hidden"
          style={{ height: 'calc(100% - 60px)' }}
        >
          <div className="absolute inset-0 grid-bg" />

          {/* Agent visualization */}
          <svg viewBox="0 0 800 600" className="w-full h-full">
            {/* Background */}
            <rect x="0" y="0" width="800" height="600" fill="rgba(15, 15, 35, 0.5)" />

            {/* Goal */}
            <g>
              <rect
                x="680"
                y="260"
                width="60"
                height="60"
                rx="8"
                fill="rgba(16, 185, 129, 0.3)"
                stroke="#10b981"
                strokeWidth="2"
              />
              <motion.circle
                cx="710"
                cy="290"
                r="15"
                fill="none"
                stroke="#10b981"
                strokeWidth="2"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 1, 0.5],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                }}
              />
            </g>

            {/* Sample obstacles */}
            <rect x="280" y="150" width="100" height="200" rx="4" fill="rgba(100, 100, 100, 0.5)" />
            <rect x="450" y="350" width="150" height="100" rx="4" fill="rgba(100, 100, 100, 0.5)" />

            {/* Agent */}
            <motion.g
              animate={{
                x: isTraining ? [0, 100, 200, 150, 300] : 0,
                y: isTraining ? [0, -50, 20, -30, 0] : 0,
              }}
              transition={{
                duration: 5,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            >
              <circle
                cx="100"
                cy="300"
                r="18"
                fill="rgba(0, 217, 255, 0.5)"
                stroke="#00d9ff"
                strokeWidth="3"
              />
              {/* Agent direction indicator */}
              <line
                x1="100"
                y1="300"
                x2="120"
                y2="300"
                stroke="#00d9ff"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </motion.g>

            {/* Training info overlay */}
            {isTraining && (
              <g>
                <text x="20" y="30" fill="#00d9ff" fontSize="14" fontFamily="JetBrains Mono">
                  Episode: {training.currentEpisode}
                </text>
                <text x="20" y="50" fill="#a855f7" fontSize="14" fontFamily="JetBrains Mono">
                  Reward: {training.totalReward.toFixed(2)}
                </text>
              </g>
            )}
          </svg>

          {/* Overlay when not training */}
          {!isTraining && !isPaused && (
            <div className="absolute inset-0 flex items-center justify-center bg-surface-300/50 backdrop-blur-sm">
              <div className="text-center">
                <Play className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                <p className="text-gray-400">Click "Start Training" to begin</p>
              </div>
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
  trend,
}: {
  label: string
  value: string
  trend?: 'up' | 'down'
}) {
  return (
    <div className="metric-card">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="flex items-center gap-2">
        <span className="text-lg font-mono font-semibold text-white">{value}</span>
        {trend && (
          <span className={trend === 'up' ? 'text-accent-green' : 'text-red-400'}>
            {trend === 'up' ? '↑' : '↓'}
          </span>
        )}
      </div>
    </div>
  )
}

