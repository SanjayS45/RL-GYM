import { useState } from 'react'
import { motion } from 'framer-motion'
import { Layers, Plus, Trash2, Grid, Navigation, Gamepad2 } from 'lucide-react'
import { useStore } from '../../store/useStore'

const environments = [
  { id: 'grid_world', name: 'Grid World', icon: Grid, description: 'Discrete grid navigation' },
  { id: 'navigation', name: 'Navigation', icon: Navigation, description: 'Continuous 2D navigation' },
  { id: 'platformer', name: 'Platformer', icon: Gamepad2, description: 'Physics-based platforming' },
]

const configs: Record<string, string[]> = {
  grid_world: ['simple', 'maze', 'cliff'],
  navigation: ['empty', 'simple_obstacles', 'maze_like', 'cluttered'],
  platformer: ['simple', 'climbing', 'gaps', 'moving_platforms'],
}

export default function EnvironmentPanel() {
  const { environment, setEnvironmentType, setEnvironmentConfig, goalText, setGoalText } = useStore()
  const [selectedEnv, setSelectedEnv] = useState(environment.type)

  const handleEnvSelect = (envId: string) => {
    setSelectedEnv(envId)
    setEnvironmentType(envId)
    setEnvironmentConfig(configs[envId][0])
  }

  return (
    <div className="h-full flex gap-6">
      {/* Environment selection */}
      <div className="w-80 flex flex-col">
        <div className="panel flex-1">
          <h2 className="panel-header">
            <Layers className="w-5 h-5" />
            Environment Type
          </h2>

          <div className="space-y-2">
            {environments.map((env) => {
              const Icon = env.icon
              const isSelected = selectedEnv === env.id

              return (
                <motion.button
                  key={env.id}
                  onClick={() => handleEnvSelect(env.id)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full p-4 rounded-lg border transition-all text-left ${
                    isSelected
                      ? 'border-accent-cyan bg-surface-200 glow-cyan'
                      : 'border-gray-700 hover:border-gray-600 bg-surface-300/30'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`p-2 rounded-lg ${
                        isSelected ? 'bg-accent-cyan/20' : 'bg-surface-200'
                      }`}
                    >
                      <Icon
                        className={`w-5 h-5 ${
                          isSelected ? 'text-accent-cyan' : 'text-gray-400'
                        }`}
                      />
                    </div>
                    <div>
                      <div className="font-medium text-white">{env.name}</div>
                      <div className="text-xs text-gray-400">{env.description}</div>
                    </div>
                  </div>
                </motion.button>
              )
            })}
          </div>

          {/* Configuration preset */}
          <div className="mt-6">
            <label className="block text-sm text-gray-400 mb-2">Configuration</label>
            <select
              value={environment.config}
              onChange={(e) => setEnvironmentConfig(e.target.value)}
              className="input-field"
            >
              {configs[selectedEnv]?.map((config) => (
                <option key={config} value={config}>
                  {config.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Goal definition */}
        <div className="panel mt-4">
          <h2 className="panel-header">
            <span className="text-accent-green">🎯</span>
            Goal Definition
          </h2>
          <textarea
            value={goalText}
            onChange={(e) => setGoalText(e.target.value)}
            placeholder="Describe your goal in natural language...&#10;e.g., 'Navigate to the top-right corner while avoiding obstacles'"
            className="input-field h-24 resize-none"
          />
          <p className="text-xs text-gray-500 mt-2">
            Goals are converted to reward functions automatically
          </p>
        </div>
      </div>

      {/* Environment preview */}
      <div className="flex-1 panel">
        <h2 className="panel-header">
          <span className="text-accent-purple">👁</span>
          Environment Preview
        </h2>

        <div className="relative bg-surface-300/50 rounded-lg overflow-hidden" style={{ height: 'calc(100% - 60px)' }}>
          {/* Grid background */}
          <div className="absolute inset-0 grid-bg" />

          {/* Canvas area */}
          <div className="absolute inset-4 border-2 border-dashed border-gray-600 rounded-lg flex items-center justify-center">
            <EnvironmentPreview type={selectedEnv} config={environment.config} />
          </div>

          {/* Info overlay */}
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center">
            <div className="bg-surface-200/80 backdrop-blur px-3 py-1.5 rounded-lg text-sm">
              <span className="text-gray-400">Size: </span>
              <span className="text-white font-mono">{environment.width}×{environment.height}</span>
            </div>
            <div className="bg-surface-200/80 backdrop-blur px-3 py-1.5 rounded-lg text-sm">
              <span className="text-gray-400">Obstacles: </span>
              <span className="text-white font-mono">{environment.obstacles.length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function EnvironmentPreview({ type, config }: { type: string; config: string }) {
  // Simple SVG-based preview
  return (
    <svg viewBox="0 0 800 600" className="w-full h-full max-w-xl">
      {/* Background */}
      <rect x="0" y="0" width="800" height="600" fill="rgba(15, 15, 35, 0.8)" />

      {/* Goal */}
      <rect x="700" y="260" width="60" height="60" rx="8" fill="rgba(16, 185, 129, 0.5)" stroke="#10b981" strokeWidth="2" />
      <text x="730" y="295" textAnchor="middle" fill="#10b981" fontSize="12">Goal</text>

      {/* Agent */}
      <circle cx="100" cy="300" r="20" fill="rgba(0, 217, 255, 0.5)" stroke="#00d9ff" strokeWidth="2" />
      <text x="100" y="305" textAnchor="middle" fill="#00d9ff" fontSize="10">Agent</text>

      {/* Sample obstacles based on config */}
      {type === 'navigation' && config === 'simple_obstacles' && (
        <>
          <rect x="280" y="150" width="100" height="200" rx="4" fill="rgba(100, 100, 100, 0.5)" stroke="#666" strokeWidth="1" />
          <rect x="450" y="350" width="150" height="100" rx="4" fill="rgba(100, 100, 100, 0.5)" stroke="#666" strokeWidth="1" />
        </>
      )}

      {type === 'grid_world' && (
        <>
          {/* Grid lines */}
          {Array.from({ length: 9 }).map((_, i) => (
            <line key={`v${i}`} x1={(i + 1) * 80} y1="0" x2={(i + 1) * 80} y2="600" stroke="rgba(99, 102, 241, 0.2)" strokeWidth="1" />
          ))}
          {Array.from({ length: 7 }).map((_, i) => (
            <line key={`h${i}`} x1="0" y1={(i + 1) * 75} x2="800" y2={(i + 1) * 75} stroke="rgba(99, 102, 241, 0.2)" strokeWidth="1" />
          ))}
        </>
      )}

      {type === 'platformer' && (
        <>
          {/* Ground */}
          <rect x="0" y="550" width="800" height="50" fill="rgba(80, 80, 80, 0.5)" stroke="#555" />
          {/* Platforms */}
          <rect x="150" y="450" width="120" height="20" rx="4" fill="rgba(80, 80, 80, 0.5)" stroke="#555" />
          <rect x="350" y="350" width="120" height="20" rx="4" fill="rgba(80, 80, 80, 0.5)" stroke="#555" />
          <rect x="550" y="250" width="120" height="20" rx="4" fill="rgba(80, 80, 80, 0.5)" stroke="#555" />
        </>
      )}
    </svg>
  )
}

