import { useState } from 'react'
import { Layers, Grid, Navigation, Gamepad2, Settings, Target, Sparkles } from 'lucide-react'
import { useStore } from '../../store/useStore'

const environments = [
  { id: 'grid_world', name: 'Grid World', icon: Grid, description: 'Discrete grid-based navigation' },
  { id: 'navigation', name: 'Navigation', icon: Navigation, description: 'Continuous 2D point navigation' },
  { id: 'platformer', name: 'Platformer', icon: Gamepad2, description: 'Physics-based platforming' },
]

const configs: Record<string, { id: string; name: string; description: string }[]> = {
  grid_world: [
    { id: 'simple', name: 'Simple', description: 'Basic grid with sparse obstacles' },
    { id: 'maze', name: 'Maze', description: 'Maze-like structure requiring path planning' },
    { id: 'cliff', name: 'Cliff Walking', description: 'Grid with cliff edges (negative reward)' },
  ],
  navigation: [
    { id: 'empty', name: 'Empty', description: 'Open space with no obstacles' },
    { id: 'simple_obstacles', name: 'Simple Obstacles', description: 'Few scattered obstacles' },
    { id: 'maze_like', name: 'Maze-like', description: 'Corridor-style obstacle layout' },
    { id: 'cluttered', name: 'Cluttered', description: 'Dense random obstacle placement' },
  ],
  platformer: [
    { id: 'simple', name: 'Simple', description: 'Basic platforms to jump across' },
    { id: 'climbing', name: 'Climbing', description: 'Vertical ascent challenge' },
    { id: 'gaps', name: 'Gaps', description: 'Platforms with varying gap distances' },
    { id: 'moving_platforms', name: 'Moving Platforms', description: 'Dynamic moving platforms' },
  ],
}

const goalPresets = [
  { text: 'Reach the green target', category: 'navigation' },
  { text: 'Navigate to goal avoiding obstacles', category: 'navigation' },
  { text: 'Find shortest path to destination', category: 'optimization' },
  { text: 'Maximize reward collection', category: 'collection' },
  { text: 'Survive as long as possible', category: 'survival' },
  { text: 'Reach goal with minimum energy expenditure', category: 'efficiency' },
]

export default function EnvironmentPanel() {
  const { environment, setEnvironmentType, setEnvironmentConfig, goalText, setGoalText, updateEnvironmentState } = useStore()
  const [selectedEnv, setSelectedEnv] = useState(environment.type || 'navigation')

  const handleEnvSelect = (envId: string) => {
    setSelectedEnv(envId)
    setEnvironmentType(envId)
    const firstConfig = configs[envId]?.[0]?.id || 'simple'
    setEnvironmentConfig(firstConfig)
  }

  const handleConfigSelect = (configId: string) => {
    setEnvironmentConfig(configId)
  }

  const currentConfigs = configs[selectedEnv] || []

  return (
    <div className="h-full flex gap-4">
      {/* Left column - Environment selection */}
      <div className="w-72 flex flex-col gap-3">
        {/* Environment Type */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#8b949e]" />
            <span className="text-xs font-medium text-[#c9d1d9]">Environment Type</span>
          </div>
          <div className="p-2 space-y-1">
            {environments.map((env) => {
              const Icon = env.icon
              const isSelected = selectedEnv === env.id
              return (
                <button
                  key={env.id}
                  onClick={() => handleEnvSelect(env.id)}
                  className={`w-full p-2.5 rounded border text-left transition-all ${
                    isSelected
                      ? 'border-[#58a6ff] bg-[#58a6ff]/10'
                      : 'border-[#21262d] hover:border-[#30363d] bg-[#161b22]'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <div className={`p-1.5 rounded ${isSelected ? 'bg-[#58a6ff]/20 text-[#58a6ff]' : 'bg-[#21262d] text-[#8b949e]'}`}>
                      <Icon className="w-3.5 h-3.5" />
                    </div>
                    <div>
                      <div className={`font-medium text-xs ${isSelected ? 'text-[#c9d1d9]' : 'text-[#8b949e]'}`}>
                        {env.name}
                      </div>
                      <div className="text-[10px] text-[#484f58]">{env.description}</div>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Configuration */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Settings className="w-4 h-4 text-[#8b949e]" />
            <span className="text-xs font-medium text-[#c9d1d9]">Configuration</span>
          </div>
          <div className="p-2 space-y-1">
            {currentConfigs.map((config) => (
              <button
                key={config.id}
                onClick={() => handleConfigSelect(config.id)}
                className={`w-full p-2 rounded border text-left transition-all ${
                  environment.config === config.id
                    ? 'border-[#a371f7] bg-[#a371f7]/10'
                    : 'border-[#21262d] hover:border-[#30363d] bg-[#161b22]'
                }`}
              >
                <div className={`font-medium text-xs ${environment.config === config.id ? 'text-[#c9d1d9]' : 'text-[#8b949e]'}`}>
                  {config.name}
                </div>
                <div className="text-[10px] text-[#484f58]">{config.description}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Environment Parameters */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Settings className="w-4 h-4 text-[#8b949e]" />
            <span className="text-xs font-medium text-[#c9d1d9]">Parameters</span>
          </div>
          <div className="p-3 space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-[10px] text-[#8b949e] mb-1">Width</label>
                <input
                  type="number"
                  value={environment.width}
                  onChange={(e) => updateEnvironmentState({ width: parseInt(e.target.value) || 800 })}
                  className="w-full bg-[#161b22] border border-[#30363d] rounded px-2 py-1 text-xs text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-[10px] text-[#8b949e] mb-1">Height</label>
                <input
                  type="number"
                  value={environment.height}
                  onChange={(e) => updateEnvironmentState({ height: parseInt(e.target.value) || 600 })}
                  className="w-full bg-[#161b22] border border-[#30363d] rounded px-2 py-1 text-xs text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="block text-[10px] text-[#8b949e] mb-1">Random Seed</label>
              <input
                type="number"
                placeholder="Auto"
                className="w-full bg-[#161b22] border border-[#30363d] rounded px-2 py-1 text-xs text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Center - Environment preview */}
      <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
        <div className="px-3 py-2 border-b border-[#30363d] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[#a371f7]">◉</span>
            <span className="text-xs font-medium text-[#c9d1d9]">Environment Preview</span>
          </div>
          <div className="text-[10px] text-[#8b949e] font-mono">
            {selectedEnv} / {environment.config}
          </div>
        </div>
        <div className="flex-1 relative bg-[#010409] overflow-hidden">
          <EnvironmentPreview type={selectedEnv} config={environment.config} />
          {/* Info overlay */}
          <div className="absolute bottom-2 left-2 right-2 flex justify-between items-center">
            <div className="bg-[#0d1117]/90 backdrop-blur px-2 py-1 rounded text-[10px] font-mono border border-[#30363d]">
              <span className="text-[#8b949e]">Size: </span>
              <span className="text-[#c9d1d9]">{environment.width}×{environment.height}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right column - Goal definition */}
      <div className="w-80 flex flex-col gap-3">
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md flex-1 flex flex-col">
          <div className="px-3 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Target className="w-4 h-4 text-[#8b949e]" />
            <span className="text-xs font-medium text-[#c9d1d9]">Goal Definition</span>
          </div>
          <div className="p-3 flex-1 flex flex-col">
            <p className="text-[10px] text-[#8b949e] mb-2">
              Define the agent's objective. This will be converted to a reward function.
            </p>
            <textarea
              value={goalText}
              onChange={(e) => setGoalText(e.target.value)}
              placeholder="Describe what the agent should accomplish...&#10;&#10;Example: Navigate to the target while avoiding obstacles and minimizing path length."
              className="flex-1 bg-[#161b22] border border-[#30363d] rounded px-3 py-2 text-xs text-[#c9d1d9] resize-none focus:border-[#58a6ff] focus:outline-none placeholder:text-[#484f58]"
            />
            
            {/* Parsed reward components */}
            <div className="mt-3 p-2 bg-[#161b22] rounded border border-[#30363d]">
              <div className="text-[10px] text-[#8b949e] mb-1.5 flex items-center gap-1.5">
                <Sparkles className="w-3 h-3" />
                Parsed Reward Components
              </div>
              <div className="text-[10px] font-mono">
                {goalText ? (
                  <div className="space-y-0.5">
                    <div className="text-[#3fb950]">+ Distance to goal (primary)</div>
                    {goalText.toLowerCase().includes('avoid') && (
                      <div className="text-[#f85149]">- Collision penalty</div>
                    )}
                    {(goalText.toLowerCase().includes('short') || goalText.toLowerCase().includes('minim')) && (
                      <div className="text-[#d29922]">- Step penalty (efficiency)</div>
                    )}
                    {goalText.toLowerCase().includes('surviv') && (
                      <div className="text-[#58a6ff]">+ Survival bonus</div>
                    )}
                  </div>
                ) : (
                  <span className="text-[#484f58]">Enter a goal to see reward shaping</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Quick presets */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-3 py-2 border-b border-[#30363d]">
            <span className="text-xs font-medium text-[#c9d1d9]">Quick Presets</span>
          </div>
          <div className="p-2 space-y-1 max-h-48 overflow-y-auto">
            {goalPresets.map((preset, i) => (
              <button
                key={i}
                onClick={() => setGoalText(preset.text)}
                className="w-full text-left px-2 py-1.5 text-[10px] text-[#8b949e] hover:text-[#c9d1d9] hover:bg-[#161b22] rounded transition-colors flex items-center justify-between"
              >
                <span>{preset.text}</span>
                <span className="text-[#484f58] text-[9px]">{preset.category}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function EnvironmentPreview({ type, config }: { type: string; config: string }) {
  return (
    <svg viewBox="0 0 800 600" className="w-full h-full">
      {/* Background */}
      <rect x="0" y="0" width="800" height="600" fill="#010409" />
      
      {/* Grid lines (subtle) */}
      <g opacity="0.15">
        {Array.from({ length: 21 }).map((_, i) => (
          <line key={`v${i}`} x1={i * 40} y1="0" x2={i * 40} y2="600" stroke="#30363d" strokeWidth="0.5" />
        ))}
        {Array.from({ length: 16 }).map((_, i) => (
          <line key={`h${i}`} x1="0" y1={i * 40} x2="800" y2={i * 40} stroke="#30363d" strokeWidth="0.5" />
        ))}
      </g>

      {/* Goal */}
      <g>
        <rect x="700" y="260" width="60" height="60" rx="4" fill="none" stroke="#238636" strokeWidth="2" strokeDasharray="4 2" />
        <rect x="715" y="275" width="30" height="30" rx="2" fill="#238636" opacity="0.3" />
        <text x="730" y="340" textAnchor="middle" fill="#3fb950" fontSize="9" fontFamily="monospace">GOAL</text>
      </g>

      {/* Agent */}
      <g>
        <circle cx="80" cy="300" r="12" fill="none" stroke="#58a6ff" strokeWidth="2" />
        <circle cx="80" cy="300" r="4" fill="#58a6ff" />
        <line x1="80" y1="300" x2="96" y2="300" stroke="#58a6ff" strokeWidth="2" />
        <text x="80" y="332" textAnchor="middle" fill="#58a6ff" fontSize="9" fontFamily="monospace">AGENT</text>
      </g>

      {/* Obstacles based on environment type and config */}
      {type === 'navigation' && config === 'simple_obstacles' && (
        <g>
          <rect x="280" y="150" width="80" height="150" fill="#21262d" stroke="#30363d" strokeWidth="1" />
          <rect x="450" y="350" width="120" height="80" fill="#21262d" stroke="#30363d" strokeWidth="1" />
          <rect x="350" y="50" width="60" height="100" fill="#21262d" stroke="#30363d" strokeWidth="1" />
        </g>
      )}

      {type === 'navigation' && config === 'maze_like' && (
        <g>
          {/* Horizontal walls */}
          <rect x="150" y="100" width="200" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="450" y="100" width="200" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="200" y="200" width="150" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="450" y="200" width="150" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="150" y="300" width="200" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="450" y="300" width="200" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="200" y="400" width="150" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="450" y="400" width="150" height="15" fill="#21262d" stroke="#30363d" />
          <rect x="150" y="500" width="200" height="15" fill="#21262d" stroke="#30363d" />
          {/* Vertical connectors */}
          <rect x="350" y="115" width="15" height="85" fill="#21262d" stroke="#30363d" />
          <rect x="400" y="215" width="15" height="85" fill="#21262d" stroke="#30363d" />
          <rect x="350" y="315" width="15" height="85" fill="#21262d" stroke="#30363d" />
          <rect x="400" y="415" width="15" height="85" fill="#21262d" stroke="#30363d" />
        </g>
      )}

      {type === 'navigation' && config === 'cluttered' && (
        <g>
          {[
            { x: 180, y: 120, w: 40, h: 40 }, { x: 250, y: 80, w: 50, h: 30 },
            { x: 320, y: 150, w: 35, h: 55 }, { x: 400, y: 100, w: 45, h: 40 },
            { x: 480, y: 160, w: 40, h: 50 }, { x: 550, y: 80, w: 55, h: 35 },
            { x: 200, y: 220, w: 40, h: 45 }, { x: 280, y: 280, w: 50, h: 40 },
            { x: 360, y: 240, w: 35, h: 50 }, { x: 440, y: 300, w: 45, h: 35 },
            { x: 520, y: 250, w: 40, h: 45 }, { x: 180, y: 380, w: 50, h: 40 },
            { x: 260, y: 420, w: 40, h: 50 }, { x: 340, y: 370, w: 55, h: 35 },
            { x: 420, y: 440, w: 35, h: 45 }, { x: 500, y: 380, w: 45, h: 40 },
            { x: 220, y: 500, w: 40, h: 35 }, { x: 380, y: 520, w: 50, h: 40 },
            { x: 540, y: 480, w: 35, h: 50 },
          ].map((obs, i) => (
            <rect key={i} x={obs.x} y={obs.y} width={obs.w} height={obs.h} fill="#21262d" stroke="#30363d" strokeWidth="1" />
          ))}
        </g>
      )}

      {type === 'navigation' && config === 'empty' && (
        <text x="400" y="300" textAnchor="middle" fill="#30363d" fontSize="12" fontFamily="monospace">
          [Empty Environment]
        </text>
      )}

      {type === 'grid_world' && (
        <>
          {/* Grid lines */}
          <g>
            {Array.from({ length: 11 }).map((_, i) => (
              <line key={`gv${i}`} x1={80 + i * 64} y1="44" x2={80 + i * 64} y2="556" stroke="#21262d" strokeWidth="1" />
            ))}
            {Array.from({ length: 9 }).map((_, i) => (
              <line key={`gh${i}`} x1="80" y1={44 + i * 64} x2="720" y2={44 + i * 64} stroke="#21262d" strokeWidth="1" />
            ))}
          </g>
          {config === 'maze' && (
            <g>
              {[
                { x: 144, y: 108, w: 64, h: 192 }, { x: 272, y: 44, w: 64, h: 128 },
                { x: 272, y: 236, w: 64, h: 192 }, { x: 400, y: 108, w: 64, h: 256 },
                { x: 528, y: 172, w: 64, h: 256 }, { x: 208, y: 364, w: 128, h: 64 },
                { x: 400, y: 428, w: 192, h: 64 },
              ].map((wall, i) => (
                <rect key={i} x={wall.x} y={wall.y} width={wall.w} height={wall.h} fill="#21262d" />
              ))}
            </g>
          )}
          {config === 'cliff' && (
            <g>
              {Array.from({ length: 8 }).map((_, i) => (
                <rect key={i} x={144 + i * 64} y="492" width="64" height="64" fill="#7f1d1d" stroke="#991b1b" strokeWidth="1" />
              ))}
              <text x="400" y="530" textAnchor="middle" fill="#f87171" fontSize="10" fontFamily="monospace">
                CLIFF (negative reward)
              </text>
            </g>
          )}
        </>
      )}

      {type === 'platformer' && (
        <>
          {/* Ground */}
          <rect x="0" y="550" width="800" height="50" fill="#21262d" stroke="#30363d" />
          
          {config === 'simple' && (
            <g>
              <rect x="150" y="450" width="100" height="12" fill="#21262d" stroke="#30363d" />
              <rect x="350" y="350" width="100" height="12" fill="#21262d" stroke="#30363d" />
              <rect x="550" y="250" width="100" height="12" fill="#21262d" stroke="#30363d" />
            </g>
          )}
          
          {config === 'climbing' && (
            <g>
              <rect x="100" y="480" width="80" height="12" fill="#21262d" stroke="#30363d" />
              <rect x="220" y="400" width="80" height="12" fill="#21262d" stroke="#30363d" />
              <rect x="340" y="320" width="80" height="12" fill="#21262d" stroke="#30363d" />
              <rect x="460" y="240" width="80" height="12" fill="#21262d" stroke="#30363d" />
              <rect x="580" y="160" width="80" height="12" fill="#21262d" stroke="#30363d" />
            </g>
          )}
          
          {config === 'gaps' && (
            <g>
              <rect x="0" y="550" width="150" height="50" fill="#21262d" stroke="#30363d" />
              <rect x="250" y="550" width="100" height="50" fill="#21262d" stroke="#30363d" />
              <rect x="450" y="550" width="80" height="50" fill="#21262d" stroke="#30363d" />
              <rect x="630" y="550" width="170" height="50" fill="#21262d" stroke="#30363d" />
              <text x="200" y="575" textAnchor="middle" fill="#f85149" fontSize="8" fontFamily="monospace">GAP</text>
              <text x="400" y="575" textAnchor="middle" fill="#f85149" fontSize="8" fontFamily="monospace">GAP</text>
              <text x="565" y="575" textAnchor="middle" fill="#f85149" fontSize="8" fontFamily="monospace">GAP</text>
            </g>
          )}
          
          {config === 'moving_platforms' && (
            <g>
              <rect x="200" y="450" width="100" height="12" fill="#8957e5" stroke="#a371f7" strokeDasharray="4 2" />
              <rect x="450" y="350" width="100" height="12" fill="#8957e5" stroke="#a371f7" strokeDasharray="4 2" />
              <text x="250" y="442" textAnchor="middle" fill="#a371f7" fontSize="8" fontFamily="monospace">← →</text>
              <text x="500" y="342" textAnchor="middle" fill="#a371f7" fontSize="8" fontFamily="monospace">← →</text>
            </g>
          )}
        </>
      )}

      {/* Legend */}
      <g transform="translate(640, 15)">
        <rect x="0" y="0" width="140" height="55" fill="#0d1117" stroke="#30363d" rx="4" />
        <text x="8" y="14" fill="#8b949e" fontSize="8" fontFamily="monospace">LEGEND</text>
        <circle cx="16" cy="28" r="4" fill="#58a6ff" />
        <text x="28" y="31" fill="#8b949e" fontSize="8">Agent Start</text>
        <rect x="12" y="40" width="8" height="8" fill="#238636" opacity="0.5" />
        <text x="28" y="47" fill="#8b949e" fontSize="8">Goal Area</text>
      </g>
    </svg>
  )
}
