import { useState } from 'react'
import { Brain, Settings2, Info, ChevronDown, ChevronRight, RotateCcw } from 'lucide-react'
import { useStore } from '../../store/useStore'

const algorithms = [
  {
    id: 'PPO',
    name: 'PPO',
    fullName: 'Proximal Policy Optimization',
    type: 'on_policy',
    description: 'Stable policy gradient with clipped objective. Good for continuous and discrete actions.',
    params: ['learningRate', 'gamma', 'batchSize', 'nSteps', 'nEpochs', 'clipRange', 'entCoef', 'vfCoef', 'maxGradNorm'],
  },
  {
    id: 'DQN',
    name: 'DQN',
    fullName: 'Deep Q-Network',
    type: 'off_policy',
    description: 'Value-based with experience replay. Best for discrete action spaces.',
    params: ['learningRate', 'gamma', 'batchSize', 'bufferSize', 'epsilonStart', 'epsilonEnd', 'epsilonDecay', 'targetUpdateFreq'],
  },
  {
    id: 'SAC',
    name: 'SAC',
    fullName: 'Soft Actor-Critic',
    type: 'off_policy',
    description: 'Maximum entropy for continuous control. Highly sample efficient.',
    params: ['learningRate', 'gamma', 'batchSize', 'bufferSize', 'tau', 'alpha', 'autoAlpha'],
  },
  {
    id: 'A2C',
    name: 'A2C',
    fullName: 'Advantage Actor-Critic',
    type: 'on_policy',
    description: 'Synchronous actor-critic. Simple and fast to train.',
    params: ['learningRate', 'gamma', 'nSteps', 'entCoef', 'vfCoef', 'maxGradNorm'],
  },
]

// Parameter configurations with descriptions
const parameterConfig: Record<string, {
  label: string
  description: string
  type: 'number' | 'boolean'
  min?: number
  max?: number
  step?: number
  format?: (v: number) => string
  default: number | boolean
}> = {
  learningRate: {
    label: 'Learning Rate',
    description: 'Step size for gradient updates',
    type: 'number',
    min: 0.000001,
    max: 0.1,
    step: 0.0001,
    format: (v) => v.toExponential(1),
    default: 0.0003,
  },
  gamma: {
    label: 'Discount Factor (γ)',
    description: 'Importance of future rewards (0-1)',
    type: 'number',
    min: 0.9,
    max: 0.9999,
    step: 0.001,
    format: (v) => v.toFixed(4),
    default: 0.99,
  },
  batchSize: {
    label: 'Batch Size',
    description: 'Samples per gradient update',
    type: 'number',
    min: 8,
    max: 1024,
    step: 8,
    format: (v) => v.toString(),
    default: 64,
  },
  nSteps: {
    label: 'Rollout Steps (n_steps)',
    description: 'Steps collected per environment before update',
    type: 'number',
    min: 1,
    max: 8192,
    step: 1,
    format: (v) => v.toString(),
    default: 2048,
  },
  nEpochs: {
    label: 'Training Epochs',
    description: 'Number of passes through rollout data (PPO)',
    type: 'number',
    min: 1,
    max: 30,
    step: 1,
    format: (v) => v.toString(),
    default: 10,
  },
  clipRange: {
    label: 'Clip Range (ε)',
    description: 'PPO clipping parameter for policy updates',
    type: 'number',
    min: 0.05,
    max: 0.5,
    step: 0.01,
    format: (v) => v.toFixed(2),
    default: 0.2,
  },
  entCoef: {
    label: 'Entropy Coefficient',
    description: 'Bonus for exploration (higher = more exploration)',
    type: 'number',
    min: 0,
    max: 0.2,
    step: 0.001,
    format: (v) => v.toFixed(4),
    default: 0.01,
  },
  vfCoef: {
    label: 'Value Function Coefficient',
    description: 'Weight of value loss in total loss',
    type: 'number',
    min: 0.1,
    max: 1,
    step: 0.1,
    format: (v) => v.toFixed(2),
    default: 0.5,
  },
  maxGradNorm: {
    label: 'Max Gradient Norm',
    description: 'Gradient clipping threshold',
    type: 'number',
    min: 0.1,
    max: 10,
    step: 0.1,
    format: (v) => v.toFixed(1),
    default: 0.5,
  },
  bufferSize: {
    label: 'Buffer Size',
    description: 'Size of experience replay buffer',
    type: 'number',
    min: 1000,
    max: 10000000,
    step: 1000,
    format: (v) => v >= 1000000 ? `${(v/1000000).toFixed(1)}M` : v >= 1000 ? `${(v/1000).toFixed(0)}K` : v.toString(),
    default: 100000,
  },
  epsilonStart: {
    label: 'ε Start',
    description: 'Initial exploration rate',
    type: 'number',
    min: 0.1,
    max: 1,
    step: 0.1,
    format: (v) => v.toFixed(2),
    default: 1.0,
  },
  epsilonEnd: {
    label: 'ε End',
    description: 'Final exploration rate',
    type: 'number',
    min: 0.01,
    max: 0.5,
    step: 0.01,
    format: (v) => v.toFixed(2),
    default: 0.05,
  },
  epsilonDecay: {
    label: 'ε Decay',
    description: 'Exploration decay factor per step',
    type: 'number',
    min: 0.99,
    max: 0.9999,
    step: 0.0001,
    format: (v) => v.toFixed(4),
    default: 0.995,
  },
  targetUpdateFreq: {
    label: 'Target Update Freq',
    description: 'Steps between target network updates',
    type: 'number',
    min: 1,
    max: 10000,
    step: 1,
    format: (v) => v.toString(),
    default: 100,
  },
  tau: {
    label: 'Soft Update (τ)',
    description: 'Target network soft update coefficient',
    type: 'number',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    format: (v) => v.toFixed(3),
    default: 0.005,
  },
  alpha: {
    label: 'Entropy α',
    description: 'Temperature parameter for entropy regularization',
    type: 'number',
    min: 0.01,
    max: 1,
    step: 0.01,
    format: (v) => v.toFixed(2),
    default: 0.2,
  },
  autoAlpha: {
    label: 'Auto-tune α',
    description: 'Automatically adjust entropy coefficient',
    type: 'boolean',
    default: true,
  },
}

export default function AgentPanel() {
  const { agent, setAlgorithm, setHyperparameter, resetAgentToDefaults } = useStore()
  const [expandedSections, setExpandedSections] = useState<string[]>(['core', 'algorithm'])
  const [customParams, setCustomParams] = useState<Record<string, number | boolean>>({})

  const selectedAlgo = algorithms.find((a) => a.id === agent.algorithm)

  const toggleSection = (section: string) => {
    setExpandedSections(prev => 
      prev.includes(section) 
        ? prev.filter(s => s !== section)
        : [...prev, section]
    )
  }

  const handleParamChange = (key: string, value: number | boolean) => {
    setCustomParams(prev => ({ ...prev, [key]: value }))
    // Map to store keys where they exist
    const storeKeyMap: Record<string, string> = {
      learningRate: 'learningRate',
      gamma: 'gamma',
      batchSize: 'batchSize',
      nSteps: 'nSteps',
      nEpochs: 'nEpochs',
      clipRange: 'clipRange',
      entCoef: 'entCoef',
    }
    if (storeKeyMap[key]) {
      setHyperparameter(storeKeyMap[key] as any, value)
    }
  }

  const getParamValue = (key: string): number | boolean => {
    if (customParams[key] !== undefined) return customParams[key]
    const storeKeyMap: Record<string, keyof typeof agent> = {
      learningRate: 'learningRate',
      gamma: 'gamma',
      batchSize: 'batchSize',
      nSteps: 'nSteps',
      nEpochs: 'nEpochs',
      clipRange: 'clipRange',
      entCoef: 'entCoef',
    }
    if (storeKeyMap[key] && agent[storeKeyMap[key]] !== undefined) {
      return agent[storeKeyMap[key]] as number
    }
    return parameterConfig[key]?.default ?? 0
  }

  const handleReset = () => {
    setCustomParams({})
    resetAgentToDefaults(agent.algorithm)
  }

  return (
    <div className="h-full flex gap-4">
      {/* Algorithm selection */}
      <div className="w-80 flex flex-col gap-4">
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md">
          <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Brain className="w-4 h-4 text-[#8b949e]" />
            <span className="text-sm font-medium text-[#c9d1d9]">RL Algorithm</span>
          </div>
          <div className="p-3 space-y-2">
            {algorithms.map((algo) => (
              <button
                key={algo.id}
                onClick={() => {
                  setAlgorithm(algo.id)
                  setCustomParams({})
                }}
                className={`w-full p-3 rounded border text-left transition-all ${
                  agent.algorithm === algo.id
                    ? 'border-[#58a6ff] bg-[#58a6ff]/10'
                    : 'border-[#30363d] hover:border-[#484f58] bg-[#161b22]'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono font-bold text-[#c9d1d9]">{algo.name}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                    algo.type === 'on_policy'
                      ? 'bg-[#238636]/20 text-[#3fb950]'
                      : 'bg-[#9e6a03]/20 text-[#d29922]'
                  }`}>
                    {algo.type.replace('_', '-')}
                  </span>
                </div>
                <p className="text-[10px] text-[#8b949e] leading-relaxed">{algo.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Algorithm info */}
        {selectedAlgo && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-md p-4">
            <div className="flex items-start gap-3">
              <Info className="w-4 h-4 text-[#58a6ff] flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="text-sm font-medium text-[#c9d1d9]">{selectedAlgo.fullName}</h3>
                <p className="text-xs text-[#8b949e] mt-1 leading-relaxed">
                  {selectedAlgo.type === 'on_policy'
                    ? 'On-policy: Learns from current policy rollouts. More stable but less sample efficient. Requires new samples after each update.'
                    : 'Off-policy: Learns from stored experiences. More sample efficient. Can reuse old data for training.'}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Hyperparameters */}
      <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
        <div className="px-4 py-2 border-b border-[#30363d] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-[#8b949e]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Hyperparameters</span>
            <span className="text-xs text-[#8b949e]">({selectedAlgo?.name})</span>
          </div>
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs text-[#8b949e] hover:text-[#58a6ff] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset defaults
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Core Parameters */}
          <CollapsibleSection
            title="Core Parameters"
            expanded={expandedSections.includes('core')}
            onToggle={() => toggleSection('core')}
          >
            <div className="grid grid-cols-2 gap-3">
              {['learningRate', 'gamma', 'batchSize'].map((key) => {
                const config = parameterConfig[key]
                if (!config || config.type === 'boolean') return null
                return (
                  <ParameterInput
                    key={key}
                    config={config}
                    value={getParamValue(key) as number}
                    onChange={(v) => handleParamChange(key, v)}
                  />
                )
              })}
            </div>
          </CollapsibleSection>

          {/* Algorithm-specific Parameters */}
          <CollapsibleSection
            title={`${selectedAlgo?.name} Parameters`}
            expanded={expandedSections.includes('algorithm')}
            onToggle={() => toggleSection('algorithm')}
          >
            <div className="grid grid-cols-2 gap-3">
              {selectedAlgo?.params
                .filter(key => !['learningRate', 'gamma', 'batchSize'].includes(key))
                .map((key) => {
                  const config = parameterConfig[key]
                  if (!config) return null
                  
                  if (config.type === 'boolean') {
                    return (
                      <BooleanInput
                        key={key}
                        config={config}
                        value={getParamValue(key) as boolean}
                        onChange={(v) => handleParamChange(key, v)}
                      />
                    )
                  }
                  
                  return (
                    <ParameterInput
                      key={key}
                      config={config}
                      value={getParamValue(key) as number}
                      onChange={(v) => handleParamChange(key, v)}
                    />
                  )
                })}
            </div>
          </CollapsibleSection>

          {/* Network Architecture */}
          <CollapsibleSection
            title="Network Architecture"
            expanded={expandedSections.includes('network')}
            onToggle={() => toggleSection('network')}
          >
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-[#8b949e] mb-1.5">Hidden Layer Sizes</label>
                <div className="flex items-center gap-2">
                  {agent.hiddenDims.map((dim, i) => (
                    <input
                      key={i}
                      type="number"
                      value={dim}
                      onChange={(e) => {
                        const newDims = [...agent.hiddenDims]
                        newDims[i] = parseInt(e.target.value) || 64
                        setHyperparameter('hiddenDims', newDims)
                      }}
                      className="w-20 bg-[#161b22] border border-[#30363d] rounded px-2 py-1.5 text-sm text-[#c9d1d9] font-mono focus:border-[#58a6ff] focus:outline-none text-center"
                    />
                  ))}
                  <button
                    onClick={() => setHyperparameter('hiddenDims', [...agent.hiddenDims, 256])}
                    className="px-2 py-1.5 text-xs text-[#8b949e] hover:text-[#c9d1d9] border border-[#30363d] rounded hover:border-[#484f58]"
                  >
                    + Add
                  </button>
                  {agent.hiddenDims.length > 1 && (
                    <button
                      onClick={() => setHyperparameter('hiddenDims', agent.hiddenDims.slice(0, -1))}
                      className="px-2 py-1.5 text-xs text-[#f85149] hover:text-[#ff7b72] border border-[#30363d] rounded hover:border-[#f85149]"
                    >
                      Remove
                    </button>
                  )}
                </div>
                <p className="text-[10px] text-[#8b949e] mt-1">
                  Architecture: Input → {agent.hiddenDims.join(' → ')} → Output
                </p>
              </div>
            </div>
          </CollapsibleSection>
        </div>
      </div>
    </div>
  )
}

function CollapsibleSection({ 
  title, 
  expanded, 
  onToggle, 
  children 
}: { 
  title: string
  expanded: boolean
  onToggle: () => void
  children: React.ReactNode 
}) {
  return (
    <div className="border border-[#30363d] rounded">
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 flex items-center justify-between hover:bg-[#161b22] transition-colors"
      >
        <span className="text-sm font-medium text-[#c9d1d9]">{title}</span>
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-[#8b949e]" />
        ) : (
          <ChevronRight className="w-4 h-4 text-[#8b949e]" />
        )}
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-1">
          {children}
        </div>
      )}
    </div>
  )
}

function ParameterInput({
  config,
  value,
  onChange,
}: {
  config: typeof parameterConfig[string]
  value: number
  onChange: (value: number) => void
}) {
  if (config.type !== 'number') return null
  
  return (
    <div className="bg-[#161b22] rounded p-3 border border-[#21262d]">
      <div className="flex items-center justify-between mb-2">
        <label className="text-xs font-medium text-[#c9d1d9]">{config.label}</label>
        <input
          type="text"
          value={config.format ? config.format(value) : value}
          onChange={(e) => {
            const parsed = parseFloat(e.target.value)
            if (!isNaN(parsed)) onChange(parsed)
          }}
          className="w-24 bg-[#0d1117] border border-[#30363d] rounded px-2 py-0.5 text-xs text-[#58a6ff] font-mono text-right focus:border-[#58a6ff] focus:outline-none"
        />
      </div>
      <input
        type="range"
        min={config.min}
        max={config.max}
        step={config.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-[#21262d] rounded-lg appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-3
                   [&::-webkit-slider-thumb]:h-3
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-[#58a6ff]
                   [&::-webkit-slider-thumb]:cursor-pointer"
      />
      <p className="text-[10px] text-[#8b949e] mt-1.5">{config.description}</p>
    </div>
  )
}

function BooleanInput({
  config,
  value,
  onChange,
}: {
  config: typeof parameterConfig[string]
  value: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <div className="bg-[#161b22] rounded p-3 border border-[#21262d]">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-xs font-medium text-[#c9d1d9]">{config.label}</label>
          <p className="text-[10px] text-[#8b949e] mt-0.5">{config.description}</p>
        </div>
        <button
          onClick={() => onChange(!value)}
          className={`relative w-10 h-5 rounded-full transition-colors ${
            value ? 'bg-[#238636]' : 'bg-[#21262d]'
          }`}
        >
          <div
            className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
              value ? 'translate-x-5' : 'translate-x-0.5'
            }`}
          />
        </button>
      </div>
    </div>
  )
}
