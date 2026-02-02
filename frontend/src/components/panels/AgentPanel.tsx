import { useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Zap, Settings2, Info } from 'lucide-react'
import { useStore } from '../../store/useStore'

const algorithms = [
  {
    id: 'PPO',
    name: 'PPO',
    fullName: 'Proximal Policy Optimization',
    type: 'on_policy',
    description: 'Stable policy gradient with clipped objective',
    color: 'cyan',
  },
  {
    id: 'DQN',
    name: 'DQN',
    fullName: 'Deep Q-Network',
    type: 'off_policy',
    description: 'Value-based with experience replay',
    color: 'purple',
  },
  {
    id: 'SAC',
    name: 'SAC',
    fullName: 'Soft Actor-Critic',
    type: 'off_policy',
    description: 'Maximum entropy for continuous control',
    color: 'pink',
  },
  {
    id: 'A2C',
    name: 'A2C',
    fullName: 'Advantage Actor-Critic',
    type: 'on_policy',
    description: 'Synchronous actor-critic method',
    color: 'green',
  },
]

export default function AgentPanel() {
  const { agent, setAlgorithm, setHyperparameter, resetAgentToDefaults } = useStore()
  const [showAdvanced, setShowAdvanced] = useState(false)

  const selectedAlgo = algorithms.find((a) => a.id === agent.algorithm)

  return (
    <div className="h-full flex gap-6">
      {/* Algorithm selection */}
      <div className="w-96 flex flex-col">
        <div className="panel flex-1">
          <h2 className="panel-header">
            <Brain className="w-5 h-5" />
            RL Algorithm
          </h2>

          <div className="grid grid-cols-2 gap-3">
            {algorithms.map((algo) => (
              <motion.button
                key={algo.id}
                onClick={() => setAlgorithm(algo.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={`p-4 rounded-lg border text-left transition-all ${
                  agent.algorithm === algo.id
                    ? `border-accent-${algo.color} bg-surface-200 glow-${algo.color}`
                    : 'border-gray-700 hover:border-gray-600 bg-surface-300/30'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-display font-bold text-lg">{algo.name}</span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      algo.type === 'on_policy'
                        ? 'bg-accent-green/20 text-accent-green'
                        : 'bg-accent-orange/20 text-accent-orange'
                    }`}
                  >
                    {algo.type.replace('_', '-')}
                  </span>
                </div>
                <p className="text-xs text-gray-400">{algo.description}</p>
              </motion.button>
            ))}
          </div>

          {/* Algorithm info */}
          {selectedAlgo && (
            <div className="mt-4 p-4 bg-surface-300/30 rounded-lg border border-gray-700/50">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-accent-cyan flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-medium text-white">{selectedAlgo.fullName}</h3>
                  <p className="text-sm text-gray-400 mt-1">
                    {selectedAlgo.type === 'on_policy'
                      ? 'Learns from current policy rollouts. More stable but sample inefficient.'
                      : 'Learns from stored experiences. More sample efficient but can be unstable.'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Hyperparameters */}
      <div className="flex-1 panel">
        <div className="flex items-center justify-between mb-4">
          <h2 className="panel-header mb-0">
            <Settings2 className="w-5 h-5" />
            Hyperparameters
          </h2>
          <button
            onClick={() => resetAgentToDefaults(agent.algorithm)}
            className="text-sm text-gray-400 hover:text-accent-cyan transition-colors"
          >
            Reset to defaults
          </button>
        </div>

        <div className="grid grid-cols-2 gap-6">
          {/* Learning Rate */}
          <HyperparameterSlider
            label="Learning Rate"
            value={agent.learningRate}
            min={0.00001}
            max={0.01}
            step={0.00001}
            format={(v) => v.toExponential(1)}
            onChange={(v) => setHyperparameter('learningRate', v)}
            description="Step size for gradient updates"
          />

          {/* Gamma */}
          <HyperparameterSlider
            label="Discount Factor (γ)"
            value={agent.gamma}
            min={0.9}
            max={0.999}
            step={0.001}
            format={(v) => v.toFixed(3)}
            onChange={(v) => setHyperparameter('gamma', v)}
            description="Importance of future rewards"
          />

          {/* Batch Size */}
          <HyperparameterSlider
            label="Batch Size"
            value={agent.batchSize}
            min={16}
            max={512}
            step={16}
            format={(v) => v.toString()}
            onChange={(v) => setHyperparameter('batchSize', v)}
            description="Samples per gradient update"
          />

          {/* N Steps */}
          {(agent.algorithm === 'PPO' || agent.algorithm === 'A2C') && (
            <HyperparameterSlider
              label="Rollout Steps"
              value={agent.nSteps}
              min={64}
              max={4096}
              step={64}
              format={(v) => v.toString()}
              onChange={(v) => setHyperparameter('nSteps', v)}
              description="Steps collected per rollout"
            />
          )}

          {/* N Epochs (PPO) */}
          {agent.algorithm === 'PPO' && (
            <HyperparameterSlider
              label="Epochs"
              value={agent.nEpochs}
              min={1}
              max={20}
              step={1}
              format={(v) => v.toString()}
              onChange={(v) => setHyperparameter('nEpochs', v)}
              description="Passes through rollout data"
            />
          )}

          {/* Clip Range (PPO) */}
          {agent.algorithm === 'PPO' && (
            <HyperparameterSlider
              label="Clip Range"
              value={agent.clipRange}
              min={0.1}
              max={0.4}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={(v) => setHyperparameter('clipRange', v)}
              description="PPO clipping parameter"
            />
          )}

          {/* Entropy Coefficient */}
          {(agent.algorithm === 'PPO' || agent.algorithm === 'A2C') && (
            <HyperparameterSlider
              label="Entropy Coefficient"
              value={agent.entCoef}
              min={0}
              max={0.1}
              step={0.005}
              format={(v) => v.toFixed(3)}
              onChange={(v) => setHyperparameter('entCoef', v)}
              description="Encourages exploration"
            />
          )}
        </div>

        {/* Network Architecture */}
        <div className="mt-8">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Network Architecture</h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Hidden Layers:</span>
              <div className="flex items-center gap-1">
                {agent.hiddenDims.map((dim, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 bg-surface-300 rounded text-sm font-mono text-accent-cyan"
                  >
                    {dim}
                  </span>
                ))}
              </div>
            </div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-gray-400 hover:text-white transition-colors"
            >
              {showAdvanced ? 'Hide' : 'Show'} advanced
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  format: (value: number) => string
  onChange: (value: number) => void
  description: string
}

function HyperparameterSlider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  description,
}: SliderProps) {
  return (
    <div className="bg-surface-300/30 rounded-lg p-4 border border-gray-700/30">
      <div className="flex items-center justify-between mb-2">
        <label className="text-sm font-medium text-white">{label}</label>
        <span className="font-mono text-accent-cyan text-sm">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-surface-200 rounded-lg appearance-none cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none
                   [&::-webkit-slider-thumb]:w-4
                   [&::-webkit-slider-thumb]:h-4
                   [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-accent-cyan
                   [&::-webkit-slider-thumb]:cursor-pointer
                   [&::-webkit-slider-thumb]:shadow-lg
                   [&::-webkit-slider-thumb]:shadow-accent-cyan/30"
      />
      <p className="text-xs text-gray-500 mt-2">{description}</p>
    </div>
  )
}

