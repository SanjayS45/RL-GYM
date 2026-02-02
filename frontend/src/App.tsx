import { useState } from 'react'
import { Brain, Play, BarChart3, Layers, Database, Settings, Activity } from 'lucide-react'

import Header from './components/Header'
import Sidebar from './components/Sidebar'
import EnvironmentPanel from './components/panels/EnvironmentPanel'
import AgentPanel from './components/panels/AgentPanel'
import TrainingPanel from './components/panels/TrainingPanel'
import MetricsPanel from './components/panels/MetricsPanel'
import DatasetPanel from './components/panels/DatasetPanel'
import { useStore } from './store/useStore'

type TabType = 'environment' | 'agent' | 'training' | 'metrics' | 'datasets'

const tabs = [
  { id: 'environment' as TabType, label: 'Environment', icon: Layers, description: 'Configure environment and goals' },
  { id: 'agent' as TabType, label: 'Agent', icon: Brain, description: 'Select algorithm and hyperparameters' },
  { id: 'training' as TabType, label: 'Training', icon: Play, description: 'Run and monitor training' },
  { id: 'metrics' as TabType, label: 'Metrics', icon: BarChart3, description: 'Analyze training performance' },
  { id: 'datasets' as TabType, label: 'Datasets', icon: Database, description: 'Manage offline datasets' },
]

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('environment')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const { training, agent, environment } = useStore()

  const renderPanel = () => {
    switch (activeTab) {
      case 'environment':
        return <EnvironmentPanel />
      case 'agent':
        return <AgentPanel />
      case 'training':
        return <TrainingPanel />
      case 'metrics':
        return <MetricsPanel />
      case 'datasets':
        return <DatasetPanel />
      default:
        return <EnvironmentPanel />
    }
  }

  return (
    <div className="min-h-screen bg-[#010409] flex flex-col">
      {/* Top bar */}
      <header className="h-12 bg-[#0d1117] border-b border-[#30363d] flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-[#58a6ff]" />
            <span className="font-bold text-[#c9d1d9]">RL-GYM</span>
          </div>
          <span className="text-[10px] text-[#484f58] border border-[#30363d] rounded px-1.5 py-0.5">
            v1.0.0
          </span>
        </div>
        
        {/* Status indicators */}
        <div className="flex items-center gap-4">
          <StatusIndicator 
            label="Environment" 
            value={environment.type} 
            color="cyan" 
          />
          <StatusIndicator 
            label="Algorithm" 
            value={agent.algorithm} 
            color="purple" 
          />
          <StatusIndicator 
            label="Status" 
            value={training.status.toUpperCase()} 
            color={training.status === 'running' ? 'green' : training.status === 'paused' ? 'yellow' : 'gray'}
            pulse={training.status === 'running'}
          />
        </div>
      </header>
      
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <aside className={`bg-[#0d1117] border-r border-[#30363d] transition-all duration-200 ${
          sidebarCollapsed ? 'w-14' : 'w-56'
        }`}>
          <div className="p-2">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="w-full p-2 rounded hover:bg-[#161b22] text-[#8b949e] hover:text-[#c9d1d9] transition-colors flex items-center justify-center"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>
          
          <nav className="px-2 space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon
              const isActive = activeTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full p-2 rounded text-left transition-all flex items-center gap-3 ${
                    isActive
                      ? 'bg-[#161b22] text-[#c9d1d9]'
                      : 'text-[#8b949e] hover:bg-[#161b22] hover:text-[#c9d1d9]'
                  }`}
                  title={sidebarCollapsed ? tab.label : undefined}
                >
                  <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-[#58a6ff]' : ''}`} />
                  {!sidebarCollapsed && (
                    <div className="min-w-0">
                      <div className="text-sm font-medium truncate">{tab.label}</div>
                      <div className="text-[10px] text-[#484f58] truncate">{tab.description}</div>
                    </div>
                  )}
                </button>
              )
            })}
          </nav>
          
          {/* Quick stats in sidebar */}
          {!sidebarCollapsed && training.status !== 'idle' && (
            <div className="mt-4 mx-2 p-3 bg-[#161b22] rounded border border-[#30363d]">
              <div className="text-[10px] text-[#8b949e] mb-2">TRAINING PROGRESS</div>
              <div className="space-y-2">
                <div className="flex justify-between text-xs">
                  <span className="text-[#8b949e]">Steps</span>
                  <span className="text-[#c9d1d9] font-mono">{training.currentStep.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[#8b949e]">Episodes</span>
                  <span className="text-[#c9d1d9] font-mono">{training.currentEpisode}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[#8b949e]">Reward</span>
                  <span className="text-[#3fb950] font-mono">
                    {training.metrics.meanReward > 0 ? training.metrics.meanReward.toFixed(2) : '--'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </aside>
        
        {/* Main content */}
        <main className="flex-1 overflow-hidden p-4 bg-[#010409]">
          <div className="h-full">
            {renderPanel()}
          </div>
        </main>
      </div>
      
      {/* Bottom status bar */}
      <footer className="h-6 bg-[#0d1117] border-t border-[#30363d] flex items-center justify-between px-4 text-[10px]">
        <div className="flex items-center gap-4 text-[#8b949e]">
          <span>Python Backend: <span className="text-[#3fb950]">Ready</span></span>
          <span>|</span>
          <span>GPU: <span className="text-[#8b949e]">Not detected</span></span>
        </div>
        <div className="text-[#484f58]">
          RL-GYM Research Platform • MIT License
        </div>
      </footer>
    </div>
  )
}

function StatusIndicator({ 
  label, 
  value, 
  color,
  pulse = false
}: { 
  label: string
  value: string
  color: 'cyan' | 'purple' | 'green' | 'yellow' | 'gray'
  pulse?: boolean
}) {
  const colorClasses = {
    cyan: 'bg-[#58a6ff]/20 text-[#58a6ff] border-[#58a6ff]/30',
    purple: 'bg-[#a371f7]/20 text-[#a371f7] border-[#a371f7]/30',
    green: 'bg-[#238636]/20 text-[#3fb950] border-[#238636]/30',
    yellow: 'bg-[#9e6a03]/20 text-[#d29922] border-[#9e6a03]/30',
    gray: 'bg-[#21262d] text-[#8b949e] border-[#30363d]',
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-[#8b949e]">{label}:</span>
      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${colorClasses[color]} ${pulse ? 'animate-pulse' : ''}`}>
        {value}
      </span>
    </div>
  )
}

export default App
