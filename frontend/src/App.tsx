import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Brain, Play, Settings, BarChart3, Layers, Database } from 'lucide-react'

import Header from './components/Header'
import Sidebar from './components/Sidebar'
import EnvironmentPanel from './components/panels/EnvironmentPanel'
import AgentPanel from './components/panels/AgentPanel'
import TrainingPanel from './components/panels/TrainingPanel'
import VisualizationCanvas from './components/visualization/VisualizationCanvas'
import MetricsPanel from './components/panels/MetricsPanel'
import DatasetPanel from './components/panels/DatasetPanel'

type TabType = 'environment' | 'agent' | 'training' | 'visualization' | 'metrics' | 'datasets'

const tabs = [
  { id: 'environment' as TabType, label: 'Environment', icon: Layers },
  { id: 'agent' as TabType, label: 'Agent', icon: Brain },
  { id: 'training' as TabType, label: 'Training', icon: Play },
  { id: 'visualization' as TabType, label: 'Visualization', icon: Settings },
  { id: 'metrics' as TabType, label: 'Metrics', icon: BarChart3 },
  { id: 'datasets' as TabType, label: 'Datasets', icon: Database },
]

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('environment')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const renderPanel = () => {
    switch (activeTab) {
      case 'environment':
        return <EnvironmentPanel />
      case 'agent':
        return <AgentPanel />
      case 'training':
        return <TrainingPanel />
      case 'visualization':
        return <VisualizationCanvas />
      case 'metrics':
        return <MetricsPanel />
      case 'datasets':
        return <DatasetPanel />
      default:
        return <EnvironmentPanel />
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        />
        
        <main className="flex-1 overflow-hidden p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              {renderPanel()}
            </motion.div>
          </AnimatePresence>
        </main>
        
        {/* Right sidebar with live metrics */}
        {activeTab === 'training' && (
          <motion.aside
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="border-l border-gray-700/50 overflow-y-auto"
          >
            <div className="p-4">
              <h3 className="font-display text-accent-cyan mb-4">Live Metrics</h3>
              <div className="space-y-3">
                <MetricCard label="Episode Reward" value="--" />
                <MetricCard label="Steps" value="0" />
                <MetricCard label="Episodes" value="0" />
                <MetricCard label="Loss" value="--" />
              </div>
            </div>
          </motion.aside>
        )}
      </div>
    </div>
  )
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-xl font-mono font-semibold text-white">{value}</div>
    </div>
  )
}

export default App

