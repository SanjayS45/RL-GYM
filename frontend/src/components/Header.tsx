import { motion } from 'framer-motion'
import { Brain, Github, BookOpen } from 'lucide-react'

export default function Header() {
  return (
    <header className="h-16 border-b border-gray-700/50 bg-surface-100/80 backdrop-blur-sm">
      <div className="h-full px-6 flex items-center justify-between">
        {/* Logo */}
        <motion.div 
          className="flex items-center gap-3"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-neural-light to-accent-purple flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-neural-light to-accent-purple blur-lg opacity-50" />
          </div>
          <div>
            <h1 className="text-xl font-display font-bold text-gradient">RL-GYM</h1>
            <p className="text-xs text-gray-400">Interactive RL Training</p>
          </div>
        </motion.div>

        {/* Status indicators */}
        <motion.div 
          className="flex items-center gap-6"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <StatusIndicator label="API" connected={true} />
          <StatusIndicator label="WS" connected={false} />
          
          <div className="h-8 w-px bg-gray-700" />
          
          <nav className="flex items-center gap-4">
            <a 
              href="https://github.com/SanjayS45/RL-GYM" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2 hover:bg-surface-200 rounded-lg transition-colors"
            >
              <Github className="w-5 h-5 text-gray-400 hover:text-white" />
            </a>
            <a 
              href="#docs" 
              className="p-2 hover:bg-surface-200 rounded-lg transition-colors"
            >
              <BookOpen className="w-5 h-5 text-gray-400 hover:text-white" />
            </a>
          </nav>
        </motion.div>
      </div>
    </header>
  )
}

function StatusIndicator({ label, connected }: { label: string; connected: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400">{label}</span>
      <div className={`w-2 h-2 rounded-full ${connected ? 'bg-accent-green animate-pulse' : 'bg-gray-500'}`} />
    </div>
  )
}

