import { motion } from 'framer-motion'
import { ChevronLeft, ChevronRight, LucideIcon } from 'lucide-react'
import clsx from 'clsx'

interface Tab {
  id: string
  label: string
  icon: LucideIcon
}

interface SidebarProps {
  tabs: Tab[]
  activeTab: string
  onTabChange: (tab: any) => void
  collapsed: boolean
  onToggleCollapse: () => void
}

export default function Sidebar({
  tabs,
  activeTab,
  onTabChange,
  collapsed,
  onToggleCollapse,
}: SidebarProps) {
  return (
    <motion.aside
      className="border-r border-gray-700/50 bg-surface-100/50 flex flex-col"
      animate={{ width: collapsed ? 64 : 200 }}
      transition={{ duration: 0.2 }}
    >
      <nav className="flex-1 p-2 space-y-1">
        {tabs.map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                'hover:bg-surface-200',
                isActive && 'bg-surface-200 border-l-2 border-accent-cyan'
              )}
            >
              <Icon
                className={clsx(
                  'w-5 h-5 flex-shrink-0 transition-colors',
                  isActive ? 'text-accent-cyan' : 'text-gray-400'
                )}
              />
              {!collapsed && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className={clsx(
                    'text-sm font-medium truncate',
                    isActive ? 'text-white' : 'text-gray-300'
                  )}
                >
                  {tab.label}
                </motion.span>
              )}
            </button>
          )
        })}
      </nav>

      {/* Collapse toggle */}
      <div className="p-2 border-t border-gray-700/50">
        <button
          onClick={onToggleCollapse}
          className="w-full flex items-center justify-center p-2 rounded-lg hover:bg-surface-200 transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronLeft className="w-5 h-5 text-gray-400" />
          )}
        </button>
      </div>
    </motion.aside>
  )
}

