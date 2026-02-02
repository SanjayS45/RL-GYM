import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, Activity, Clock } from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts'
import { useStore } from '../../store/useStore'

export default function MetricsPanel() {
  const { training } = useStore()

  // Generate chart data from history
  const rewardData = useMemo(() => {
    return training.history.rewards.map((reward, i) => ({
      episode: i + 1,
      reward,
      avgReward: training.history.rewards
        .slice(Math.max(0, i - 10), i + 1)
        .reduce((a, b) => a + b, 0) / Math.min(i + 1, 10),
    }))
  }, [training.history.rewards])

  const lossData = useMemo(() => {
    return training.history.losses.map((loss, i) => ({
      step: training.history.steps[i] || i * 100,
      loss,
    }))
  }, [training.history.losses, training.history.steps])

  // Sample data for demo when no training
  const sampleRewardData = Array.from({ length: 50 }, (_, i) => ({
    episode: i + 1,
    reward: Math.sin(i / 5) * 30 + 50 + Math.random() * 20 + i * 0.5,
    avgReward: Math.sin(i / 5) * 20 + 50 + i * 0.5,
  }))

  const sampleLossData = Array.from({ length: 100 }, (_, i) => ({
    step: i * 1000,
    loss: 1 / (1 + i * 0.1) + Math.random() * 0.1,
  }))

  const displayRewardData = rewardData.length > 0 ? rewardData : sampleRewardData
  const displayLossData = lossData.length > 0 ? lossData : sampleLossData

  return (
    <div className="h-full flex flex-col gap-6">
      {/* Summary metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricSummaryCard
          icon={<TrendingUp className="w-5 h-5" />}
          label="Mean Reward"
          value={training.metrics.meanReward.toFixed(2)}
          change="+12.5%"
          trend="up"
          color="cyan"
        />
        <MetricSummaryCard
          icon={<BarChart3 className="w-5 h-5" />}
          label="Max Reward"
          value={training.metrics.maxReward.toFixed(2)}
          change="+8.3%"
          trend="up"
          color="green"
        />
        <MetricSummaryCard
          icon={<Activity className="w-5 h-5" />}
          label="Current Loss"
          value={training.metrics.loss.toFixed(4)}
          change="-15.2%"
          trend="down"
          color="purple"
        />
        <MetricSummaryCard
          icon={<Clock className="w-5 h-5" />}
          label="Avg Episode Length"
          value={training.metrics.episodeLength.toFixed(0)}
          change="+5.1%"
          trend="up"
          color="orange"
        />
      </div>

      {/* Charts */}
      <div className="flex-1 grid grid-cols-2 gap-6">
        {/* Reward chart */}
        <div className="panel">
          <h2 className="panel-header">
            <TrendingUp className="w-5 h-5" />
            Episode Rewards
          </h2>
          <div className="h-[calc(100%-60px)]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={displayRewardData}>
                <defs>
                  <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d9ff" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00d9ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="episode"
                  stroke="#666"
                  fontSize={12}
                  tickLine={false}
                />
                <YAxis stroke="#666" fontSize={12} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a2e',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: '#999' }}
                />
                <Area
                  type="monotone"
                  dataKey="reward"
                  stroke="#00d9ff"
                  strokeWidth={2}
                  fill="url(#rewardGradient)"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="avgReward"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                  strokeDasharray="5 5"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Loss chart */}
        <div className="panel">
          <h2 className="panel-header">
            <Activity className="w-5 h-5" />
            Training Loss
          </h2>
          <div className="h-[calc(100%-60px)]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayLossData}>
                <defs>
                  <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis
                  dataKey="step"
                  stroke="#666"
                  fontSize={12}
                  tickLine={false}
                  tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                />
                <YAxis stroke="#666" fontSize={12} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a2e',
                    border: '1px solid #333',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: '#999' }}
                  formatter={(value: number) => [value.toFixed(4), 'Loss']}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={false}
                  fill="url(#lossGradient)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Training log */}
      <div className="panel h-48">
        <h2 className="panel-header">
          <span className="text-accent-green">📋</span>
          Training Log
        </h2>
        <div className="h-[calc(100%-50px)] overflow-y-auto font-mono text-xs space-y-1">
          {training.history.rewards.length === 0 ? (
            <div className="text-gray-500 text-center py-4">
              Training logs will appear here...
            </div>
          ) : (
            training.history.rewards.slice(-20).map((reward, i) => (
              <div key={i} className="flex gap-4 text-gray-400">
                <span className="text-gray-600">[Episode {training.history.rewards.length - 20 + i + 1}]</span>
                <span>
                  Reward: <span className="text-accent-cyan">{reward.toFixed(2)}</span>
                </span>
                <span>
                  Loss: <span className="text-accent-purple">{training.history.losses[training.history.losses.length - 20 + i]?.toFixed(4) || 'N/A'}</span>
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function MetricSummaryCard({
  icon,
  label,
  value,
  change,
  trend,
  color,
}: {
  icon: React.ReactNode
  label: string
  value: string
  change: string
  trend: 'up' | 'down'
  color: 'cyan' | 'green' | 'purple' | 'orange'
}) {
  const colorClasses = {
    cyan: 'text-accent-cyan bg-accent-cyan/10',
    green: 'text-accent-green bg-accent-green/10',
    purple: 'text-accent-purple bg-accent-purple/10',
    orange: 'text-accent-orange bg-accent-orange/10',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="panel"
    >
      <div className="flex items-center gap-3 mb-3">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>{icon}</div>
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-2xl font-mono font-bold text-white">{value}</span>
        <span
          className={`text-sm ${
            trend === 'up' ? 'text-accent-green' : 'text-red-400'
          }`}
        >
          {change}
        </span>
      </div>
    </motion.div>
  )
}

