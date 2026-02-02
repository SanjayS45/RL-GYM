import { useMemo } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, TrendingUp, Activity, Clock, AlertCircle } from 'lucide-react'
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

  const hasData = training.history.rewards.length > 0

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

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Summary metrics - only show real values when data exists */}
      <div className="grid grid-cols-4 gap-3">
        <MetricSummaryCard
          icon={<TrendingUp className="w-4 h-4" />}
          label="Mean Reward"
          value={hasData ? training.metrics.meanReward.toFixed(2) : '--'}
          change={hasData ? undefined : undefined}
          trend={hasData && training.metrics.meanReward > 0 ? 'up' : undefined}
          color="cyan"
        />
        <MetricSummaryCard
          icon={<BarChart3 className="w-4 h-4" />}
          label="Max Reward"
          value={hasData ? training.metrics.maxReward.toFixed(2) : '--'}
          change={undefined}
          trend={hasData && training.metrics.maxReward > 0 ? 'up' : undefined}
          color="green"
        />
        <MetricSummaryCard
          icon={<Activity className="w-4 h-4" />}
          label="Current Loss"
          value={hasData ? training.metrics.loss.toFixed(6) : '--'}
          change={undefined}
          trend={undefined}
          color="purple"
        />
        <MetricSummaryCard
          icon={<Clock className="w-4 h-4" />}
          label="Avg Episode Length"
          value={hasData ? training.metrics.episodeLength.toFixed(0) : '--'}
          change={undefined}
          trend={undefined}
          color="orange"
        />
      </div>

      {/* Charts - only show when data exists */}
      <div className="flex-1 grid grid-cols-2 gap-4">
        {/* Reward chart */}
        <div className="panel">
          <h2 className="panel-header text-sm">
            <TrendingUp className="w-4 h-4" />
            Episode Rewards
          </h2>
          <div className="h-[calc(100%-50px)]">
            {hasData ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rewardData}>
                  <defs>
                    <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00d9ff" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#00d9ff" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                  <XAxis
                    dataKey="episode"
                    stroke="#4a4a5e"
                    fontSize={10}
                    tickLine={false}
                    axisLine={{ stroke: '#2a2a3e' }}
                  />
                  <YAxis 
                    stroke="#4a4a5e" 
                    fontSize={10} 
                    tickLine={false}
                    axisLine={{ stroke: '#2a2a3e' }}
                    tickFormatter={(v) => v.toFixed(1)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a2e',
                      border: '1px solid #2a2a3e',
                      borderRadius: '4px',
                      fontSize: '11px',
                    }}
                    labelStyle={{ color: '#666' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="reward"
                    stroke="#00d9ff"
                    strokeWidth={1.5}
                    fill="url(#rewardGradient)"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="avgReward"
                    stroke="#a855f7"
                    strokeWidth={1.5}
                    dot={false}
                    strokeDasharray="4 4"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChartState message="Reward data will appear here once training begins" />
            )}
          </div>
        </div>

        {/* Loss chart */}
        <div className="panel">
          <h2 className="panel-header text-sm">
            <Activity className="w-4 h-4" />
            Training Loss
          </h2>
          <div className="h-[calc(100%-50px)]">
            {hasData && lossData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
                  <XAxis
                    dataKey="step"
                    stroke="#4a4a5e"
                    fontSize={10}
                    tickLine={false}
                    axisLine={{ stroke: '#2a2a3e' }}
                    tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`}
                  />
                  <YAxis 
                    stroke="#4a4a5e" 
                    fontSize={10} 
                    tickLine={false}
                    axisLine={{ stroke: '#2a2a3e' }}
                    tickFormatter={(v) => v.toExponential(1)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1a1a2e',
                      border: '1px solid #2a2a3e',
                      borderRadius: '4px',
                      fontSize: '11px',
                    }}
                    labelStyle={{ color: '#666' }}
                    formatter={(value: number) => [value.toExponential(4), 'Loss']}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#a855f7"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChartState message="Loss curve will appear here once training begins" />
            )}
          </div>
        </div>
      </div>

      {/* Training log */}
      <div className="panel h-40">
        <h2 className="panel-header text-sm">
          <span className="text-green-500">▸</span>
          Training Log
        </h2>
        <div className="h-[calc(100%-40px)] overflow-y-auto font-mono text-xs">
          {training.history.rewards.length === 0 ? (
            <div className="text-gray-500 text-center py-6">
              <AlertCircle className="w-5 h-5 mx-auto mb-2 opacity-50" />
              <p>No training data available</p>
              <p className="text-gray-600 text-[10px] mt-1">Start a training run to see logs</p>
            </div>
          ) : (
            <div className="space-y-0.5">
              {training.history.rewards.slice(-30).map((reward, i) => {
                const episodeNum = training.history.rewards.length - 30 + i + 1
                const loss = training.history.losses[training.history.losses.length - 30 + i]
                return (
                  <div key={i} className="flex gap-3 text-gray-400 hover:bg-surface-200/30 px-2 py-0.5">
                    <span className="text-gray-600 w-16">[E{episodeNum.toString().padStart(4, '0')}]</span>
                    <span className="w-28">
                      R: <span className="text-cyan-400">{reward.toFixed(3)}</span>
                    </span>
                    <span className="w-32">
                      L: <span className="text-purple-400">{loss?.toExponential(2) || 'N/A'}</span>
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function EmptyChartState({ message }: { message: string }) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 mx-auto mb-3 rounded-lg bg-surface-300/50 flex items-center justify-center">
          <BarChart3 className="w-6 h-6 text-gray-600" />
        </div>
        <p className="text-gray-500 text-xs max-w-[200px]">{message}</p>
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
  change?: string
  trend?: 'up' | 'down'
  color: 'cyan' | 'green' | 'purple' | 'orange'
}) {
  const colorClasses = {
    cyan: 'text-cyan-400 bg-cyan-400/10 border-cyan-400/20',
    green: 'text-green-400 bg-green-400/10 border-green-400/20',
    purple: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
    orange: 'text-orange-400 bg-orange-400/10 border-orange-400/20',
  }

  return (
    <div className="panel py-3 px-4">
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1.5 rounded border ${colorClasses[color]}`}>{icon}</div>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-xl font-mono font-semibold text-white">{value}</span>
        {change && trend && (
          <span
            className={`text-xs ${
              trend === 'up' ? 'text-green-400' : 'text-red-400'
            }`}
          >
            {change}
          </span>
        )}
      </div>
    </div>
  )
}
