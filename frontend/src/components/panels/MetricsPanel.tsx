import { useMemo } from 'react'
import { BarChart3, TrendingUp, Activity, Clock, AlertCircle, Download } from 'lucide-react'
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
    if (!hasData) return []
    return training.history.rewards.map((reward, i) => ({
      episode: i + 1,
      reward,
      avgReward: training.history.rewards
        .slice(Math.max(0, i - 10), i + 1)
        .reduce((a, b) => a + b, 0) / Math.min(i + 1, 10),
    }))
  }, [training.history.rewards, hasData])

  const lossData = useMemo(() => {
    if (!hasData || training.history.losses.length === 0) return []
    return training.history.losses.map((loss, i) => ({
      step: training.history.steps[i] || i * 100,
      loss,
    }))
  }, [training.history.losses, training.history.steps, hasData])

  // Calculate statistics
  const stats = useMemo(() => {
    if (!hasData) return null
    const rewards = training.history.rewards
    const losses = training.history.losses
    
    return {
      meanReward: rewards.reduce((a, b) => a + b, 0) / rewards.length,
      maxReward: Math.max(...rewards),
      minReward: Math.min(...rewards),
      stdReward: Math.sqrt(rewards.reduce((sum, r) => sum + Math.pow(r - (rewards.reduce((a, b) => a + b, 0) / rewards.length), 2), 0) / rewards.length),
      lastLoss: losses.length > 0 ? losses[losses.length - 1] : null,
      totalEpisodes: rewards.length,
    }
  }, [training.history.rewards, training.history.losses, hasData])

  const handleExportCSV = () => {
    if (!hasData) return
    
    const header = 'episode,reward,loss,step\n'
    const rows = training.history.rewards.map((reward, i) => 
      `${i + 1},${reward},${training.history.losses[i] || ''},${training.history.steps[i] || ''}`
    ).join('\n')
    
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training_metrics_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="h-full flex flex-col gap-4">
      {/* Summary metrics */}
      <div className="grid grid-cols-5 gap-3">
        <MetricSummaryCard
          icon={<TrendingUp className="w-4 h-4" />}
          label="Mean Reward"
          value={stats ? stats.meanReward.toFixed(3) : '--'}
          subtext={stats ? `σ = ${stats.stdReward.toFixed(3)}` : undefined}
          color="cyan"
        />
        <MetricSummaryCard
          icon={<BarChart3 className="w-4 h-4" />}
          label="Max Reward"
          value={stats ? stats.maxReward.toFixed(3) : '--'}
          subtext={stats ? `Min: ${stats.minReward.toFixed(3)}` : undefined}
          color="green"
        />
        <MetricSummaryCard
          icon={<Activity className="w-4 h-4" />}
          label="Current Loss"
          value={stats?.lastLoss ? stats.lastLoss.toExponential(3) : '--'}
          color="purple"
        />
        <MetricSummaryCard
          icon={<Clock className="w-4 h-4" />}
          label="Episodes"
          value={stats ? stats.totalEpisodes.toString() : '0'}
          color="orange"
        />
        <MetricSummaryCard
          icon={<Clock className="w-4 h-4" />}
          label="Total Steps"
          value={training.currentStep.toLocaleString()}
          color="blue"
        />
      </div>

      {/* Charts */}
      <div className="flex-1 grid grid-cols-2 gap-4 min-h-0">
        {/* Reward chart */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
          <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-[#3fb950]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Episode Rewards</span>
            {hasData && (
              <span className="text-xs text-[#8b949e] ml-auto">{rewardData.length} episodes</span>
            )}
          </div>
          <div className="flex-1 p-4 min-h-0">
            {hasData && rewardData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={rewardData}>
                  <defs>
                    <linearGradient id="rewardGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3fb950" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3fb950" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                  <XAxis
                    dataKey="episode"
                    stroke="#484f58"
                    fontSize={10}
                    tickLine={false}
                    axisLine={{ stroke: '#21262d' }}
                  />
                  <YAxis 
                    stroke="#484f58" 
                    fontSize={10} 
                    tickLine={false}
                    axisLine={{ stroke: '#21262d' }}
                    tickFormatter={(v) => v.toFixed(1)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#161b22',
                      border: '1px solid #30363d',
                      borderRadius: '6px',
                      fontSize: '11px',
                    }}
                    labelStyle={{ color: '#8b949e' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="reward"
                    stroke="#3fb950"
                    strokeWidth={1.5}
                    fill="url(#rewardGradient)"
                    dot={false}
                    name="Reward"
                  />
                  <Line
                    type="monotone"
                    dataKey="avgReward"
                    stroke="#58a6ff"
                    strokeWidth={1.5}
                    dot={false}
                    strokeDasharray="4 4"
                    name="Moving Avg (10)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyChartState message="Reward data will appear here once training begins" />
            )}
          </div>
        </div>

        {/* Loss chart */}
        <div className="bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
          <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#a371f7]" />
            <span className="text-sm font-medium text-[#c9d1d9]">Training Loss</span>
            {hasData && lossData.length > 0 && (
              <span className="text-xs text-[#8b949e] ml-auto">{lossData.length} samples</span>
            )}
          </div>
          <div className="flex-1 p-4 min-h-0">
            {hasData && lossData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
                  <XAxis
                    dataKey="step"
                    stroke="#484f58"
                    fontSize={10}
                    tickLine={false}
                    axisLine={{ stroke: '#21262d' }}
                    tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                  />
                  <YAxis 
                    stroke="#484f58" 
                    fontSize={10} 
                    tickLine={false}
                    axisLine={{ stroke: '#21262d' }}
                    tickFormatter={(v) => v.toExponential(1)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#161b22',
                      border: '1px solid #30363d',
                      borderRadius: '6px',
                      fontSize: '11px',
                    }}
                    labelStyle={{ color: '#8b949e' }}
                    formatter={(value: number) => [value.toExponential(4), 'Loss']}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#a371f7"
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
      <div className="bg-[#0d1117] border border-[#30363d] rounded-md" style={{ height: '160px' }}>
        <div className="px-4 py-2 border-b border-[#30363d] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-[#3fb950]">▸</span>
            <span className="text-sm font-medium text-[#c9d1d9]">Data Log</span>
          </div>
          {hasData && (
            <button 
              onClick={handleExportCSV}
              className="flex items-center gap-1.5 text-xs text-[#8b949e] hover:text-[#c9d1d9] transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
          )}
        </div>
        <div className="h-[calc(100%-36px)] overflow-y-auto p-3 font-mono text-xs">
          {!hasData ? (
            <div className="text-[#8b949e] text-center py-6">
              <AlertCircle className="w-5 h-5 mx-auto mb-2 opacity-50" />
              <p>No training data available</p>
              <p className="text-[#484f58] text-[10px] mt-1">Start a training run to see data</p>
            </div>
          ) : (
            <div className="space-y-0.5">
              <div className="text-[#484f58] mb-2">
                {'  EP  |    REWARD    |     LOSS     |    STEP'}
              </div>
              <div className="text-[#484f58] mb-2">
                {'─────┼──────────────┼──────────────┼─────────'}
              </div>
              {training.history.rewards.slice(-20).map((reward, i) => {
                const episodeNum = training.history.rewards.length - 20 + i + 1
                if (episodeNum < 1) return null
                const loss = training.history.losses[training.history.losses.length - 20 + i]
                const step = training.history.steps[training.history.steps.length - 20 + i]
                return (
                  <div key={i} className="text-[#8b949e] hover:bg-[#161b22] px-1">
                    <span className="text-[#484f58]">{episodeNum.toString().padStart(4, ' ')}</span>
                    {' | '}
                    <span className="text-[#3fb950]">{reward.toFixed(6).padStart(12, ' ')}</span>
                    {' | '}
                    <span className="text-[#a371f7]">{loss ? loss.toExponential(4).padStart(12, ' ') : '         N/A'}</span>
                    {' | '}
                    <span className="text-[#8b949e]">{step ? step.toString().padStart(8, ' ') : '     N/A'}</span>
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
        <div className="w-12 h-12 mx-auto mb-3 rounded-lg bg-[#21262d] flex items-center justify-center">
          <BarChart3 className="w-6 h-6 text-[#484f58]" />
        </div>
        <p className="text-[#8b949e] text-xs max-w-[200px]">{message}</p>
      </div>
    </div>
  )
}

function MetricSummaryCard({
  icon,
  label,
  value,
  subtext,
  color,
}: {
  icon: React.ReactNode
  label: string
  value: string
  subtext?: string
  color: 'cyan' | 'green' | 'purple' | 'orange' | 'blue'
}) {
  const colorClasses = {
    cyan: 'text-[#58a6ff] bg-[#58a6ff]/10 border-[#58a6ff]/20',
    green: 'text-[#3fb950] bg-[#3fb950]/10 border-[#3fb950]/20',
    purple: 'text-[#a371f7] bg-[#a371f7]/10 border-[#a371f7]/20',
    orange: 'text-[#d29922] bg-[#d29922]/10 border-[#d29922]/20',
    blue: 'text-[#58a6ff] bg-[#58a6ff]/10 border-[#58a6ff]/20',
  }

  return (
    <div className="bg-[#0d1117] border border-[#30363d] rounded-md px-4 py-3">
      <div className="flex items-center gap-2 mb-2">
        <div className={`p-1.5 rounded border ${colorClasses[color]}`}>{icon}</div>
        <span className="text-xs text-[#8b949e]">{label}</span>
      </div>
      <div className="flex items-end justify-between">
        <span className="text-xl font-mono font-semibold text-[#c9d1d9]">{value}</span>
        {subtext && (
          <span className="text-[10px] text-[#8b949e]">{subtext}</span>
        )}
      </div>
    </div>
  )
}
