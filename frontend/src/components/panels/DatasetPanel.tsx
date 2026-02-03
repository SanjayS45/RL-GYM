import { useState, useRef } from 'react'
import { Database, Upload, FileText, Trash2, Check, AlertCircle, Download, FolderOpen } from 'lucide-react'
import { useStore } from '../../store/useStore'

interface Dataset {
  id: string
  name: string
  type: 'demonstrations' | 'trajectories' | 'offline'
  size: number
  format: string
  compatible: boolean
  source: 'uploaded' | 'sample'
  description?: string
  algorithm?: string
}

// Sample datasets that can be "downloaded" or loaded
const sampleDatasets: (Dataset & { downloadUrl: string })[] = [
  // General datasets
  {
    id: 'sample_nav_expert',
    name: 'Navigation Expert Demos',
    type: 'demonstrations',
    size: 10000,
    format: 'json',
    compatible: true,
    source: 'sample',
    description: 'Expert trajectories from navigation environment. Useful for behavior cloning.',
    downloadUrl: '/datasets/navigation_expert_demos.json',
  },
  {
    id: 'sample_grid_random',
    name: 'GridWorld Random Policy',
    type: 'trajectories',
    size: 50000,
    format: 'json',
    compatible: true,
    source: 'sample',
    description: 'Random exploration data from GridWorld. Good for offline RL pretraining.',
    downloadUrl: '/datasets/gridworld_random_policy.json',
  },
  {
    id: 'sample_platformer_mixed',
    name: 'Platformer Mixed Quality',
    type: 'offline',
    size: 25000,
    format: 'json',
    compatible: true,
    source: 'sample',
    description: 'Mixed-quality trajectories combining novice and expert play.',
    downloadUrl: '/datasets/platformer_mixed_quality.json',
  },
  // PPO-specific datasets
  {
    id: 'ppo_nav_continuous',
    name: 'PPO Navigation Continuous',
    type: 'trajectories',
    size: 102400,
    format: 'json',
    compatible: true,
    source: 'sample',
    description: 'PPO rollout data for continuous 2D navigation. Includes log probs and value estimates.',
    downloadUrl: '/datasets/ppo_navigation_continuous.json',
    algorithm: 'PPO',
  },
  {
    id: 'ppo_grid_discrete',
    name: 'PPO GridWorld Discrete',
    type: 'trajectories',
    size: 102400,
    format: 'json',
    compatible: true,
    source: 'sample',
    description: 'PPO rollout data for discrete GridWorld maze. Includes action probs and GAE advantages.',
    downloadUrl: '/datasets/ppo_gridworld_discrete.json',
    algorithm: 'PPO',
  },
  {
    id: 'ppo_platformer_physics',
    name: 'PPO Platformer Physics',
    type: 'trajectories',
    size: 38400,
    format: 'json',
    compatible: true,
    source: 'sample',
    description: 'PPO rollout data for platformer with physics jumping. Multi-action format.',
    downloadUrl: '/datasets/ppo_platformer_physics.json',
    algorithm: 'PPO',
  },
]

export default function DatasetPanel() {
  const { datasetState, addDataset, removeDataset, selectDataset, setDatasetUsage } = useStore()
  const [isUploading, setIsUploading] = useState(false)
  const [activeTab, setActiveTab] = useState<'uploaded' | 'samples'>('uploaded')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    
    // Simulate upload delay
    await new Promise((resolve) => setTimeout(resolve, 1000))
    
    const newDataset: Dataset = {
      id: `upload_${Date.now()}`,
      name: file.name,
      type: 'trajectories',
      size: file.size,
      format: file.name.split('.').pop() || 'unknown',
      compatible: true,
      source: 'uploaded',
      description: `Uploaded file: ${file.name}`,
    }
    
    addDataset(newDataset)
    selectDataset(newDataset.id)
    setIsUploading(false)
    setActiveTab('uploaded')
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleUseSample = (sample: Dataset) => {
    // Check if already added
    if (datasetState.datasets.find(d => d.id === sample.id)) {
      selectDataset(sample.id)
      setActiveTab('uploaded')
      return
    }
    
    addDataset({ ...sample, source: 'sample' })
    selectDataset(sample.id)
    setActiveTab('uploaded')
  }

  const handleDelete = (id: string) => {
    removeDataset(id)
  }

  const handleDownloadSample = (sample: Dataset & { downloadUrl?: string }) => {
    if (sample.downloadUrl) {
      // Download from actual file
      const a = document.createElement('a')
      a.href = sample.downloadUrl
      a.download = `${sample.id}.json`
      a.click()
    } else {
      // Generate sample data as fallback
      const sampleData = generateSampleData(sample.type, 100)
      const blob = new Blob([JSON.stringify(sampleData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${sample.id}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const typeColors = {
    demonstrations: 'bg-[#238636]/20 text-[#3fb950] border-[#238636]/30',
    trajectories: 'bg-[#1f6feb]/20 text-[#58a6ff] border-[#1f6feb]/30',
    offline: 'bg-[#8957e5]/20 text-[#a371f7] border-[#8957e5]/30',
  }

  const selectedDataset = [...datasetState.datasets, ...sampleDatasets].find(d => d.id === datasetState.selectedDatasetId)

  return (
    <div className="h-full flex gap-4">
      {/* Dataset list */}
      <div className="w-96 flex flex-col gap-4">
        {/* Tabs */}
        <div className="flex border-b border-[#30363d]">
          <button
            onClick={() => setActiveTab('uploaded')}
            className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'uploaded'
                ? 'text-[#c9d1d9] border-b-2 border-[#f78166]'
                : 'text-[#8b949e] hover:text-[#c9d1d9]'
            }`}
          >
            My Datasets ({datasetState.datasets.length})
          </button>
          <button
            onClick={() => setActiveTab('samples')}
            className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'samples'
                ? 'text-[#c9d1d9] border-b-2 border-[#f78166]'
                : 'text-[#8b949e] hover:text-[#c9d1d9]'
            }`}
          >
            Sample Datasets
          </button>
        </div>

        {activeTab === 'uploaded' ? (
          <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
            <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
              <Database className="w-4 h-4 text-[#8b949e]" />
              <span className="text-sm font-medium text-[#c9d1d9]">Loaded Datasets</span>
            </div>
            
            <div className="p-3 flex-1 overflow-y-auto">
              {/* Upload button */}
              <label className="block mb-4">
                <div
                  className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                    isUploading
                      ? 'border-[#58a6ff] bg-[#58a6ff]/5'
                      : 'border-[#30363d] hover:border-[#484f58]'
                  }`}
                >
                  {isUploading ? (
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-4 h-4 border-2 border-[#58a6ff] border-t-transparent rounded-full animate-spin" />
                      <span className="text-[#58a6ff] text-sm">Uploading...</span>
                    </div>
                  ) : (
                    <>
                      <Upload className="w-8 h-8 mx-auto text-[#484f58] mb-2" />
                      <p className="text-sm text-[#8b949e]">
                        Click to upload dataset
                      </p>
                      <p className="text-xs text-[#484f58] mt-1">
                        JSON, CSV, HDF5 supported
                      </p>
                    </>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept=".h5,.hdf5,.csv,.json,.npz"
                  onChange={handleFileUpload}
                  disabled={isUploading}
                />
              </label>

              {/* Dataset list */}
              {datasetState.datasets.length === 0 ? (
                <div className="text-center py-8 text-[#8b949e]">
                  <FolderOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No datasets loaded</p>
                  <p className="text-xs text-[#484f58] mt-1">
                    Upload a file or use a sample dataset
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {datasetState.datasets.map((dataset) => (
                    <div
                      key={dataset.id}
                      className={`p-3 rounded border cursor-pointer transition-all ${
                        datasetState.selectedDatasetId === dataset.id
                          ? 'border-[#58a6ff] bg-[#58a6ff]/10'
                          : 'border-[#30363d] hover:border-[#484f58] bg-[#161b22]'
                      }`}
                      onClick={() => selectDataset(dataset.id)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-[#8b949e]" />
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-[#c9d1d9] text-sm">{dataset.name}</span>
                              {datasetState.selectedDatasetId === dataset.id && (
                                <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#238636]/20 text-[#3fb950] border border-[#238636]/30">
                                  SELECTED
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 mt-1">
                              <span className={`text-[10px] px-1.5 py-0.5 rounded border ${typeColors[dataset.type]}`}>
                                {dataset.type}
                              </span>
                              <span className="text-[10px] text-[#8b949e]">
                                {formatSize(dataset.size)}
                              </span>
                            </div>
                          </div>
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleDelete(dataset.id)
                          }}
                          className="p-1 hover:bg-[#da3633]/20 rounded transition-colors"
                        >
                          <Trash2 className="w-4 h-4 text-[#8b949e] hover:text-[#f85149]" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md flex flex-col">
            <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
              <Download className="w-4 h-4 text-[#8b949e]" />
              <span className="text-sm font-medium text-[#c9d1d9]">Sample Datasets</span>
            </div>
            <div className="p-3 flex-1 overflow-y-auto">
              <p className="text-xs text-[#8b949e] mb-3">
                Pre-made datasets for testing and learning. Click "Use" to load into your session.
              </p>
              <div className="space-y-2">
                {sampleDatasets.map((sample) => (
                  <div
                    key={sample.id}
                    className="p-3 rounded border border-[#30363d] bg-[#161b22]"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-[#c9d1d9] text-sm">{sample.name}</span>
                          {sample.algorithm && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#a371f7]/20 text-[#a371f7] border border-[#a371f7]/30 font-mono">
                              {sample.algorithm}
                            </span>
                          )}
                          {datasetState.datasets.find(d => d.id === sample.id) && (
                            <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#238636]/20 text-[#3fb950] border border-[#238636]/30">
                              LOADED
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded border ${typeColors[sample.type]}`}>
                            {sample.type}
                          </span>
                          <span className="text-[10px] text-[#8b949e]">
                            {sample.size.toLocaleString()} samples
                          </span>
                        </div>
                      </div>
                    </div>
                    <p className="text-[10px] text-[#8b949e] mb-2">{sample.description}</p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleUseSample(sample)}
                        className={`flex-1 text-xs font-medium py-1.5 px-3 rounded flex items-center justify-center gap-1.5 transition-colors ${
                          datasetState.datasets.find(d => d.id === sample.id)
                            ? 'bg-[#238636]/20 text-[#3fb950] border border-[#238636]/30'
                            : 'bg-[#238636] hover:bg-[#2ea043] text-white'
                        }`}
                      >
                        <Check className="w-3 h-3" />
                        {datasetState.datasets.find(d => d.id === sample.id) ? 'Select' : 'Use Dataset'}
                      </button>
                      <button
                        onClick={() => handleDownloadSample(sample)}
                        className="bg-[#30363d] hover:bg-[#484f58] text-[#c9d1d9] text-xs font-medium py-1.5 px-3 rounded flex items-center justify-center gap-1.5 transition-colors"
                      >
                        <Download className="w-3 h-3" />
                        Download
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Dataset details */}
      <div className="flex-1 bg-[#0d1117] border border-[#30363d] rounded-md">
        {selectedDataset ? (
          <DatasetDetails
            dataset={selectedDataset}
            useForPretraining={datasetState.useForPretraining}
            useForFinetuning={datasetState.useForFinetuning}
            onSetUsage={setDatasetUsage}
          />
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Database className="w-12 h-12 text-[#30363d] mx-auto mb-4" />
              <p className="text-[#8b949e] text-sm">Select a dataset to view details</p>
              <p className="text-[#484f58] text-xs mt-1">Or upload/load a dataset from the left panel</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function DatasetDetails({ 
  dataset, 
  useForPretraining, 
  useForFinetuning,
  onSetUsage 
}: { 
  dataset: Dataset
  useForPretraining: boolean
  useForFinetuning: boolean
  onSetUsage: (pretraining: boolean, finetuning: boolean) => void
}) {
  const typeColors = {
    demonstrations: 'bg-[#238636]/10 border-[#238636]/30 text-[#3fb950]',
    trajectories: 'bg-[#1f6feb]/10 border-[#1f6feb]/30 text-[#58a6ff]',
    offline: 'bg-[#8957e5]/10 border-[#8957e5]/30 text-[#a371f7]',
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 py-2 border-b border-[#30363d] flex items-center gap-2">
        <FileText className="w-4 h-4 text-[#8b949e]" />
        <span className="text-sm font-medium text-[#c9d1d9]">{dataset.name}</span>
        {dataset.algorithm && (
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-[#a371f7]/20 text-[#a371f7] border border-[#a371f7]/30 font-mono ml-auto">
            {dataset.algorithm}
          </span>
        )}
      </div>

      <div className="flex-1 p-4 overflow-y-auto">
        {/* Compatibility check */}
        <div className={`flex items-center gap-2 p-3 rounded-md mb-4 border ${
          dataset.compatible
            ? 'bg-[#238636]/10 border-[#238636]/30'
            : 'bg-[#da3633]/10 border-[#da3633]/30'
        }`}>
          {dataset.compatible ? (
            <>
              <Check className="w-4 h-4 text-[#3fb950]" />
              <span className="text-[#3fb950] text-sm">Compatible with current environment</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-4 h-4 text-[#f85149]" />
              <span className="text-[#f85149] text-sm">May not be compatible</span>
            </>
          )}
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-[#161b22] border border-[#30363d] rounded-md p-3">
            <div className="text-[10px] text-[#8b949e] mb-1">Samples</div>
            <div className="text-lg font-mono font-semibold text-[#c9d1d9]">
              {dataset.size.toLocaleString()}
            </div>
          </div>
          <div className="bg-[#161b22] border border-[#30363d] rounded-md p-3">
            <div className="text-[10px] text-[#8b949e] mb-1">Format</div>
            <div className="text-lg font-mono font-semibold text-[#c9d1d9] uppercase">
              {dataset.format}
            </div>
          </div>
          <div className="bg-[#161b22] border border-[#30363d] rounded-md p-3">
            <div className="text-[10px] text-[#8b949e] mb-1">Type</div>
            <div className={`text-sm font-medium px-2 py-0.5 rounded border inline-block ${typeColors[dataset.type]}`}>
              {dataset.type}
            </div>
          </div>
        </div>

        {/* Description */}
        {dataset.description && (
          <div className="mb-4">
            <h3 className="text-xs font-medium text-[#8b949e] mb-2">Description</h3>
            <p className="text-sm text-[#c9d1d9]">{dataset.description}</p>
          </div>
        )}

        {/* Sample data preview */}
        <div className="mb-4">
          <h3 className="text-xs font-medium text-[#8b949e] mb-2">Sample Data Structure</h3>
          <div className="bg-[#161b22] border border-[#30363d] rounded-md p-3 font-mono text-xs overflow-x-auto">
            <pre className="text-[#8b949e]">
{`{
  "observation": [0.234, -0.156, 0.891, 0.023],
  "action": ${dataset.type === 'demonstrations' ? '2' : '[0.5, -0.3]'},
  "reward": 0.5,
  "next_observation": [0.245, -0.142, 0.903, 0.031],
  "done": false,
  "info": {}
}`}
            </pre>
          </div>
        </div>

        {/* Usage Options */}
        <div className="mb-4">
          <h3 className="text-xs font-medium text-[#8b949e] mb-2">Training Usage</h3>
          <div className="space-y-2">
            <label className="flex items-center gap-3 p-3 bg-[#161b22] border border-[#30363d] rounded-md cursor-pointer hover:border-[#484f58]">
              <input
                type="checkbox"
                checked={useForPretraining}
                onChange={(e) => onSetUsage(e.target.checked, useForFinetuning)}
                className="w-4 h-4 rounded border-[#30363d] text-[#238636] focus:ring-[#238636]"
              />
              <div>
                <div className="text-sm text-[#c9d1d9]">Use for Pretraining</div>
                <div className="text-[10px] text-[#8b949e]">Initialize policy from this dataset before RL training</div>
              </div>
            </label>
            <label className="flex items-center gap-3 p-3 bg-[#161b22] border border-[#30363d] rounded-md cursor-pointer hover:border-[#484f58]">
              <input
                type="checkbox"
                checked={useForFinetuning}
                onChange={(e) => onSetUsage(useForPretraining, e.target.checked)}
                className="w-4 h-4 rounded border-[#30363d] text-[#238636] focus:ring-[#238636]"
              />
              <div>
                <div className="text-sm text-[#c9d1d9]">Use for Fine-tuning</div>
                <div className="text-[10px] text-[#8b949e]">Mix with online RL data during training</div>
              </div>
            </label>
          </div>
        </div>

        {/* Status */}
        {(useForPretraining || useForFinetuning) && (
          <div className="p-3 bg-[#238636]/10 border border-[#238636]/30 rounded-md">
            <div className="flex items-center gap-2">
              <Check className="w-4 h-4 text-[#3fb950]" />
              <span className="text-sm text-[#3fb950]">
                Dataset configured for {useForPretraining && useForFinetuning ? 'pretraining and fine-tuning' : useForPretraining ? 'pretraining' : 'fine-tuning'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function generateSampleData(type: string, count: number) {
  const data = []
  for (let i = 0; i < count; i++) {
    data.push({
      observation: [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1],
      action: type === 'demonstrations' ? Math.floor(Math.random() * 4) : [Math.random() * 2 - 1, Math.random() * 2 - 1],
      reward: Math.random() * 10 - 2,
      next_observation: [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1],
      done: Math.random() < 0.1,
      info: {},
    })
  }
  return data
}
