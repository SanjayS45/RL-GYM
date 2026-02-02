import { useState } from 'react'
import { motion } from 'framer-motion'
import { Database, Upload, FileText, Trash2, Check, AlertCircle } from 'lucide-react'

interface Dataset {
  id: string
  name: string
  type: 'demonstrations' | 'trajectories' | 'offline'
  size: number
  format: string
  compatible: boolean
}

const sampleDatasets: Dataset[] = [
  {
    id: '1',
    name: 'Navigation Expert Demos',
    type: 'demonstrations',
    size: 10000,
    format: 'h5',
    compatible: true,
  },
  {
    id: '2',
    name: 'Grid World Random',
    type: 'trajectories',
    size: 50000,
    format: 'json',
    compatible: true,
  },
]

export default function DatasetPanel() {
  const [datasets, setDatasets] = useState<Dataset[]>(sampleDatasets)
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    
    // Simulate upload
    await new Promise((resolve) => setTimeout(resolve, 1500))
    
    const newDataset: Dataset = {
      id: Date.now().toString(),
      name: file.name,
      type: 'trajectories',
      size: Math.floor(Math.random() * 10000) + 1000,
      format: file.name.split('.').pop() || 'unknown',
      compatible: true,
    }
    
    setDatasets([...datasets, newDataset])
    setIsUploading(false)
  }

  const handleDelete = (id: string) => {
    setDatasets(datasets.filter((d) => d.id !== id))
    if (selectedDataset === id) {
      setSelectedDataset(null)
    }
  }

  const typeColors = {
    demonstrations: 'bg-accent-green/20 text-accent-green',
    trajectories: 'bg-accent-cyan/20 text-accent-cyan',
    offline: 'bg-accent-purple/20 text-accent-purple',
  }

  return (
    <div className="h-full flex gap-6">
      {/* Dataset list */}
      <div className="w-96 flex flex-col gap-4">
        <div className="panel flex-1">
          <h2 className="panel-header">
            <Database className="w-5 h-5" />
            Datasets
          </h2>

          {/* Upload button */}
          <label className="block mb-4">
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
                isUploading
                  ? 'border-accent-cyan bg-accent-cyan/5'
                  : 'border-gray-600 hover:border-gray-500'
              }`}
            >
              {isUploading ? (
                <div className="flex items-center justify-center gap-2">
                  <div className="w-5 h-5 border-2 border-accent-cyan border-t-transparent rounded-full animate-spin" />
                  <span className="text-accent-cyan">Uploading...</span>
                </div>
              ) : (
                <>
                  <Upload className="w-8 h-8 mx-auto text-gray-500 mb-2" />
                  <p className="text-sm text-gray-400">
                    Drag & drop or click to upload
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    Supports HDF5, CSV, JSON
                  </p>
                </>
              )}
            </motion.div>
            <input
              type="file"
              className="hidden"
              accept=".h5,.hdf5,.csv,.json"
              onChange={handleFileUpload}
              disabled={isUploading}
            />
          </label>

          {/* Dataset list */}
          <div className="space-y-2">
            {datasets.map((dataset) => (
              <motion.div
                key={dataset.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedDataset === dataset.id
                    ? 'border-accent-cyan bg-surface-200'
                    : 'border-gray-700 hover:border-gray-600 bg-surface-300/30'
                }`}
                onClick={() => setSelectedDataset(dataset.id)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <FileText className="w-5 h-5 text-gray-400" />
                    <div>
                      <div className="font-medium text-white text-sm">
                        {dataset.name}
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span
                          className={`text-xs px-2 py-0.5 rounded ${
                            typeColors[dataset.type]
                          }`}
                        >
                          {dataset.type}
                        </span>
                        <span className="text-xs text-gray-500">
                          {dataset.size.toLocaleString()} samples
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      handleDelete(dataset.id)
                    }}
                    className="p-1 hover:bg-red-500/20 rounded transition-colors"
                  >
                    <Trash2 className="w-4 h-4 text-gray-500 hover:text-red-400" />
                  </button>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Dataset details */}
      <div className="flex-1 panel">
        {selectedDataset ? (
          <DatasetDetails
            dataset={datasets.find((d) => d.id === selectedDataset)!}
          />
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Database className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">Select a dataset to view details</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function DatasetDetails({ dataset }: { dataset: Dataset }) {
  return (
    <>
      <h2 className="panel-header">
        <FileText className="w-5 h-5" />
        {dataset.name}
      </h2>

      {/* Compatibility check */}
      <div
        className={`flex items-center gap-2 p-3 rounded-lg mb-6 ${
          dataset.compatible
            ? 'bg-accent-green/10 border border-accent-green/30'
            : 'bg-red-500/10 border border-red-500/30'
        }`}
      >
        {dataset.compatible ? (
          <>
            <Check className="w-5 h-5 text-accent-green" />
            <span className="text-accent-green">
              Compatible with selected environment
            </span>
          </>
        ) : (
          <>
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-400">
              May not be compatible with selected environment
            </span>
          </>
        )}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="metric-card">
          <div className="text-xs text-gray-400 mb-1">Total Samples</div>
          <div className="text-xl font-mono font-semibold text-white">
            {dataset.size.toLocaleString()}
          </div>
        </div>
        <div className="metric-card">
          <div className="text-xs text-gray-400 mb-1">Format</div>
          <div className="text-xl font-mono font-semibold text-white uppercase">
            {dataset.format}
          </div>
        </div>
        <div className="metric-card">
          <div className="text-xs text-gray-400 mb-1">Type</div>
          <div className="text-xl font-mono font-semibold text-white capitalize">
            {dataset.type}
          </div>
        </div>
      </div>

      {/* Sample data preview */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Sample Data</h3>
        <div className="bg-surface-300/50 rounded-lg p-4 font-mono text-xs overflow-x-auto">
          <pre className="text-gray-300">
{`{
  "observation": [0.234, -0.156, 0.891, 0.023],
  "action": 2,
  "reward": 0.5,
  "next_observation": [0.245, -0.142, 0.903, 0.031],
  "done": false
}`}
          </pre>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button className="btn-primary flex-1">
          Use for Pretraining
        </button>
        <button className="btn-secondary flex-1">
          Use for Fine-tuning
        </button>
      </div>
    </>
  )
}

