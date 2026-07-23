import { memo, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Sparkles, ChevronDown, Check, Cpu, Zap } from 'lucide-react'
import clsx from 'clsx'

export interface ModelOption {
  id: string
  name: string
  tag: string
  description: string
  badge: string
}

interface ModelSelectorProps {
  apiBaseUrl: string
  selectedModel: string
  onSelectModel: (modelId: string) => void
  disabled?: boolean
}

const DEFAULT_MODELS: ModelOption[] = [
  {
    id: 'gemini-2.0-flash',
    name: 'Gemini 2.0 Flash',
    tag: 'Recommended',
    description: 'Ultra-fast multimodal reasoning model optimized for RAG',
    badge: 'Default & Fast',
  },
  {
    id: 'gemini-2.5-flash',
    name: 'Gemini 2.5 Flash',
    tag: 'Advanced Reasoning',
    description: 'Enhanced reasoning engine (Free tier: 20 requests/day)',
    badge: 'High Intellect',
  },
  {
    id: 'gemini-1.5-pro',
    name: 'Gemini 1.5 Pro',
    tag: 'Deep Research',
    description: '2M token context window for complex synthesis',
    badge: 'Deep Research',
  },
  {
    id: 'gemini-2.5-pro',
    name: 'Gemini 2.5 Pro',
    tag: 'Flagship Pro',
    description: 'Top-tier flagship reasoning model for technical tasks',
    badge: 'Flagship',
  },
]

export const ModelSelector = memo(function ModelSelector({
  apiBaseUrl,
  selectedModel,
  onSelectModel,
  disabled = false,
}: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [models, setModels] = useState<ModelOption[]>(DEFAULT_MODELS)

  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await fetch(`${apiBaseUrl}/api/models`)
        if (res.ok) {
          const data = await res.json()
          if (data.models && data.models.length > 0) {
            setModels(data.models)
          }
          if (data.current_model && !localStorage.getItem('satgraffin.selected_model')) {
            onSelectModel(data.current_model)
          }
        }
      } catch (err) {
        console.warn('Could not fetch models list from backend, using defaults', err)
      }
    }
    fetchModels()
  }, [apiBaseUrl, onSelectModel])

  const activeModel = models.find((m) => m.id === selectedModel) || models[0]

  const handleSelect = async (modelId: string) => {
    onSelectModel(modelId)
    setIsOpen(false)
    localStorage.setItem('satgraffin.selected_model', modelId)

    try {
      await fetch(`${apiBaseUrl}/api/set-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      })
    } catch (err) {
      console.error('Failed to update backend model choice', err)
    }
  }

  return (
    <div className="model-selector">
      <button
        type="button"
        className={clsx(
          'model-selector__trigger',
          isOpen && 'model-selector__trigger--open',
          disabled && 'model-selector__trigger--disabled'
        )}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        title="Click to switch Gemini AI Model"
      >
        <span className="model-selector__icon-wrap">
          <Cpu size={15} className="model-selector__cpu-icon" />
        </span>
        <div className="model-selector__info">
          <span className="model-selector__name">{activeModel.name}</span>
          <span className="model-selector__tag">{activeModel.tag}</span>
        </div>
        <ChevronDown
          size={14}
          className={clsx('model-selector__chevron', isOpen && 'model-selector__chevron--open')}
        />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="model-selector__dropdown"
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
          >
            <div className="model-selector__dropdown-header">
              <Sparkles size={14} className="model-selector__sparkle" />
              <span>Select Gemini Engine</span>
            </div>

            <ul className="model-selector__list">
              {models.map((model) => {
                const isSelected = model.id === selectedModel
                return (
                  <li key={model.id}>
                    <button
                      type="button"
                      className={clsx(
                        'model-selector__option',
                        isSelected && 'model-selector__option--selected'
                      )}
                      onClick={() => handleSelect(model.id)}
                    >
                      <div className="model-selector__option-top">
                        <div className="model-selector__option-title">
                          {model.id.includes('flash') ? (
                            <Zap size={14} className="model-selector__icon--fast" />
                          ) : (
                            <Cpu size={14} className="model-selector__icon--pro" />
                          )}
                          <span>{model.name}</span>
                        </div>
                        <span
                          className={clsx(
                            'model-selector__badge',
                            model.badge === 'Default' && 'model-selector__badge--default',
                            model.badge === 'High Intellect' && 'model-selector__badge--warn'
                          )}
                        >
                          {model.badge}
                        </span>
                      </div>
                      <p className="model-selector__option-desc">{model.description}</p>
                      {isSelected && <Check size={14} className="model-selector__check" />}
                    </button>
                  </li>
                )
              })}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
})
