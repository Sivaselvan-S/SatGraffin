import { memo, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, Compass, ChevronDown, Check, Sparkles } from 'lucide-react'
import clsx from 'clsx'

export interface RAGModeOption {
  id: 'normal' | 'divex'
  name: string
  tag: string
  description: string
  badge: string
  saverMode: boolean
}

const RAG_MODES: RAGModeOption[] = [
  {
    id: 'normal',
    name: 'Normal mode',
    tag: 'Fast & Efficient',
    description: 'Uses local CPU heuristics to save API quota (1 API call/query).',
    badge: 'Quota Saver',
    saverMode: true,
  },
  {
    id: 'divex',
    name: 'DiveX mode',
    tag: 'Deep RAG Reasoning',
    description: 'Full multi-agent RAG with LLM HyDE, CRAG grading & query analysis (4 calls/query).',
    badge: 'Deep RAG',
    saverMode: false,
  },
]

interface ModeSelectorProps {
  apiBaseUrl: string
  disabled?: boolean
}

export const ModeSelector = memo(function ModeSelector({
  apiBaseUrl,
  disabled = false,
}: ModeSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedModeId, setSelectedModeId] = useState<'normal' | 'divex'>('normal')

  useEffect(() => {
    async function fetchStatus() {
      try {
        const res = await fetch(`${apiBaseUrl}/api/saver-mode`)
        if (res.ok) {
          const data = await res.json()
          if (data.status === 'ok') {
            setSelectedModeId(data.api_saver_mode ? 'normal' : 'divex')
          }
        }
      } catch (err) {
        console.warn('Could not fetch RAG mode from backend', err)
      }
    }
    fetchStatus()
  }, [apiBaseUrl])

  const activeMode = RAG_MODES.find((m) => m.id === selectedModeId) || RAG_MODES[0]

  const handleSelect = async (mode: RAGModeOption) => {
    setSelectedModeId(mode.id)
    setIsOpen(false)

    try {
      await fetch(`${apiBaseUrl}/api/set-saver-mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: mode.saverMode }),
      })
    } catch (err) {
      console.error('Failed to update backend RAG mode', err)
    }
  }

  return (
    <div className="model-selector" style={{ position: 'relative' }}>
      <button
        type="button"
        className={clsx(
          'model-selector__trigger',
          isOpen && 'model-selector__trigger--open',
          disabled && 'model-selector__trigger--disabled'
        )}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        title="Click to switch RAG Pipeline Mode"
        style={{
          borderColor: activeMode.id === 'normal' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(99, 102, 241, 0.4)',
        }}
      >
        <span className="model-selector__icon-wrap">
          {activeMode.id === 'normal' ? (
            <Zap size={15} style={{ color: '#10b981' }} />
          ) : (
            <Compass size={15} style={{ color: '#818cf8' }} />
          )}
        </span>
        <div className="model-selector__info">
          <span className="model-selector__name">{activeMode.name}</span>
          <span className="model-selector__tag">{activeMode.tag}</span>
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
            style={{ width: '280px' }}
          >
            <div className="model-selector__dropdown-header">
              <Sparkles size={14} className="model-selector__sparkle" />
              <span>Select RAG Execution Mode</span>
            </div>

            <ul className="model-selector__list">
              {RAG_MODES.map((mode) => {
                const isSelected = mode.id === selectedModeId
                return (
                  <li key={mode.id}>
                    <button
                      type="button"
                      className={clsx(
                        'model-selector__option',
                        isSelected && 'model-selector__option--selected'
                      )}
                      onClick={() => handleSelect(mode)}
                    >
                      <div className="model-selector__option-top">
                        <div className="model-selector__option-title">
                          {mode.id === 'normal' ? (
                            <Zap size={14} style={{ color: '#10b981' }} />
                          ) : (
                            <Compass size={14} style={{ color: '#818cf8' }} />
                          )}
                          <span>{mode.name}</span>
                        </div>
                        <span
                          className="model-selector__badge"
                          style={{
                            backgroundColor: mode.id === 'normal' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(99, 102, 241, 0.15)',
                            color: mode.id === 'normal' ? '#10b981' : '#818cf8',
                            border: `1px solid ${mode.id === 'normal' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(99, 102, 241, 0.3)'}`,
                          }}
                        >
                          {mode.badge}
                        </span>
                      </div>
                      <p className="model-selector__option-desc">{mode.description}</p>
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
