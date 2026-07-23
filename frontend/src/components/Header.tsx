import { Flame, Satellite } from 'lucide-react'
import { motion } from 'framer-motion'
import { ModelSelector } from './ModelSelector'
import { ModeSelector } from './ModeSelector'

interface HeaderProps {
  apiBaseUrl: string
  selectedModel: string
  onSelectModel: (modelId: string) => void
  disabled?: boolean
}

export function Header({ apiBaseUrl, selectedModel, onSelectModel, disabled }: HeaderProps) {
  return (
    <header className="header">
      <div className="header__top-bar" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', maxWidth: '840px', margin: '0 auto 0.75rem auto', flexWrap: 'wrap', gap: '0.75rem' }}>
        <motion.div
          className="header__badge"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Flame size={16} />
          AI Research Assistant
        </motion.div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <ModeSelector apiBaseUrl={apiBaseUrl} disabled={disabled} />
          <ModelSelector
            apiBaseUrl={apiBaseUrl}
            selectedModel={selectedModel}
            onSelectModel={onSelectModel}
            disabled={disabled}
          />
        </div>
      </div>

      <motion.h1
        className="header__title"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        SatGraffin
      </motion.h1>

      <motion.p
        className="header__subtitle"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        Ask any question and get accurate, source-backed answers powered by real-time web search and AI.
      </motion.p>

      <motion.div
        className="header__meta"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <Satellite size={18} />
        Powered by Web Search + RAG + Gemini AI
      </motion.div>
    </header>
  )
}
