import { Compass, ListChecks, Sparkle } from 'lucide-react'
import { motion } from 'framer-motion'

const prompts = [
  'What are the latest developments in quantum computing?',
  'How does climate change affect ocean currents?',
  'Explain the differences between React and Vue.js',
  'What is the current state of AI regulation worldwide?',
]

interface EmptyStateProps {
  onPromptClick?: (prompt: string) => void
}

export function EmptyState({ onPromptClick }: EmptyStateProps) {
  return (
    <div className="empty-state">
      <motion.div 
        className="empty-state__icon"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
      >
        <Sparkle size={28} />
      </motion.div>
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        Start the conversation
      </motion.h2>
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        SatGraffin searches the web in real-time, scrapes relevant sources, and uses AI to provide accurate, well-sourced answers. Try one of these prompts or ask anything.
      </motion.p>
      <ul className="empty-state__prompts">
        {prompts.map((prompt, index) => (
          <motion.li 
            key={prompt}
            onClick={() => onPromptClick?.(prompt)}
            className="empty-state__prompt-item"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 + index * 0.1 }}
            whileHover={{ scale: 1.02, x: 4 }}
            whileTap={{ scale: 0.98 }}
          >
            <Compass size={16} />
            {prompt}
          </motion.li>
        ))}
      </ul>
      <motion.p 
        className="empty-state__hint"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
      >
        <ListChecks size={14} /> Responses include source trails so you can verify the facts.
      </motion.p>
    </div>
  )
}
