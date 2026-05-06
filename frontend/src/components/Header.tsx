import { Flame, Satellite } from 'lucide-react'
import { motion } from 'framer-motion'

export function Header() {
  return (
    <header className="header">
      <motion.div
        className="header__badge"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Flame size={16} />
        AI Research Assistant
      </motion.div>

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
