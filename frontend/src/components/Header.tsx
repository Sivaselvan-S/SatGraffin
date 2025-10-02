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
        Retrieval-Augmented Intelligence
      </motion.div>

      <motion.h1
        className="header__title"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        SatGraffin Copilot
      </motion.h1>

      <motion.p
        className="header__subtitle"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        Ask anything about the MOSDAC knowledge base and get traceable, source-backed responses in seconds.
      </motion.p>

      <motion.div
        className="header__meta"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <Satellite size={18} />
        Connected to SatGraffin RAG backend
      </motion.div>
    </header>
  )
}
