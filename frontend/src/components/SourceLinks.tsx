import { ExternalLink } from 'lucide-react'
import { motion } from 'framer-motion'

interface SourceLinksProps {
  links: string[]
}

export function SourceLinks({ links }: SourceLinksProps) {
  return (
    <motion.div
      className="sources"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <span className="sources__heading">Sources</span>
      <div className="sources__grid">
        {links.map((link) => (
          <a
            key={link}
            href={link}
            target="_blank"
            rel="noreferrer"
            className="sources__pill"
          >
            <ExternalLink size={14} />
            <span>{link.replace(/^https?:\/\//, '')}</span>
          </a>
        ))}
      </div>
    </motion.div>
  )
}
