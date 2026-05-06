import { useState } from 'react'
import { ExternalLink, ChevronDown, ChevronUp } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface SourceLinksProps {
  links: string[]
}

function getDomain(url: string): string {
  try {
    const hostname = new URL(url).hostname
    return hostname.replace(/^www\./, '')
  } catch {
    return url.replace(/^https?:\/\//, '').split('/')[0]
  }
}

export function SourceLinks({ links }: SourceLinksProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const displayLimit = 3
  const hasMore = links.length > displayLimit
  const visibleLinks = isExpanded ? links : links.slice(0, displayLimit)

  return (
    <motion.div
      className="sources"
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <div className="sources__header">
        <span className="sources__heading">
          Sources ({links.length})
        </span>
        {hasMore && (
          <button
            type="button"
            className="sources__toggle"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? (
              <>
                Show less <ChevronUp size={14} />
              </>
            ) : (
              <>
                Show all <ChevronDown size={14} />
              </>
            )}
          </button>
        )}
      </div>
      <div className="sources__grid">
        <AnimatePresence mode="popLayout">
          {visibleLinks.map((link, index) => (
            <motion.a
              key={link}
              href={link}
              target="_blank"
              rel="noreferrer"
              className="sources__pill"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ delay: index * 0.05 }}
            >
              <ExternalLink size={14} />
              <span className="sources__domain">{getDomain(link)}</span>
            </motion.a>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}
