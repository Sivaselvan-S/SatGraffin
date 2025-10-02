import { memo } from 'react'
import { motion } from 'framer-motion'
import { Bot, UserRound } from 'lucide-react'
import clsx from 'clsx'
import type { ChatMessage } from '../types'
import { SourceLinks } from './SourceLinks'

interface MessageBubbleProps {
  message: ChatMessage
}

export const MessageBubble = memo(function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'

  return (
    <motion.li
      layout
      className={clsx('message', isUser ? 'message--user' : 'message--assistant')}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -12 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
    >
      <span className="message__avatar" aria-hidden>
        {isUser ? <UserRound size={18} /> : <Bot size={18} />}
      </span>
      <div
        className={clsx(
          'message__body',
          message.isError && 'message__body--error',
        )}
      >
        <p className="message__content">{message.content}</p>
        {message.sources && message.sources.length > 0 && (
          <SourceLinks links={message.sources} />
        )}
        <span className="message__timestamp">
          {new Date(message.createdAt).toLocaleTimeString(undefined, {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </motion.li>
  )
})
