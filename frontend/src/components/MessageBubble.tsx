import { memo, useState } from 'react'
import { motion } from 'framer-motion'
import { Bot, UserRound, Copy, Check, RotateCcw, ThumbsUp, ThumbsDown } from 'lucide-react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage } from '../types'
import { SourceLinks } from './SourceLinks'
import { ContextSelector } from './ContextSelector'

interface MessageBubbleProps {
  message: ChatMessage
  onRetry?: (content: string) => void
  onContextSelect?: (context: string) => void
  onFeedback?: (message: ChatMessage, isThumbsUp: boolean) => void
}

export const MessageBubble = memo(function MessageBubble({ 
  message, 
  onRetry,
  onContextSelect,
  onFeedback
}: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)
  const [feedbackState, setFeedbackState] = useState<'up' | 'down' | null>(null)

  const handleFeedbackClick = (isUp: boolean) => {
    if (feedbackState) return;
    setFeedbackState(isUp ? 'up' : 'down');
    onFeedback?.(message, isUp);
  }

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

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
        <div className="message__content">
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Custom styling for markdown elements
                p: ({ children }) => <p className="markdown-p">{children}</p>,
                ul: ({ children }) => <ul className="markdown-ul">{children}</ul>,
                ol: ({ children }) => <ol className="markdown-ol">{children}</ol>,
                li: ({ children }) => <li className="markdown-li">{children}</li>,
                code: ({ className, children, ...props }) => {
                  const isInline = !className
                  return isInline ? (
                    <code className="markdown-code-inline" {...props}>{children}</code>
                  ) : (
                    <code className={clsx('markdown-code-block', className)} {...props}>{children}</code>
                  )
                },
                pre: ({ children }) => <pre className="markdown-pre">{children}</pre>,
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="markdown-link">
                    {children}
                  </a>
                ),
                strong: ({ children }) => <strong className="markdown-strong">{children}</strong>,
                h1: ({ children }) => <h3 className="markdown-heading">{children}</h3>,
                h2: ({ children }) => <h4 className="markdown-heading">{children}</h4>,
                h3: ({ children }) => <h5 className="markdown-heading">{children}</h5>,
                blockquote: ({ children }) => <blockquote className="markdown-blockquote">{children}</blockquote>,
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>
        
        {/* Action buttons for assistant messages */}
        {!isUser && (
          <div className="message__actions">
            <button
              type="button"
              className="message__action-btn"
              onClick={handleCopy}
              title={copied ? 'Copied!' : 'Copy response'}
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
            
            {message.query && !message.isError && (
               <div style={{ display: 'flex', gap: '4px' }}>
                 <button
                   type="button"
                   className="message__action-btn"
                   style={{ color: feedbackState === 'up' ? '#10b981' : undefined }}
                   onClick={() => handleFeedbackClick(true)}
                   disabled={feedbackState !== null}
                   title="Good answer"
                 >
                   <ThumbsUp size={14} />
                 </button>
                 <button
                   type="button"
                   className="message__action-btn"
                   style={{ color: feedbackState === 'down' ? '#ef4444' : undefined }}
                   onClick={() => handleFeedbackClick(false)}
                   disabled={feedbackState !== null}
                   title="Bad answer"
                 >
                   <ThumbsDown size={14} />
                 </button>
               </div>
            )}

            {message.isError && onRetry && (
              <button
                type="button"
                className="message__action-btn message__action-btn--retry"
                onClick={() => onRetry(message.content)}
                title="Retry this query"
              >
                <RotateCcw size={14} />
                Retry
              </button>
            )}
          </div>
        )}
        
        {message.sources && message.sources.length > 0 && (
          <SourceLinks links={message.sources} />
        )}

        {/* Context Selector for ambiguous responses */}
        {!isUser && message.isAmbiguous && message.disambiguationOptions && message.disambiguationOptions.length > 0 && (
          <ContextSelector
            options={message.disambiguationOptions}
            onSelect={(context) => onContextSelect?.(context)}
          />
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
