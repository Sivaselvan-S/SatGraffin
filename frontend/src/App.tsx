import { useEffect, useRef, useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { Header } from './components/Header'
import { ChatInput } from './components/ChatInput'
import { EmptyState } from './components/EmptyState'
import { MessageBubble } from './components/MessageBubble'
import { LoadingDots } from './components/LoadingDots'
import { StatusBar } from './components/StatusBar'
import { useChatHistory } from './hooks/useChatHistory'
import type { ChatMessage, QueryResponse } from './types'
import './styles/app.css'

const DEFAULT_ASSISTANT_MESSAGE: ChatMessage = {
  id: 'assistant-welcome',
  role: 'assistant',
  content:
    'Hello! I am **SatGraffin**, your AI research assistant. 🚀\n\nAsk me anything and I\'ll:\n- 🔍 Search the web in real-time\n- 📊 Analyze reliable sources\n- ✅ Provide accurate, well-sourced answers\n\nNo hallucinations — just facts backed by sources.',
  createdAt: Date.now(),
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'
const USER_ID_KEY = 'satgraffin.user.id'

function getOrCreateUserId(): string {
  if (typeof window === 'undefined') return `web-${Date.now()}`
  
  const stored = localStorage.getItem(USER_ID_KEY)
  if (stored) return stored
  
  const newId = typeof crypto !== 'undefined' && 'randomUUID' in crypto 
    ? crypto.randomUUID() 
    : `web-${Date.now()}`
  localStorage.setItem(USER_ID_KEY, newId)
  return newId
}

function createMessage(role: ChatMessage['role'], content: string, sources?: string[]): ChatMessage {
  const id = typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `${role}-${Date.now()}`
  return {
    id,
    role,
    content,
    createdAt: Date.now(),
    sources,
  }
}

function App() {
  const [status, setStatus] = useState<'idle' | 'connecting' | 'success' | 'error'>('idle')
  const [error, setError] = useState<string | undefined>()
  const [isLoading, setIsLoading] = useState(false)
  const [pendingPrompt, setPendingPrompt] = useState<string | undefined>()
  const [lastUserQuery, setLastUserQuery] = useState<string | undefined>()
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)

  const { messages, setMessages, clearHistory, hasHistory } = useChatHistory([DEFAULT_ASSISTANT_MESSAGE])
  const userId = getOrCreateUserId()

  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) {
      return
    }
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }, [messages, isLoading])

  const sendMessage = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) {
      return
    }

    setLastUserQuery(trimmed)
    const userMessage = createMessage('user', trimmed)
    setMessages((prev) => [...prev, userMessage])
    setPendingPrompt(undefined)

    setIsLoading(true)
    setStatus('connecting')
    setError(undefined)

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: trimmed, user_id: userId }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      const data = (await response.json()) as QueryResponse
      const assistantMessage = createMessage('assistant', data.response, data.source_links ?? [])
      assistantMessage.query = trimmed
      
      // Add disambiguation info if present
      if (data.is_ambiguous && data.disambiguation_options) {
        assistantMessage.isAmbiguous = true
        assistantMessage.disambiguationOptions = data.disambiguation_options
      }
      
      setMessages((prev) => [...prev, assistantMessage])
      setStatus('success')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error'
      setStatus('error')
      setError(message)

      const fallback = createMessage(
        'assistant',
        '⚠️ I ran into a connectivity issue while reaching the knowledge store.\n\nPlease check your connection and try again.',
      )
      fallback.isError = true
      setMessages((prev) => [...prev, fallback])
    } finally {
      setIsLoading(false)
    }
  }

  const handleRetry = () => {
    if (lastUserQuery) {
      // Remove the last error message
      setMessages((prev) => prev.slice(0, -1))
      sendMessage(lastUserQuery)
    }
  }

  const handlePromptClick = (prompt: string) => {
    setPendingPrompt(prompt)
  }

  const handleContextSelect = async (context: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/set-context`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId, selected_context: context }),
      })

      if (!response.ok) {
        console.error('Failed to set context preference')
        return
      }

      console.log(`Context preference set to: ${context}`)
    } catch (err) {
      console.error('Error setting context:', err)
    }
  }

  const handleClearHistory = () => {
    clearHistory()
    setStatus('idle')
    setError(undefined)
    setMessages([DEFAULT_ASSISTANT_MESSAGE])
  }

  const handleFeedback = async (message: ChatMessage, isThumbsUp: boolean) => {
    if (!message.query) return;
    try {
      await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: message.query,
          answer: message.content,
          source_links: message.sources || [],
          is_thumbs_up: isThumbsUp
        }),
      });
    } catch (err) {
      console.error('Failed to submit feedback', err);
    }
  };

  return (
    <div className="app-shell">
      <div className="app-shell__background" aria-hidden />
      <Header />
      <StatusBar status={status} message={error} />

      <section className="chat-panel">
        <div className="chat-panel__scroll" ref={scrollContainerRef}>
          {messages.length <= 1 && !isLoading ? (
            <EmptyState onPromptClick={handlePromptClick} />
          ) : (
            <ul className="chat-panel__messages">
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <MessageBubble 
                    key={message.id} 
                    message={message} 
                    onRetry={message.isError ? handleRetry : undefined}
                    onContextSelect={handleContextSelect}
                    onFeedback={handleFeedback}
                  />
                ))}
              </AnimatePresence>
              {isLoading && (
                <li className="chat-panel__loading">
                  <LoadingDots />
                  <span>Searching the web and analyzing sources…</span>
                </li>
              )}
            </ul>
          )}
        </div>

        {hasHistory && messages.length > 1 && (
          <div className="chat-panel__toolbar">
            <button
              type="button"
              onClick={handleClearHistory}
              disabled={isLoading}
              className="chat-panel__clear"
            >
              Clear conversation
            </button>
          </div>
        )}
      </section>

      <ChatInput onSubmit={sendMessage} disabled={isLoading} initialValue={pendingPrompt} />

      <footer className="app-footer">
        <p>
          Responses are grounded in real-time web search and analysis. All answers include source links for verification.
        </p>
      </footer>
    </div>
  )
}

export default App
