import { useEffect, useMemo, useRef, useState } from 'react'
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
    'Hello explorer! I am SatGraffin, your guide to the MOSDAC knowledge universe. Ask me about satellite missions, data access workflows, instrumentation specs, or anything space-data related.',
  createdAt: Date.now(),
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

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
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)

  const { messages, setMessages, clearHistory, hasHistory } = useChatHistory([DEFAULT_ASSISTANT_MESSAGE])
  const userId = useMemo(
    () => (typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `web-${Date.now()}`),
    [],
  )

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

    const userMessage = createMessage('user', trimmed)
    setMessages((prev) => [...prev, userMessage])

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
      setMessages((prev) => [...prev, assistantMessage])
      setStatus('success')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error'
      setStatus('error')
      setError(message)

      const fallback = createMessage(
        'assistant',
        'I ran into a connectivity issue while reaching the knowledge store. Please retry in a moment.',
      )
      fallback.isError = true
      setMessages((prev) => [...prev, fallback])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearHistory = () => {
    clearHistory()
    setStatus('idle')
    setError(undefined)
  }

  return (
    <div className="app-shell">
      <div className="app-shell__background" aria-hidden />
      <Header />
      <StatusBar status={status} message={error} />

      <section className="chat-panel">
        <div className="chat-panel__scroll" ref={scrollContainerRef}>
          {messages.length === 0 && !isLoading ? (
            <EmptyState />
          ) : (
            <ul className="chat-panel__messages">
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
              </AnimatePresence>
              {isLoading && (
                <li className="chat-panel__loading">
                  <LoadingDots />
                  <span>Consulting the SatGraffin graph…</span>
                </li>
              )}
            </ul>
          )}
        </div>

        {hasHistory && (
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

      <ChatInput onSubmit={sendMessage} disabled={isLoading} />

      <footer className="app-footer">
        <p>
          Responses are grounded in the curated SatGraffin vector store. For production use, set{' '}
          <code>VITE_API_BASE_URL</code> to your deployed backend URL.
        </p>
      </footer>
    </div>
  )
}

export default App
