import { useEffect, useRef, useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { Header } from './components/Header'
import { ChatInput } from './components/ChatInput'
import { EmptyState } from './components/EmptyState'
import { MessageBubble } from './components/MessageBubble'
import { LoadingDots } from './components/LoadingDots'
import { StatusBar } from './components/StatusBar'
import { useChatHistory } from './hooks/useChatHistory'
import { useStreamingQuery } from './hooks/useStreamingQuery'
import type { ChatMessage, ThinkingStep } from './types'
import './styles/app.css'

const DEFAULT_ASSISTANT_MESSAGE: ChatMessage = {
  id: 'assistant-welcome',
  role: 'assistant',
  content:
    'Hello! I am **SatGraffin v3.0**, your Industrial Agentic RAG assistant. 🚀\n\nFeatures enabled:\n- ⚡ **Real-time SSE Streaming** response\n- 🔍 **Sub-query decomposition** & parallel web retrieval\n- 🎯 **HyDE** & **CRAG** relevance grading\n- 🖼️ **Multimodal image & document reasoning**',
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
  const [selectedModel, setSelectedModel] = useState<string>(() => {
    const stored = localStorage.getItem('satgraffin.selected_model')
    return (stored && stored !== 'gemini-1.5-flash') ? stored : 'gemini-2.0-flash'
  })
  const scrollContainerRef = useRef<HTMLDivElement | null>(null)

  const { messages, setMessages, clearHistory, hasHistory } = useChatHistory([DEFAULT_ASSISTANT_MESSAGE])
  const { streamQuery } = useStreamingQuery()
  const userId = getOrCreateUserId()

  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }, [messages, isLoading])

  const sendMessage = async (text: string, file?: File) => {
    const trimmed = text.trim()
    if (!trimmed && !file) return

    const queryText = trimmed || (file ? `Analyze uploaded file: ${file.name}` : '')
    setLastUserQuery(queryText)
    
    const userMessage = createMessage('user', queryText)
    if (file) {
      userMessage.imageUrl = URL.createObjectURL(file)
    }

    const assistantId = typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `assistant-${Date.now()}`
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      query: queryText,
      thinkingSteps: [],
      isStreaming: true,
    }

    setMessages((prev) => [...prev, userMessage, assistantMessage])
    setPendingPrompt(undefined)
    setIsLoading(true)
    setStatus('connecting')
    setError(undefined)

    await streamQuery(
      queryText,
      {
        apiBaseUrl: API_BASE_URL,
        userId,
        model: selectedModel,
        onThinkingStep: (step: ThinkingStep) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id === assistantId) {
                const currentSteps = msg.thinkingSteps || []
                return {
                  ...msg,
                  thinkingSteps: [...currentSteps, step],
                }
              }
              return msg
            })
          )
        },
        onSources: (sources: string[]) => {
          setMessages((prev) =>
            prev.map((msg) => (msg.id === assistantId ? { ...msg, sources } : msg))
          )
        },
        onToken: (token: string) => {
          setStatus('success')
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: msg.content + token }
                : msg
            )
          )
        },
        onError: (errMessage: string) => {
          setStatus('error')
          setError(errMessage)
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: '⚠️ I ran into a connection error while processing your request. Please try again.',
                    isError: true,
                    isStreaming: false,
                  }
                : msg
            )
          )
        },
        onDone: () => {
          setIsLoading(false)
          setStatus('idle')
          setMessages((prev) =>
            prev.map((msg) => (msg.id === assistantId ? { ...msg, isStreaming: false } : msg))
          )
        },
      },
      file
    )
  }

  const handleRetry = () => {
    if (lastUserQuery) {
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, selected_context: context }),
      })
      if (!response.ok) {
        console.error('Failed to set context preference')
        return
      }
      if (lastUserQuery) {
        sendMessage(lastUserQuery)
      }
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
    if (!message.query) return
    try {
      await fetch(`${API_BASE_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: message.query,
          answer: message.content,
          source_links: message.sources || [],
          is_thumbs_up: isThumbsUp,
        }),
      })
    } catch (err) {
      console.error('Failed to submit feedback', err)
    }
  }

  return (
    <div className="app-shell">
      <div className="app-shell__background" aria-hidden />
      <Header
        apiBaseUrl={API_BASE_URL}
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        disabled={isLoading}
      />
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
                  <span>Processing sub-queries and synthesizing response…</span>
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
          SatGraffin v3.0 Industrial Agentic RAG. Answers are grounded in real-time multi-lane retrieval and streaming synthesis.
        </p>
      </footer>
    </div>
  )
}

export default App
