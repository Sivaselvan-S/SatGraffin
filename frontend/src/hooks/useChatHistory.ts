import { useEffect, useState } from 'react'
import type { ChatMessage } from '../types'

const STORAGE_KEY = 'satgraffin.chat.history.v1'

export function useChatHistory(initialMessages: ChatMessage[]) {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (typeof window === 'undefined') {
      return initialMessages
    }

    try {
      const raw = window.localStorage.getItem(STORAGE_KEY)
      if (!raw) {
        return initialMessages
      }

      const parsed = JSON.parse(raw) as ChatMessage[]
      if (!Array.isArray(parsed) || parsed.length === 0) {
        return initialMessages
      }

      return parsed
    } catch (error) {
      console.error('Failed to parse cached chat history', error)
      return initialMessages
    }
  })

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(messages))
    } catch (error) {
      console.warn('Unable to persist chat history', error)
    }
  }, [messages])

  const clearHistory = () => {
    setMessages([])
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(STORAGE_KEY)
    }
  }

  return {
    messages,
    setMessages,
    clearHistory,
    hasHistory: messages.length > 0,
  }
}
