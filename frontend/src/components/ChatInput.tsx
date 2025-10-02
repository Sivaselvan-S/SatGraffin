import { useState } from 'react'
import type { FormEvent } from 'react'
import { motion } from 'framer-motion'
import { ArrowUpRight, Sparkles } from 'lucide-react'

interface ChatInputProps {
  disabled?: boolean
  onSubmit: (message: string) => void
}

export function ChatInput({ disabled, onSubmit }: ChatInputProps) {
  const [value, setValue] = useState('')

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || disabled) {
      return
    }

    onSubmit(trimmed)
    setValue('')
  }

  return (
    <motion.form
      className="chat-input"
      onSubmit={handleSubmit}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="chat-input__wrapper">
        <textarea
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Ask me about satellites, payload specs, data access or anything MOSDAC offers..."
          rows={1}
          className="chat-input__textarea"
          disabled={disabled}
        />
        <div className="chat-input__actions">
          <button
            type="button"
            className="chat-input__suggest"
            onClick={() => setValue('Summarise the key missions hosted on MOSDAC and their primary objectives.')}
            disabled={disabled}
          >
            <Sparkles size={14} />
            Inspire me
          </button>
          <button type="submit" className="chat-input__submit" disabled={disabled || value.trim().length === 0}>
            Send
            <ArrowUpRight size={16} />
          </button>
        </div>
      </div>
    </motion.form>
  )
}
