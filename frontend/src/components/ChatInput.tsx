import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { motion } from 'framer-motion'
import { ArrowUpRight, Sparkles } from 'lucide-react'
import { FileUpload } from './FileUpload'

interface ChatInputProps {
  disabled?: boolean
  onSubmit: (message: string, file?: File) => void
  initialValue?: string
}

export function ChatInput({ disabled, onSubmit, initialValue }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (initialValue) {
      setValue(initialValue)
      textareaRef.current?.focus()
    }
  }, [initialValue])

  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [value])

  const send = () => {
    const trimmed = value.trim()
    if ((!trimmed && !selectedFile) || disabled) return
    onSubmit(trimmed, selectedFile || undefined)
    setValue('')
    setSelectedFile(null)
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    send()
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
          ref={textareaRef}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          placeholder="Ask me anything — I'll search the web, analyze documents, and stream sourced answers..."
          rows={1}
          className="chat-input__textarea"
          disabled={disabled}
        />
        <div className="chat-input__actions">
          <FileUpload
            selectedFile={selectedFile}
            onFileSelect={setSelectedFile}
            disabled={disabled}
          />
          <button
            type="button"
            className="chat-input__suggest"
            onClick={() => setValue('What are the key breakthroughs in AI research this year?')}
            disabled={disabled}
          >
            <Sparkles size={14} />
            Inspire me
          </button>
          <button
            type="submit"
            className="chat-input__submit"
            disabled={disabled || (value.trim().length === 0 && !selectedFile)}
          >
            Send
            <ArrowUpRight size={16} />
          </button>
        </div>
      </div>
    </motion.form>
  )
}
