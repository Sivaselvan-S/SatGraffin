import { memo, useState } from 'react'
import { motion } from 'framer-motion'
import { ChevronDown, Check, Lightbulb } from 'lucide-react'
import clsx from 'clsx'

interface ContextSelectorProps {
  options: string[]
  onSelect: (option: string) => void
  disabled?: boolean
}

export const ContextSelector = memo(function ContextSelector({
  options,
  onSelect,
  disabled = false,
}: ContextSelectorProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedOption, setSelectedOption] = useState<string | null>(null)

  const handleSelect = (option: string) => {
    setSelectedOption(option)
    setIsOpen(false)
    onSelect(option)
  }

  if (options.length === 0) return null

  return (
    <motion.div
      className="context-selector"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.2 }}
    >
      <div className="context-selector__header">
        <Lightbulb size={16} className="context-selector__icon" />
        <span className="context-selector__label">
          {selectedOption 
            ? `Context set to: ${selectedOption}` 
            : 'This topic has multiple meanings. Select your context:'}
        </span>
      </div>

      {!selectedOption && (
        <div className="context-selector__dropdown-wrapper">
          <button
            type="button"
            className={clsx(
              'context-selector__trigger',
              isOpen && 'context-selector__trigger--open',
              disabled && 'context-selector__trigger--disabled'
            )}
            onClick={() => !disabled && setIsOpen(!isOpen)}
            disabled={disabled}
          >
            <span>Choose a context...</span>
            <ChevronDown
              size={16}
              className={clsx(
                'context-selector__chevron',
                isOpen && 'context-selector__chevron--open'
              )}
            />
          </button>

          {isOpen && (
            <motion.ul
              className="context-selector__menu"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
            >
              {options.map((option, index) => (
                <li key={index}>
                  <button
                    type="button"
                    className="context-selector__option"
                    onClick={() => handleSelect(option)}
                  >
                    <span>{option}</span>
                  </button>
                </li>
              ))}
            </motion.ul>
          )}
        </div>
      )}

      {selectedOption && (
        <motion.div
          className="context-selector__selected"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
        >
          <Check size={14} className="context-selector__check" />
          <span>Future questions will be answered in this context</span>
        </motion.div>
      )}
    </motion.div>
  )
})
