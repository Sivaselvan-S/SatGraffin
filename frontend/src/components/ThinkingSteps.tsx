import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ThinkingStep } from '../types';

interface ThinkingStepsProps {
  steps: ThinkingStep[];
  isStreaming?: boolean;
}

export const ThinkingSteps: React.FC<ThinkingStepsProps> = ({ steps, isStreaming }) => {
  const [isOpen, setIsOpen] = useState(true);

  if (!steps || steps.length === 0) return null;

  const activeStep = steps[steps.length - 1];

  return (
    <div className="thinking-steps">
      <div
        className="thinking-steps__header"
        onClick={() => setIsOpen(!isOpen)}
        role="button"
        aria-expanded={isOpen}
      >
        <div className="thinking-steps__title">
          {isStreaming ? (
            <span className="thinking-steps__pulse" aria-hidden />
          ) : (
            <span className="thinking-steps__done" aria-hidden>✓</span>
          )}
          <span>
            {isStreaming
              ? `Thinking: ${activeStep?.message || 'Processing...'}`
              : `Reasoning chain (${steps.length} steps)`}
          </span>
        </div>
        <span className="thinking-steps__toggle">
          {isOpen ? 'Collapse' : 'Expand'}
        </span>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.ul
            className="thinking-steps__list"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
          >
            {steps.map((s, idx) => (
              <motion.li
                key={idx}
                className="thinking-steps__item"
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.15, delay: idx * 0.05 }}
              >
                <span className="thinking-steps__arrow" aria-hidden>›</span>
                <span>{s.message}</span>
              </motion.li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
};
