import { Compass, ListChecks, Sparkle } from 'lucide-react'

const prompts = [
  'What satellites contribute weather data to MOSDAC?',
  'How can I access archived oceanographic datasets?',
  'List recent updates from MOSDAC relevant to disaster management.',
  'Explain the calibration pipeline for INSAT imagery.',
]

export function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">
        <Sparkle size={28} />
      </div>
      <h2>Start the conversation</h2>
      <p>SatGraffin blends retrieval and generation to ground responses in MOSDAC&apos;s knowledge base. Try one of these prompts or craft your own.</p>
      <ul className="empty-state__prompts">
        {prompts.map((prompt) => (
          <li key={prompt}>
            <Compass size={16} />
            {prompt}
          </li>
        ))}
      </ul>
      <p className="empty-state__hint">
        <ListChecks size={14} /> Responses include source trails so you can verify the facts.
      </p>
    </div>
  )
}
