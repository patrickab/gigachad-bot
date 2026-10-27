"use client"

import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { PageHeader, Spinner } from "@/features/memory/MemoryChrome"
import { formatTopicHeading, normalizeMemoryTopic } from "@/features/memory/memoryUtils"
import { topicRows, type ScopeContent, type ScopeMemory } from "@/features/memory/scopeEdits"
import { MEMORY_BUTTON_CLASS, MEMORY_TEXT_BUTTON_CLASS } from "@/features/memory/styles"

/** Proposed regrouping of one scope: facts that change topic name their previous one. */
export function RegroupPage({ scope, proposal, onBack, onApply }: {
  scope: ScopeMemory
  proposal: ScopeContent | null
  onBack: () => void
  onApply: (proposal: ScopeContent) => void
}) {
  useEscapeLayer(onBack)
  const previousTopic = new Map(scope.memories.map((m) => [m.id, normalizeMemoryTopic(m)]))

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader scope={scope} onBack={onBack} />
      <div className="flex shrink-0 items-start justify-between gap-4 px-6 pt-3">
        <h2 className="text-lg font-semibold text-ink">Regroup topics</h2>
        <div className="flex items-center gap-2">
          <button onClick={onBack} className={MEMORY_TEXT_BUTTON_CLASS}>Cancel</button>
          <button onClick={() => proposal && onApply(proposal)} disabled={!proposal} className={`shrink-0 ${MEMORY_BUTTON_CLASS}`}>Apply</button>
        </div>
      </div>
      {proposal ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
          {topicRows(proposal).map(({ topic, memories }) => (
            <section key={topic.name} className="mt-6">
              <h3 className="text-sm font-semibold text-ink">{formatTopicHeading(topic.name)}</h3>
              {topic.summary && <p className="mt-0.5 text-sm text-ink-muted">{topic.summary}</p>}
              <ul className="mt-1.5 list-disc space-y-1 pl-6 text-sm leading-6 text-ink marker:text-ink-subtle">
                {memories.map((memory) => {
                  const from = previousTopic.get(memory.id)
                  return (
                    <li key={memory.id}>
                      {memory.text}
                      {from && from !== topic.name && <span className="text-ink-subtle"> · from {formatTopicHeading(from)}</span>}
                    </li>
                  )
                })}
              </ul>
            </section>
          ))}
        </div>
      ) : <Spinner label="Regrouping" />}
    </div>
  )
}
