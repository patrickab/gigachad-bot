"use client"

import { ArrowDown, ArrowUp } from "lucide-react"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { PageHeader } from "@/features/memory/MemoryChrome"
import { formatTopicHeading, normalizeMemoryTopic } from "@/features/memory/memoryUtils"
import { shiftMemory, type ScopeChange, type ScopeMemory } from "@/features/memory/scopeEdits"

const ICON_BUTTON_CLASS = "rounded p-0.5 text-ink-subtle transition-colors hover:text-ink disabled:opacity-30 disabled:hover:text-ink-subtle focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** Every fact of a scope in relevance order, the order they reach the model. */
export function RankingPage({ scope, onBack, onEdit }: {
  scope: ScopeMemory
  onBack: () => void
  onEdit: (change: ScopeChange) => void
}) {
  useEscapeLayer(onBack)
  const last = scope.memories.length - 1
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader scope={scope} onBack={onBack} />
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
        <h3 className="pt-3 text-lg font-semibold text-ink">Relevance</h3>
        <ol className="mt-3 space-y-1.5 text-sm leading-6 text-ink">
          {scope.memories.map((memory, rank) => (
            <li key={memory.id} className="group/memory flex items-start gap-3">
              <span className="w-6 shrink-0 text-right text-xs leading-6 tabular-nums text-ink-subtle">{rank + 1}</span>
              <div className="min-w-0 flex-1">
                <span className="text-xs text-ink-subtle">{formatTopicHeading(normalizeMemoryTopic(memory))}</span>
                <p>{memory.text}</p>
              </div>
              <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover/memory:opacity-100">
                <button onClick={() => onEdit(shiftMemory(memory.id, -1))} disabled={rank === 0} aria-label="Move up" className={ICON_BUTTON_CLASS}>
                  <ArrowUp className="h-3.5 w-3.5" />
                </button>
                <button onClick={() => onEdit(shiftMemory(memory.id, 1))} disabled={rank === last} aria-label="Move down" className={ICON_BUTTON_CLASS}>
                  <ArrowDown className="h-3.5 w-3.5" />
                </button>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}
