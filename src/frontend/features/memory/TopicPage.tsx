"use client"

import { useState } from "react"
import { Plus, Trash2 } from "lucide-react"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { EditableText } from "@/features/memory/EditableText"
import { PageHeader } from "@/features/memory/MemoryChrome"
import { formatTopicHeading, topicNameFromTitle } from "@/features/memory/memoryUtils"
import { addMemory, editMemory, mergeTopic, moveMemory, renameTopic, setTopicSummary, topicNames, type ScopeChange, type ScopeMemory, type TopicRow } from "@/features/memory/scopeEdits"
import { MEMORY_BUTTON_CLASS } from "@/features/memory/styles"

/** One topic: title, summary and facts, all editable in place. A title naming another topic offers to merge into it. */
export function TopicPage({ scope, row, onBack, onEdit, onOpen, onDeleteMemory, onDelete }: {
  scope: ScopeMemory
  row: TopicRow
  onBack: () => void
  onEdit: (change: ScopeChange) => void
  onOpen: (name: string) => void
  onDeleteMemory: (id: string) => void
  onDelete: () => void
}) {
  const [adding, setAdding] = useState(false)
  const [mergeTarget, setMergeTarget] = useState<string | null>(null)
  useEscapeLayer(onBack)
  const topic = row.topic.name
  const names = topicNames(scope)
  const topicOptions = names.map((name) => ({ value: name, label: formatTopicHeading(name) }))
  const lastUpdated = row.updatedAt
    ? new Date(row.updatedAt).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })
    : "Never"

  const editAndOpen = (change: ScopeChange, name: string) => {
    onEdit(change)
    onOpen(name)
  }

  const rename = (title: string) => {
    const name = topicNameFromTitle(title)
    if (!name || name === topic) return
    if (names.includes(name)) setMergeTarget(name)
    else editAndOpen(renameTopic(topic, name), name)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <PageHeader scope={scope} onBack={onBack} />
      <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
        <div className="flex items-start justify-between gap-4 pt-3">
          <EditableText value={formatTopicHeading(topic)} label="Topic title" onCommit={rename}
            className="min-w-0 flex-1 text-lg font-semibold text-ink" />
          <button onClick={onDelete} className={`shrink-0 ${MEMORY_BUTTON_CLASS}`}>Delete</button>
        </div>
        {mergeTarget && (
          <div className="mt-2 max-w-md">
            <InlineConfirm question={`Merge into ${formatTopicHeading(mergeTarget)}?`}
              onYes={() => editAndOpen(mergeTopic(topic, mergeTarget), mergeTarget)} onNo={() => setMergeTarget(null)} />
          </div>
        )}

        <h4 className="mt-6 text-xs text-ink-subtle">Last updated</h4>
        <p className="mt-1.5 text-sm text-ink">{lastUpdated}</p>

        <h4 className="mt-6 text-xs text-ink-subtle">Summary</h4>
        <EditableText value={row.topic.summary} label="Topic summary" multiline onCommit={(summary) => onEdit(setTopicSummary(topic, summary))}
          className="mt-1.5 text-sm text-ink" />

        <h4 className="mt-6 text-xs text-ink-subtle">Details</h4>
        <ul className="mt-1.5 list-disc space-y-1.5 pl-6 text-sm leading-6 text-ink marker:text-ink-subtle">
          {row.memories.map((memory) => (
            <li key={memory.id} className="group/memory">
              <div className="flex items-start gap-2">
                <EditableText value={memory.text} label="Memory" multiline className="min-w-0 flex-1"
                  onCommit={(text) => onEdit(editMemory(memory.id, text))} />
                <div className="flex shrink-0 items-center gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover/memory:opacity-100">
                  {topicOptions.length > 1 && (
                    <StyledSelect variant="inline" ariaLabel="Move memory to topic" value={topic} options={topicOptions}
                      onChange={(next) => onEdit(moveMemory(memory.id, next))} className="text-xs text-ink-subtle hover:text-ink" />
                  )}
                  <button onClick={() => onDeleteMemory(memory.id)} aria-label="Delete memory"
                    className="rounded p-0.5 text-ink-subtle hover:text-danger focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </li>
          ))}
          {adding && (
            <li>
              <EditableText value="" label="New memory" multiline startEditing onCommit={(text) => onEdit(addMemory(topic, text, scope.scope))} onDone={() => setAdding(false)} />
            </li>
          )}
        </ul>
        {!adding && (
          <button onClick={() => setAdding(true)}
            className="mt-3 flex items-center gap-1.5 rounded-md px-1 py-0.5 text-xs text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
            <Plus className="h-3.5 w-3.5" />Add
          </button>
        )}
      </div>
    </div>
  )
}
