"use client"

import { useState } from "react"
import { Plus, Trash2 } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import type { Fact } from "@/lib/domain/memory"
import { EditableText } from "@/features/memory/EditableText"

/** A memory: its description and active facts, newest first, all editable in place. */
export function MemoryPage({ description, facts, onDescribe, onAdd, onUpdate, onRetire }: {
  description: string
  facts: Fact[]
  onDescribe: (description: string) => void
  onAdd: (text: string) => void
  onUpdate: (id: string, text: string) => void
  onRetire: (id: string) => void
}) {
  const [adding, setAdding] = useState(false)

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8">
      <EditableText value={description} label="Description" onCommit={onDescribe} className="mt-3 text-sm text-ink-muted" />
      <ul className="mt-4 list-disc space-y-1.5 pl-6 text-sm leading-6 text-ink marker:text-ink-subtle">
        {adding && (
          <li>
            <EditableText value="" label="New fact" multiline startEditing onCommit={onAdd} onDone={() => setAdding(false)} />
          </li>
        )}
        {facts.map((fact) => (
          <li key={fact.id} className="group/fact">
            <div className="flex items-start gap-2">
              <EditableText value={fact.text} label="Fact" multiline className="min-w-0 flex-1"
                onCommit={(text) => onUpdate(fact.id, text)} />
              <button onClick={() => onRetire(fact.id)} aria-label="Delete fact"
                className="shrink-0 rounded p-0.5 text-ink-subtle opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted group-hover/fact:opacity-100">
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          </li>
        ))}
      </ul>
      {!adding && (
        <button onClick={() => setAdding(true)}
          className="mt-3 flex items-center gap-1.5 rounded-md px-1 py-0.5 text-xs text-ink-muted transition-colors hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
          <Plus className="h-3.5 w-3.5" />Add
        </button>
      )}
    </div>
  )
}

/** A document, read-only, rendered as markdown. */
export function DocumentPage({ text }: { text: string }) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-8 pt-3">
      <LaTeXMarkdown content={text} />
    </div>
  )
}
