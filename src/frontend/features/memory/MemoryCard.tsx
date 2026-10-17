"use client"

import { useCallback, useState } from "react"
import { Check, FolderOpen, Globe, Trash2, X } from "lucide-react"
import { type MemoryRecord, memoryText } from "@/features/memory/memoryUtils"
import type { MemoryStatus } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

const STATUS_STYLES: Record<MemoryStatus, string> = {
  "new": "bg-emerald-500/15 text-emerald-400",
  "combined": "bg-amber-500/15 text-amber-400",
  "pre-existing": "bg-surface-elevated text-ink-subtle",
}

export function AddMemoryCard({ scope, disabled = false, onAddMemory, onCancel }: {
  scope: "global" | "project"
  disabled?: boolean
  onAddMemory: (scope: "global" | "project", memory: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState("")
  const ScopeIcon = scope === "global" ? Globe : FolderOpen
  const scopeLabel = scope === "global" ? "Global" : "Project"

  function submit() {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onAddMemory(scope, trimmed)
    setValue("")
    onCancel()
  }

  return (
    <div className="rounded-lg border border-divider bg-surface p-4">
      <div className="mb-3 flex items-center gap-2">
        <ScopeIcon className="h-3.5 w-3.5 text-ink-muted" />
        <span className="text-xs font-medium text-ink-muted">{scopeLabel}</span>
        <span className="rounded-full bg-surface-elevated px-2 py-0.5 text-[10px] text-ink-subtle">manual</span>
      </div>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault()
            submit()
          }
        }}
        disabled={disabled}
        rows={3}
        autoFocus
        className="w-full resize-none bg-transparent text-sm leading-6 text-ink outline-none placeholder:text-ink-faint disabled:opacity-50"
        placeholder="Write a memory to incorporate..."
      />
      <div className="mt-3 flex justify-end gap-2">
        <button onClick={onCancel} disabled={disabled}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
          <X className="h-3 w-3" />Deny
        </button>
        <button onClick={submit} disabled={disabled || !value.trim()}
          className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
          <Check className="h-3 w-3" />Accept
        </button>
      </div>
    </div>
  )
}

export function MemoryErrorCard({ message }: { message: string }) {
  return (
    <div className="flex flex-1 items-center justify-center p-8 min-h-0">
      <div className="w-full max-w-lg rounded-xl border border-divider bg-surface p-5">
        <div className="text-sm font-medium text-ink">Memory error</div>
        <div className="mt-2 whitespace-pre-wrap text-xs leading-5 text-ink-muted">{message}</div>
      </div>
    </div>
  )
}

interface MemoryCardProps {
  memory: MemoryRecord
  disabled?: boolean
  mode?: "review" | "display"
  onChange?: (memoryId: string, text: string) => void
  onAccept?: (memoryId: string) => void
  onCancel?: (memoryId: string) => void
  onRemove?: (memoryId: string) => void
}

/** Memory card, click to edit. `review` shows accept/deny, `display` shows hover-to-remove. */
export function MemoryCard({
  memory, disabled = false, mode = "review",
  onChange, onAccept, onCancel, onRemove,
}: MemoryCardProps) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState("")
  useEscapeLayer(() => setEditing(false), editing)
  const status: MemoryStatus | undefined = "status" in memory ? memory.status : undefined

  const handleEdit = useCallback(() => {
    if (disabled) return
    setDraft(memoryText(memory))
    setEditing(true)
  }, [memory, disabled])

  const handleSave = useCallback(() => {
    const trimmed = draft.trim()
    if (trimmed && trimmed !== memoryText(memory)) {
      onChange?.(memory.id, trimmed)
    }
    setEditing(false)
  }, [draft, memory, onChange])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); handleSave() }
  }, [handleSave])

  return (
    <div className="p-3 group/card relative">
      {mode === "review" && (
        <div className="flex items-center gap-2 mb-2">
          {status && (
            <span className={cn("rounded-full px-2 py-0.5 text-[10px]", STATUS_STYLES[status])}>{status}</span>
          )}
        </div>
      )}

      {editing ? (
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleSave}
          rows={Math.max(2, draft.split("\n").length)}
          autoFocus
          disabled={disabled}
          className="w-full resize-none bg-transparent text-sm leading-6 text-ink outline-none placeholder:text-ink-faint disabled:opacity-50 cursor-text"
        />
      ) : (
        <div
          onClick={handleEdit}
          onMouseDown={(e) => e.stopPropagation()}
          className="text-sm leading-6 text-ink whitespace-pre-wrap select-text cursor-text"
        >
          {memoryText(memory)}
        </div>
      )}

      {mode === "display" && onRemove && !editing && (
        <button aria-label={`Remove memory ${memory.id}`}
          onClick={(e) => { e.stopPropagation(); onRemove(memory.id) }}
          disabled={disabled}
          className="absolute bottom-1.5 right-1.5 rounded p-0.5 text-ink-faint hover:text-danger opacity-0 group-hover/card:opacity-100 transition-opacity disabled:opacity-40 cursor-pointer"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      )}

      {mode === "review" && onAccept && onCancel && (
        <div className="mt-3 flex justify-end gap-2">
          <button onClick={() => onCancel(memory.id)} disabled={disabled}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
            <X className="h-3 w-3" />Deny
          </button>
          <button onClick={() => onAccept(memory.id)} disabled={disabled}
            className="flex items-center gap-1.5 text-xs text-ink-muted hover:text-ink disabled:opacity-40 transition-colors">
            <Check className="h-3 w-3" />Accept
          </button>
        </div>
      )}
    </div>
  )
}
