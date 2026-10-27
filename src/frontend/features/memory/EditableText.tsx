"use client"

import { useCallback, useLayoutEffect, useRef, useState } from "react"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { cn } from "@/lib/utils"

/**
 * Text that turns into an input on click: Enter saves, Shift+Enter adds a line when `multiline`, Esc cancels.
 *
 * - `onCommit` runs only for a changed, non-empty value, a returned string is shown as the error and keeps editing
 * - `onDone` runs whenever editing ends, `startEditing` opens it in edit mode (new entries)
 */
export function EditableText({ value, label, multiline = false, startEditing = false, className, onCommit, onDone }: {
  value: string
  label: string
  multiline?: boolean
  startEditing?: boolean
  className?: string
  onCommit: (next: string) => string | void
  onDone?: () => void
}) {
  const [editing, setEditing] = useState(startEditing)
  const [draft, setDraft] = useState(value)
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Grow with wrapped lines, not only explicit newlines.
  useLayoutEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = "auto"
    textarea.style.height = `${textarea.scrollHeight}px`
  }, [draft, editing])

  const stop = useCallback(() => {
    setEditing(false)
    setError(null)
    onDone?.()
  }, [onDone])
  useEscapeLayer(stop, editing)

  const commit = () => {
    const next = draft.trim()
    if (next && next !== value) {
      const failure = onCommit(next)
      if (failure) { setError(failure); return }
    }
    stop()
  }

  if (!editing) {
    return (
      <button onClick={() => { setDraft(value); setEditing(true) }} aria-label={`Edit ${label}`}
        className={cn("block w-full whitespace-pre-wrap rounded text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted", className)}>
        {value || <span className="text-ink-faint">Empty</span>}
      </button>
    )
  }

  return (
    <div className={className}>
      <textarea
        ref={textareaRef} autoFocus aria-label={label} value={draft} rows={1}
        onChange={(e) => { setDraft(e.target.value); setError(null) }}
        onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !(multiline && e.shiftKey)) { e.preventDefault(); commit() }
        }}
        className="block w-full resize-none overflow-hidden rounded bg-transparent outline-none ring-1 ring-divider focus:ring-divider-strong"
      />
      {error && <p role="alert" className="mt-1 text-xs text-danger">{error}</p>}
    </div>
  )
}
