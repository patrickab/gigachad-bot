"use client"

import { useState, useRef, type ElementType, type ReactNode } from "react"
import { Save } from "lucide-react"
import { cn } from "@/lib/utils"
import { Dialog } from "./Dialog"

interface SaveAsModalProps {
  open: boolean
  onClose: () => void
  /** Receives the trimmed name; blank names are ignored. */
  onSave: (name: string) => void
  title?: string
  icon?: ElementType
  placeholder?: string
  /** Rendered between the header and the input. */
  beforeInput?: ReactNode
  initialName?: string
  pending?: boolean
  error?: string
  submitLabel?: string
}

/** Modal with one text input; Enter submits the trimmed value. */
export function SaveAsModal({ open, onClose, onSave, title = "Save Chat", icon = Save, placeholder, beforeInput, initialName = "", pending = false, error, submitLabel }: SaveAsModalProps) {
  const [name, setName] = useState(initialName)
  const [wasOpen, setWasOpen] = useState(open)
  const inputRef = useRef<HTMLInputElement>(null)

  // Reset to the supplied name on each opening.
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) setName(initialName)
  }

  return (
    <Dialog open={open} onClose={onClose} icon={icon} title={title} initialFocusRef={inputRef}>
      {beforeInput}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (name.trim() && !pending) onSave(name.trim())
        }}
        className="px-5 pb-5"
      >
        <input
          ref={inputRef}
          type="text"
          aria-label={title}
          disabled={pending}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={placeholder}
          className={cn(
            "w-full rounded-lg border border-divider bg-surface/60 px-3 py-2.5 text-sm text-ink placeholder-ink-faint",
            "outline-none transition-all duration-200",
            "focus:border-ink-muted"
          )}
          autoComplete="off"
          spellCheck={false}
        />
        {error && <p role="alert" className="mt-2 text-xs text-danger">{error}</p>}
        {submitLabel && <button type="submit" disabled={pending || !name.trim()} className="mt-3 rounded-md px-3 py-1.5 text-sm text-ink hover:bg-hover disabled:opacity-40">{pending ? "Renaming…" : submitLabel}</button>}
      </form>
    </Dialog>
  )
}
