"use client"

import { cn } from "@/lib/utils"

/**
 * An on/off switch. Off: a recessed track (paper under surface-elevated). On: a filled ink track with a paper knob,
 * readable in both themes.
 */
export function Switch({ label, checked, onToggle, disabled = false }: { label: string; checked: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-label={label}
      aria-checked={checked}
      disabled={disabled}
      onClick={onToggle}
      className={cn("relative h-5 w-9 shrink-0 rounded-full border transition-colors duration-200 ease-out focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-ink-muted disabled:pointer-events-none disabled:opacity-40 motion-reduce:transition-none",
        checked ? "border-ink-muted bg-ink-muted" : "border-divider bg-paper shadow-[var(--inner-highlight)]")}
    >
      <span className={cn("absolute left-0.5 top-0.5 h-3.5 w-3.5 rounded-full shadow-[var(--shadow-sm)] transition-transform duration-200 ease-out motion-reduce:transition-none",
        checked ? "translate-x-4 bg-paper" : "translate-x-0 bg-ink-faint")} />
    </button>
  )
}
