"use client"

import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

const PILL_CLASS = "rounded-full px-3 py-1 text-[11px] font-medium transition-colors"
const PILL_FOCUS_CLASS = "focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

interface TabSemantics {
  /** Accessible name of the tablist. */
  label: string
  /** Tab ids are `<idPrefix>-<value>`, so the panel can point back with `aria-labelledby`. */
  idPrefix: string
  /** Id of the panel every tab controls. */
  panelId: string
}

/**
 * A row of rounded pill buttons choosing one of `options`.
 *
 * - With `tabs` the row is a `tablist` of focus-ringed `tab`s controlling one panel
 * - Without it the pills are plain buttons
 */
export function PillTabs<T extends string>({ options, value, onChange, tabs, className }: {
  options: readonly { value: T; label: ReactNode }[]
  value: T
  onChange: (value: T) => void
  tabs?: TabSemantics
  className?: string
}) {
  return (
    <div role={tabs && "tablist"} aria-label={tabs?.label} className={cn("flex gap-1", className)}>
      {options.map((option) => {
        const selected = option.value === value
        return (
          <button
            key={option.value}
            {...tabs && {
              id: `${tabs.idPrefix}-${option.value}`,
              type: "button" as const,
              role: "tab",
              "aria-selected": selected,
              "aria-controls": tabs.panelId,
            }}
            onClick={() => onChange(option.value)}
            className={cn(
              PILL_CLASS,
              tabs && PILL_FOCUS_CLASS,
              selected ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface/50 hover:text-ink",
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
