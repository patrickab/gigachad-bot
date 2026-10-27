"use client"

import { Check, ChevronDown } from "lucide-react"
import { useRef, useState } from "react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { cn } from "@/lib/utils"

interface StyledSelectProps {
  options: { value: string; label: string }[]
  value: string
  onChange: (value: string) => void
  ariaLabel?: string
  /** "inline": the trigger is plain text styled by `className` (e.g. a label on a diagram node), and the list grows to fit. */
  variant?: "box" | "inline"
  className?: string
}

/** Custom listbox with arrow, Home, and End key navigation. Arrows change the value directly, so the list need not be open. */
export function StyledSelect({ options, value, onChange, ariaLabel = "Select option", variant = "box", className }: StyledSelectProps) {
  const inline = variant === "inline"
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const selected = options.find((option) => option.value === value)
  useClickOutside(ref, () => setOpen(false))

  const select = (next: string) => { onChange(next); setOpen(false) }
  const move = (direction: 1 | -1) => {
    if (options.length === 0) return
    const index = Math.max(0, options.findIndex((option) => option.value === value))
    select(options[(index + direction + options.length) % options.length].value)
  }

  const toggle = () => setOpen((current) => !current)
  const keyActions: Record<string, () => void> = {
    ArrowDown: () => open ? move(1) : setOpen(true),
    ArrowUp: () => open ? move(-1) : setOpen(true),
    ...options.length > 0 && { Home: () => select(options[0].value), End: () => select(options[options.length - 1].value) },
    Enter: toggle,
    " ": toggle,
  }

  return <div ref={ref} className={cn("relative", inline && "inline-block")}>
    <button type="button" aria-label={ariaLabel} aria-haspopup="listbox" aria-expanded={open} onClick={toggle} onKeyDown={(event) => {
      const action = keyActions[event.key]
      if (!action) return
      event.preventDefault()
      action()
    }} className={inline
      ? cn("cursor-pointer border-0 bg-transparent p-0 focus:outline-none", className)
      : "flex w-full items-center gap-2 rounded-lg border border-divider bg-paper px-2.5 py-2 text-left text-xs text-ink shadow-[var(--inner-highlight)] transition-colors hover:bg-surface-elevated focus:outline-none focus-visible:border-divider-strong"}>
      {inline ? selected?.label ?? value : <>
        <span className="min-w-0 flex-1 truncate">{selected?.label ?? value}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-ink-subtle transition-transform", open && "rotate-180")} />
      </>}
    </button>
    {open && <div role="listbox" aria-label={ariaLabel} className={cn("absolute z-50 mt-1 max-h-52 overflow-y-auto rounded-lg border border-divider bg-surface p-1 shadow-[var(--shadow-lg)]", inline ? "left-0 w-max" : "w-full")}>
      {options.map((option) => <button key={option.value} type="button" role="option" aria-selected={option.value === value} onClick={() => select(option.value)} className={cn("flex w-full items-center gap-2 rounded-md text-left transition-colors", inline ? "px-2 py-1 text-[11px]" : "px-2 py-1.5 text-xs", option.value === value ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface-elevated hover:text-ink")}>
        <span className="min-w-0 flex-1 truncate">{option.label}</span>
        {option.value === value && <Check className={cn("shrink-0", inline ? "h-3 w-3" : "h-3.5 w-3.5")} />}
      </button>)}
    </div>}
  </div>
}
