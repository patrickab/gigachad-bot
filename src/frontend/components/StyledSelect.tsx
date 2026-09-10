"use client"

import { Check, ChevronDown } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"

interface StyledSelectProps {
  options: { value: string; label: string }[]
  value: string
  onChange: (value: string) => void
  ariaLabel?: string
}

export function StyledSelect({ options, value, onChange, ariaLabel = "Select option" }: StyledSelectProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const selected = options.find((option) => option.value === value)

  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", close)
    return () => document.removeEventListener("mousedown", close)
  }, [])

  const select = (next: string) => { onChange(next); setOpen(false) }
  const move = (direction: 1 | -1) => {
    if (options.length === 0) return
    const index = Math.max(0, options.findIndex((option) => option.value === value))
    select(options[(index + direction + options.length) % options.length].value)
  }

  return <div ref={ref} className="relative">
    <button type="button" aria-label={ariaLabel} aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((current) => !current)} onKeyDown={(event) => {
      if (event.key === "Escape") { setOpen(false); return }
      if (event.key === "ArrowDown") { event.preventDefault(); open ? move(1) : setOpen(true); return }
      if (event.key === "ArrowUp") { event.preventDefault(); open ? move(-1) : setOpen(true); return }
      if (event.key === "Home" && options[0]) { event.preventDefault(); select(options[0].value); return }
      if (event.key === "End" && options.at(-1)) { event.preventDefault(); select(options.at(-1)!.value); return }
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setOpen((current) => !current) }
    }} className="flex w-full items-center gap-2 rounded-lg border border-divider bg-paper px-2.5 py-2 text-left text-xs text-ink shadow-[var(--inner-highlight)] transition-colors hover:bg-surface-elevated focus:outline-none focus-visible:border-divider-strong">
      <span className="min-w-0 flex-1 truncate">{selected?.label ?? value}</span>
      <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-ink-subtle transition-transform", open && "rotate-180")} />
    </button>
    {open && <div role="listbox" aria-label={ariaLabel} className="absolute z-50 mt-1 max-h-52 w-full overflow-y-auto rounded-lg border border-divider bg-surface p-1 shadow-[var(--shadow-lg)]">
      {options.map((option) => <button key={option.value} type="button" role="option" aria-selected={option.value === value} onClick={() => select(option.value)} className={cn("flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors", option.value === value ? "bg-surface-elevated text-ink" : "text-ink-muted hover:bg-surface-elevated hover:text-ink")}>
        <span className="min-w-0 flex-1 truncate">{option.label}</span>
        {option.value === value && <Check className="h-3.5 w-3.5 shrink-0" />}
      </button>)}
    </div>}
  </div>
}
