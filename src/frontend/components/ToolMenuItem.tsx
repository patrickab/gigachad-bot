"use client"

import type { ReactNode } from "react"
import { type LucideIcon, Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface ToolMenuItemProps {
  icon: LucideIcon
  label: string
  active: boolean
  onClick: () => void
}

/** A selectable row in the composer's tool menu. Active state is a border plus a checkmark,
 *  never background tint alone, so it reads on a monochrome palette. */
export function ToolMenuItem({ icon: Icon, label, active, onClick }: ToolMenuItemProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "grid w-full grid-cols-[1rem_1fr_1rem] items-center gap-2 rounded-md border px-3 py-2 text-xs transition-colors",
        active
          ? "border-divider-strong bg-surface-elevated/60 text-ink"
          : "border-transparent text-ink hover:bg-surface-elevated/50"
      )}
    >
      <Icon className={cn("h-3.5 w-3.5", active ? "text-ink" : "text-ink-muted")} />
      <span className="text-center">{label}</span>
      <Check className={cn("h-3.5 w-3.5 justify-self-end", active ? "text-ink" : "text-transparent")} />
    </button>
  )
}

/** Groups menu rows under a small tracked-out label, with a divider above every group after
 *  the first so the list reads as sections rather than one flat stack. */
export function ToolMenuSection({ title, first, children }: { title: string; first?: boolean; children: ReactNode }) {
  return (
    <div className={cn(!first && "mt-1 border-t border-divider pt-1")}>
      <div className="px-3 pb-1 pt-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">{title}</div>
      {children}
    </div>
  )
}
