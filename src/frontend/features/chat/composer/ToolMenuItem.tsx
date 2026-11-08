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

/** A toggle row in the composer's tool menu: icon and label on the left, a quiet check on the right when on.
 *  The check, not a tint, carries the state, so it reads on a monochrome palette. */
export function ToolMenuItem({ icon: Icon, label, active, onClick }: ToolMenuItemProps) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-xs transition-colors hover:bg-hover focus-visible:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:-outline-offset-1 focus-visible:outline-ink-muted"
    >
      <Icon className={cn("h-3.5 w-3.5 shrink-0 transition-colors", active ? "text-ink" : "text-ink-faint")} />
      <span className={cn("min-w-0 flex-1 truncate transition-colors", active ? "text-ink" : "text-ink-muted")}>{label}</span>
      <Check className={cn("h-3.5 w-3.5 shrink-0 text-ink-muted transition-opacity duration-150 ease-out", active ? "opacity-100" : "opacity-0")} />
    </button>
  )
}

/** Groups menu rows under a small tracked-out label. */
export function ToolMenuSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <div className="px-2.5 pb-1 pt-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">{title}</div>
      {children}
    </div>
  )
}
