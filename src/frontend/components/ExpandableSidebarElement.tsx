"use client"

import { ElementType, ReactNode } from "react"
import { ChevronDown, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <div
      className={cn(
        "grid transition-[grid-template-rows] duration-200 ease-in-out",
        open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
      )}
    >
      <div className="overflow-hidden">{children}</div>
    </div>
  )
}

export interface ExpandableSidebarElementProps {
  icon: ElementType
  title: string
  badge?: ReactNode
  action?: ReactNode
  open: boolean
  onOpenChange: (open: boolean) => void
  children: ReactNode
}

export function ExpandableSidebarElement({ icon: Icon, title, badge, action, open, onOpenChange, children }: ExpandableSidebarElementProps) {
  return (
    <div className="border-b border-divider/50">
      <div className="flex items-center">
        <button
          onClick={() => onOpenChange(!open)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 px-3 py-2 text-ink hover:bg-surface/50 transition-colors rounded"
        >
          {open ? (
            <ChevronDown className="h-3 w-3 text-ink-subtle shrink-0" />
          ) : (
            <ChevronRight className="h-3 w-3 text-ink-subtle shrink-0" />
          )}
          <Icon className="h-3.5 w-3.5 text-ink-muted shrink-0" />
          <span className="text-[11px] font-medium truncate flex-1 text-left">{title}</span>
          {badge != null && <span className="text-[10px] text-ink-faint shrink-0">{badge}</span>}
        </button>
        {action != null && <div className="shrink-0 pr-1.5">{action}</div>}
      </div>
      <Collapse open={open}>{children}</Collapse>
    </div>
  )
}
