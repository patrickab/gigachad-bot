"use client"

import { useId, useState, type ReactNode } from "react"
import { ChevronRight, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

interface DisclosureProps {
  /** The row's content beside the chevron: a name, counts. Also the toggle's accessible name. */
  summary: ReactNode
  /** Drawn between the chevron and the summary, e.g. what kind of thing the row is. */
  icon?: LucideIcon
  defaultOpen?: boolean
  /** Classes of the row button, e.g. its ink. */
  summaryClassName?: string
  /** Controls at the row's right, outside its toggle, shown on hover or focus: e.g. a view switch. */
  actions?: ReactNode
  /** Rendered below the row whether open or not, above the content: a description toggle, a meta line. */
  between?: ReactNode
  /** Rendered below the row while open. Omitted, the row is static: no button, a blank where the chevron sits. */
  children?: ReactNode
  className?: string
}

const ROW = "flex h-7 w-full min-w-0 items-center gap-2 rounded text-left text-xs text-ink-muted"

/** The row with its `actions` beside it; their space is kept while hidden, so nothing shifts when they show. */
function WithActions({ row, actions }: { row: ReactNode; actions?: ReactNode }) {
  if (actions === undefined || actions === null || actions === false) return row
  return (
    <div className="group/row flex min-w-0 items-center gap-1">
      {row}
      <div className="flex shrink-0 items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover/row:opacity-100">{actions}</div>
    </div>
  )
}

/**
 * A row that opens and closes the content below it: a chevron that turns, `aria-expanded`, a keyboard focus ring.
 * The content mounts only while open, so what it loads waits until asked for.
 */
export function Disclosure({ summary, icon: Icon, actions, defaultOpen = false, summaryClassName, between, children, className }: DisclosureProps) {
  const [open, setOpen] = useState(defaultOpen)
  const contentId = useId()
  const icon = Icon && <Icon aria-hidden className="h-3 w-3 shrink-0 self-center" />
  if (children === undefined) {
    return (
      <div className={className}>
        <WithActions actions={actions} row={
          <div className={cn(ROW, summaryClassName)}>
            <span aria-hidden className="w-3 shrink-0" />
            {icon}
            {summary}
          </div>
        } />
        {between}
      </div>
    )
  }
  return (
    <div className={className}>
      <WithActions actions={actions} row={
        <button
          type="button"
          aria-expanded={open}
          aria-controls={open ? contentId : undefined}
          onClick={() => setOpen((shown) => !shown)}
          className={cn(ROW, "hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted", summaryClassName)}
        >
          <ChevronRight aria-hidden className={cn("h-3 w-3 shrink-0 self-center text-ink-faint transition-transform", open && "rotate-90")} />
          {icon}
          {summary}
        </button>
      } />
      {between}
      {open && <div id={contentId}>{children}</div>}
    </div>
  )
}
