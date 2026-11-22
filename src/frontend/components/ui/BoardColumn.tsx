"use client"

import type { HTMLAttributes, ReactNode } from "react"
import { AnimatePresence } from "framer-motion"
import { ElevatedContainer } from "@/components/ui/ElevatedContainer"
import { cn } from "@/lib/utils"

interface BoardColumnProps extends Omit<HTMLAttributes<HTMLDivElement>, "title" | "children"> {
  title: string
  /** `h3` announces the column title as a heading. */
  titleAs?: "span" | "h3"
  count: number
  /** Shown before the title, such as a status dot. */
  marker?: ReactNode
  /** Header control on the right, such as an add button. */
  action?: ReactNode
  /** Classes and handlers (drop targets) of the scrolling card list. */
  bodyProps?: HTMLAttributes<HTMLDivElement>
  /** Cards, animated in and out together. */
  children: ReactNode
  /** Rendered below the cards, outside their animation. */
  footer?: ReactNode
}

/** Kanban column: an elevated frame with a quiet title, its count (hidden and the title dimmed when empty) and an optional action over a scrolling card list. */
export function BoardColumn({ title, titleAs: Title = "span", count, marker, action, bodyProps, children, footer, className, ...frameProps }: BoardColumnProps) {
  return (
    <ElevatedContainer {...frameProps} className={cn("flex flex-1 flex-col min-h-0 overflow-hidden rounded-2xl border", className)}>
      <div className="flex items-center justify-between px-4 pt-4 pb-3 select-none shrink-0">
        <div className="flex items-center gap-2">
          {marker}
          <Title className={cn("text-xs font-medium", count === 0 ? "text-ink-faint" : "text-ink-muted")}>{title}</Title>
          {count > 0 && <span className="text-[11px] text-ink-faint tabular-nums">{count}</span>}
        </div>
        {action}
      </div>
      <div {...bodyProps} className={cn("overflow-y-auto px-4 pb-4 scrollbar-thin", bodyProps?.className)}>
        <AnimatePresence mode="popLayout" initial={false}>
          {children}
        </AnimatePresence>
        {footer}
      </div>
    </ElevatedContainer>
  )
}
