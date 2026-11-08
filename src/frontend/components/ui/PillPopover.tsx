"use client"

import { AnimatePresence, motion } from "framer-motion"
import { useCallback, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from "react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { cn } from "@/lib/utils"

/** Raised card look shared by the panel and anything shown beside it. */
export const surfaceClass = "rounded-xl border border-divider bg-surface shadow-[var(--shadow-lg)]"

interface PillPopoverState {
  open: boolean
  show: () => void
  close: () => void
  rootRef: RefObject<HTMLDivElement | null>
  triggerRef: RefObject<HTMLButtonElement | null>
  panelRef: RefObject<HTMLDivElement | null>
}

/** Open state of a `PillPopover`: a press outside or Escape closes it, closing puts focus inside the panel back on the trigger, then runs `onClose`. */
export function usePillPopover(onClose: () => void): PillPopoverState {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const close = useCallback(() => {
    // Focus inside the closing panel goes back to the trigger instead of the page
    if (panelRef.current?.contains(document.activeElement)) triggerRef.current?.focus()
    setOpen(false)
    onClose()
  }, [onClose])
  const show = useCallback(() => setOpen(true), [])
  useClickOutside(rootRef, close)
  return { open, show, close, rootRef, triggerRef, panelRef }
}

/** Pill trigger toggling an animated panel above or below it, `children` fill the panel.
 *
 * Attributes
 * ----------
 * popover : PillPopoverState
 *     From `usePillPopover`, owned by the caller so its content can close the panel.
 * triggerLabel : string
 *     Accessible name of the trigger, e.g. `Model: Claude Opus 5`.
 * label : string
 *     Accessible name of the panel.
 * panelClassName : string | undefined
 *     Extra panel classes, e.g. a width.
 * onKeyDown : ((event: KeyboardEvent<HTMLDivElement>) => void) | undefined
 *     Keys pressed on the trigger or inside the panel.
 * onAnimationComplete : (() => void) | undefined
 *     Runs whenever the panel finishes animating in or out.
 */
export function PillPopover({
  popover, triggerLabel, triggerContent, triggerClassName, label, placement = "above", panelClassName, onKeyDown, onAnimationComplete, children,
}: {
  popover: PillPopoverState
  triggerLabel: string
  triggerContent: ReactNode
  triggerClassName?: string
  label: string
  placement?: "above" | "below"
  panelClassName?: string
  onKeyDown?: (event: KeyboardEvent<HTMLDivElement>) => void
  onAnimationComplete?: () => void
  children: ReactNode
}) {
  const { open, show, close, rootRef, triggerRef, panelRef } = popover
  return <div ref={rootRef} onKeyDown={onKeyDown} className="relative min-w-0">
    <button
      ref={triggerRef}
      type="button"
      onClick={() => open ? close() : show()}
      aria-label={triggerLabel}
      aria-haspopup="dialog"
      aria-expanded={open}
      className={cn("flex h-7 min-w-0 items-center gap-1.5 rounded-full px-3 text-xs text-ink-muted transition-colors hover:bg-hover hover:text-ink", open && "text-ink", triggerClassName)}
    >
      {triggerContent}
    </button>
    <AnimatePresence>{open && <motion.div
      ref={panelRef}
      role="dialog"
      aria-label={label}
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 4 }}
      transition={{ duration: 0.15 }}
      onAnimationComplete={onAnimationComplete}
      className={cn(
        "absolute left-0 z-50 flex w-64 flex-col",
        surfaceClass,
        placement === "below" ? "top-full mt-2" : "bottom-full mb-2",
        panelClassName,
      )}
    >{children}</motion.div>}</AnimatePresence>
  </div>
}
