"use client"

import { useCallback, useRef, useState, type ReactNode, type RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { useOutsidePress } from "@/hooks/useOutsidePress"
import { cn } from "@/lib/utils"

interface MenuPopoverState {
  /** Wraps the trigger and the menu; a press outside it closes the menu. */
  ref: RefObject<HTMLDivElement | null>
  open: boolean
  toggle: () => void
  close: () => void
}

/** Open state of a trigger plus popover menu, closed by a mousedown outside `ref` and, with `escape`, by Escape on the document. */
export function useMenuPopover({ escape = true }: { escape?: boolean } = {}): MenuPopoverState {
  const ref = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const toggle = useCallback(() => setOpen((current) => !current), [])
  const close = useCallback(() => setOpen(false), [])
  useOutsidePress(ref, close, { active: open })
  useEventListener("document", "keydown", (event) => {
    if (event.key === "Escape") close()
  }, { active: open && escape })
  return { ref, open, toggle, close }
}

/**
 * A trigger and its popover menu inside one `relative` wrapper, closed by an outside press or Escape.
 *
 * - `trigger` renders the button, `children` the menu; both get the open state, so the menu decides how it appears (conditionally, or inside `AnimatePresence`)
 * - `escape={false}` leaves Escape alone
 */
export function MenuPopover({ trigger, children, className, escape }: {
  trigger: (open: boolean, toggle: () => void) => ReactNode
  children: (open: boolean, close: () => void) => ReactNode
  className?: string
  escape?: boolean
}) {
  const { ref, open, toggle, close } = useMenuPopover({ escape })
  return (
    <div ref={ref} className={cn("relative", className)}>
      {trigger(open, toggle)}
      {children(open, close)}
    </div>
  )
}
