"use client"

import { useCallback, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from "react"
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

/** The bottom and top edges a menu inside `wrapper` may use: the viewport, narrowed by every scrolling or clipping ancestor. */
function visibleBand(wrapper: HTMLElement): { top: number; bottom: number } {
  let top = 0
  let bottom = window.innerHeight
  for (let node = wrapper.parentElement; node; node = node.parentElement) {
    if (getComputedStyle(node).overflowY === "visible") continue
    const rect = node.getBoundingClientRect()
    top = Math.max(top, rect.top)
    bottom = Math.min(bottom, rect.bottom)
  }
  return { top, bottom }
}

/**
 * Opens a downward menu upward when it would run past the bottom of its scroll container or the viewport
 * and there is more room above, keeping its gap to the trigger. Runs before paint, so the menu never jumps.
 */
function useFlipUp(ref: RefObject<HTMLDivElement | null>, open: boolean) {
  useLayoutEffect(() => {
    const wrapper = ref.current
    const menu = wrapper?.children[1]
    if (!open || !wrapper || !(menu instanceof HTMLElement) || getComputedStyle(menu).position !== "absolute") return
    // Layout offsets, not the menu's rect: the entrance animation already shifts and scales it.
    const anchor = wrapper.getBoundingClientRect()
    const gap = menu.offsetTop - wrapper.offsetHeight
    if (gap < 0) return
    const band = visibleBand(wrapper)
    if (anchor.bottom + gap + menu.offsetHeight <= band.bottom || anchor.top - band.top <= band.bottom - anchor.bottom) return
    const flipped: Record<string, string> = {
      top: "auto",
      bottom: `calc(100% + ${gap}px)`,
      "margin-top": "0",
      "transform-origin": "bottom",
      "--motion-popover-shift": "4px",
    }
    const saved = Object.keys(flipped).map((name) => [name, menu.style.getPropertyValue(name)] as const)
    for (const [name, value] of Object.entries(flipped)) menu.style.setProperty(name, value)
    return () => {
      for (const [name, value] of saved) menu.style.setProperty(name, value)
    }
  }, [ref, open])
}

/**
 * A trigger and its popover menu inside one `relative` wrapper, closed by an outside press or Escape.
 *
 * - `trigger` renders the button, `children` the menu; both get the open state, so the menu decides how it appears (conditionally, or inside `AnimatePresence`)
 * - The menu, the element after the trigger, fades and rises from the trigger on mount (`motion-popover` in globals.css): up from a `bottom-full` menu, down otherwise
 * - A downward menu without room below in its scroll container or the viewport opens upward instead
 * - `escape={false}` leaves Escape alone
 */
export function MenuPopover({ trigger, children, className, escape }: {
  trigger: (open: boolean, toggle: () => void) => ReactNode
  children: (open: boolean, close: () => void) => ReactNode
  className?: string
  escape?: boolean
}) {
  const { ref, open, toggle, close } = useMenuPopover({ escape })
  useFlipUp(ref, open)
  return (
    <div ref={ref} data-menu-popover className={cn("relative", className)}>
      {trigger(open, toggle)}
      {children(open, close)}
    </div>
  )
}
