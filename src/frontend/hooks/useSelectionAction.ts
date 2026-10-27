"use client"

import { useCallback, useRef, useState, type RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"

/**
 * The text selection inside `containerRef` as `read` interprets it, for a floating action such as `SelectionBubble`.
 *
 * - Captured on pointer or key release inside the container
 * - Default: only a selection changed since the last capture counts, so a click that keeps it (a scrollbar) does not re-read, and a release `read` rejects keeps the previous value
 * - `live`: every selection change and every release re-reads, and a rejected read clears
 * - Cleared when the selection collapses, and on scroll or resize, which would strand the action
 */
export function useSelectionAction<T>(
  containerRef: RefObject<HTMLElement | null>,
  read: (container: HTMLElement) => T | null,
  { live = false }: { live?: boolean } = {},
): { selected: T | null; clear: () => void } {
  const [selected, setSelected] = useState<T | null>(null)
  const changedRef = useRef(false)

  useEventListener("document", "selectionchange", () => {
    if (live) {
      const container = containerRef.current
      setSelected(container ? read(container) : null)
      return
    }
    changedRef.current = true
    if (window.getSelection()?.isCollapsed) setSelected(null)
  })

  const capture = () => {
    const container = containerRef.current
    if (!container) return
    if (live) {
      setSelected(read(container))
      return
    }
    if (!changedRef.current) return
    changedRef.current = false
    const found = read(container)
    if (found !== null) setSelected(found)
  }
  useEventListener(containerRef, "pointerup", capture)
  useEventListener(containerRef, "keyup", capture)

  const clear = useCallback(() => setSelected(null), [])
  useEventListener("window", "scroll", clear, { capture: true })
  useEventListener("window", "resize", clear)

  return { selected, clear }
}
