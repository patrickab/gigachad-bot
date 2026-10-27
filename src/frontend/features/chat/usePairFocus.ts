"use client"

import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { isNeovimKeyEvent } from "@/lib/utils"

/** Ctrl+Up/Down keyboard focus over a chat's question/answer pairs. */
export function usePairFocus({ pairs, scrollContainerRef, togglePair }: {
  pairs: readonly { globalIndex: number }[]
  scrollContainerRef: RefObject<HTMLDivElement | null>
  /** Expand or collapse the pair starting at this message index. */
  togglePair: (globalIndex: number) => void
}) {
  const [keyboardFocusIdx, setKeyboardFocusIdx] = useState<number | null>(null)
  const collapsedRefs = useRef<Map<number, HTMLDivElement>>(new Map())

  const focusedGlobalIndex =
    keyboardFocusIdx !== null && keyboardFocusIdx >= 0 && keyboardFocusIdx < pairs.length
      ? pairs[keyboardFocusIdx].globalIndex
      : null

  const navigateFocus = useCallback((key: "ArrowUp" | "ArrowDown") => {
    if (pairs.length === 0) {
      setKeyboardFocusIdx(null)
      return
    }
    setKeyboardFocusIdx((cur) => {
      const start = cur ?? (key === "ArrowDown" ? -1 : pairs.length)
      return Math.max(0, Math.min(start + (key === "ArrowDown" ? 1 : -1), pairs.length - 1))
    })
  }, [pairs])

  const confirmFocus = useCallback(() => {
    if (keyboardFocusIdx === null || keyboardFocusIdx < 0 || keyboardFocusIdx >= pairs.length) return
    togglePair(pairs[keyboardFocusIdx].globalIndex)
    setKeyboardFocusIdx(null)
  }, [keyboardFocusIdx, pairs, togglePair])

  // Capture phase so these shortcuts win over editor and input handlers
  useEventListener("window", "keydown", (e) => {
    const eat = () => { e.preventDefault(); e.stopPropagation() }
    if (isNeovimKeyEvent(e)) return
    const ctrl = e.ctrlKey || e.metaKey
    if (ctrl && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
      eat()
      navigateFocus(e.key)
      scrollContainerRef.current?.focus()
    } else if (e.key === "Enter" && keyboardFocusIdx !== null) {
      eat()
      confirmFocus()
    } else if (e.key === "Escape" && keyboardFocusIdx !== null) {
      eat()
      setKeyboardFocusIdx(null)
    }
  }, { capture: true })

  useEffect(() => {
    if (keyboardFocusIdx === null) return
    const container = scrollContainerRef.current
    if (!container) return

    // Scroll so the focused pair sits 3 rows below the top, or align to the top when near it
    const atTop = keyboardFocusIdx < 3
    const targetIdx = atTop ? 0 : keyboardFocusIdx - 3
    const targetGlobalIdx = pairs[targetIdx]?.globalIndex
    if (targetGlobalIdx === undefined) return
    const el = collapsedRefs.current.get(targetGlobalIdx)
    if (!el) return

    const elRect = el.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    const offset = (atTop ? elRect.top : elRect.bottom) - containerRect.top

    container.scrollTo({
      top: container.scrollTop + offset,
      behavior: "smooth"
    })
  }, [keyboardFocusIdx, pairs, scrollContainerRef])

  /** Ref callback registering the collapsed card of the pair at `globalIndex`. */
  const collapsedRef = (globalIndex: number) => (el: HTMLDivElement | null) => {
    if (el) collapsedRefs.current.set(globalIndex, el)
    else collapsedRefs.current.delete(globalIndex)
  }

  return {
    focusedGlobalIndex,
    clearKeyboardFocus: () => setKeyboardFocusIdx(null),
    collapsedRef,
  }
}
