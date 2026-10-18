"use client"

import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import { isNeovimKeyEvent } from "@/lib/utils"

/** Hover-preview and Ctrl+Up/Down keyboard focus over a chat's question/answer pairs. */
export function usePairFocus({ pairs, scrollContainerRef, togglePair }: {
  pairs: readonly { globalIndex: number }[]
  scrollContainerRef: RefObject<HTMLDivElement | null>
  /** Expand or collapse the pair starting at this message index. */
  togglePair: (globalIndex: number) => void
}) {
  const [hoveredPair, setHoveredPair] = useState<number | null>(null)
  const [keyboardFocusIdx, setKeyboardFocusIdx] = useState<number | null>(null)
  const collapsedRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const leaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const focusedGlobalIndex =
    keyboardFocusIdx !== null && keyboardFocusIdx >= 0 && keyboardFocusIdx < pairs.length
      ? pairs[keyboardFocusIdx].globalIndex
      : null

  // Hover preview opens after 500ms, closes after a 100ms grace so moving between rows doesn't flicker
  const handleMouseEnter = (globalIndex: number) => {
    if (leaveTimeoutRef.current) {
      clearTimeout(leaveTimeoutRef.current)
      leaveTimeoutRef.current = null
    }
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredPair(globalIndex)
    }, 500)
  }

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    leaveTimeoutRef.current = setTimeout(() => {
      setHoveredPair(null)
      leaveTimeoutRef.current = null
    }, 100)
  }

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
      if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    }
  }, [])

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
  useEffect(() => {
    const eat = (e: KeyboardEvent) => { e.preventDefault(); e.stopPropagation() }
    const handleKeyDown = (e: KeyboardEvent) => {
      if (isNeovimKeyEvent(e)) return
      const ctrl = e.ctrlKey || e.metaKey
      if (ctrl && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        eat(e)
        navigateFocus(e.key)
        scrollContainerRef.current?.focus()
      } else if (e.key === "Enter" && keyboardFocusIdx !== null) {
        eat(e)
        confirmFocus()
      } else if (e.key === "Escape" && keyboardFocusIdx !== null) {
        eat(e)
        setKeyboardFocusIdx(null)
      }
    }
    window.addEventListener("keydown", handleKeyDown, true)
    return () => window.removeEventListener("keydown", handleKeyDown, true)
  }, [navigateFocus, confirmFocus, keyboardFocusIdx, scrollContainerRef])

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
    hoveredPair,
    focusedGlobalIndex,
    clearKeyboardFocus: () => setKeyboardFocusIdx(null),
    handleMouseEnter,
    handleMouseLeave,
    collapsedRef,
  }
}
