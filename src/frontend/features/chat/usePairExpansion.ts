"use client"

import { useEffect, useState, type RefObject } from "react"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { pairStart, type Pair } from "@/lib/domain/pairs"
import type { ChatFocus } from "@/features/chat/ChatContainer"

/** Trailing pairs expanded by default, older ones render collapsed. */
const DEFAULT_EXPANDED_TAIL = 1

/**
 * Which pairs are expanded, keyed by the pair's message index.
 *
 * - The trailing pairs are expanded and older ones collapsed unless toggled
 * - A new `focus` request replaces every toggle: it expands only the focused pair and scrolls to it
 * - An emptied chat drops the toggles
 */
export function usePairExpansion({ pairs, focus, messageCount, scrollContainerRef }: {
  pairs: Pair[]
  focus?: ChatFocus
  /** The length, not the array: in an empty chat every re-render hands down a fresh [] and would reset the toggles each time. */
  messageCount: number
  scrollContainerRef: RefObject<HTMLDivElement | null>
}) {
  const focusQaIndex = focus?.qaIndex ?? null
  const focusKey = focus?.key
  const focusOverrides = () => new Map<number, boolean>(focusQaIndex == null ? [] : [[pairStart(focusQaIndex), true]])
  const [overrides, setOverrides] = useState(focusOverrides)
  useResetOnChange(`${focusQaIndex}:${focusKey}`, () => setOverrides(focusOverrides()))
  useResetOnChange(messageCount, (count) => { if (count === 0) setOverrides(new Map()) })

  // Scroll to the pair before the focused one so the target sits just below a collapsed header
  useEffect(() => {
    if (focusQaIndex == null) return
    const start = pairStart(focusQaIndex)
    requestAnimationFrame(() => {
      const el = scrollContainerRef.current
      if (!el) return
      const anchor = start >= 2 ? start - 2 : start
      el.querySelector(`[data-qa-index="${anchor}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" })
    })
  }, [focusQaIndex, focusKey, scrollContainerRef])

  // Pairs from this message index on are expanded unless toggled
  const tailStart = (pairs.at(-DEFAULT_EXPANDED_TAIL) ?? pairs[0])?.globalIndex ?? -1
  const isExpanded = (globalIndex: number) => overrides.get(globalIndex) ?? globalIndex >= tailStart
  const toggle = (globalIndex: number) => setOverrides((prev) => new Map(prev).set(globalIndex, !isExpanded(globalIndex)))

  return { isExpanded, toggle }
}
