"use client"

import { useCallback, useEffect, useRef, type RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import type { Message } from "@/lib/domain/types"

/** Within this many px of the bottom still counts as pinned, so small overscrolls keep following. */
const PINNED_SLACK_PX = 80

/** Keep the chat scrolled to `bottomRef` as `messages` change while the reader is at the bottom; `pin` re-pins it, e.g. on send. */
export function useStickToBottom(scrollRef: RefObject<HTMLDivElement | null>, bottomRef: RefObject<HTMLDivElement | null>, messages: Message[]): { pin: () => void } {
  // Ref, not state: updated on every scroll and read by the message effect without re-rendering
  const pinnedRef = useRef(true)
  const previousCount = useRef(0)

  useEventListener(scrollRef, "scroll", () => {
    const el = scrollRef.current
    if (el) pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < PINNED_SLACK_PX
  }, { passive: true })

  useEffect(() => {
    const wasEmpty = previousCount.current === 0
    previousCount.current = messages.length
    if (!pinnedRef.current) return
    // Restored histories jump straight to the end, avoiding work in every preview along a smooth scroll.
    bottomRef.current?.scrollIntoView({ behavior: wasEmpty ? "instant" : "smooth" })
  }, [messages, bottomRef])

  const pin = useCallback(() => { pinnedRef.current = true }, [])
  return { pin }
}
