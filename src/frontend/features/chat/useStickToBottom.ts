"use client"

import { useCallback, useEffect, useRef, type RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import type { ChatFocus } from "@/features/chat/ChatContainer"
import type { Message } from "@/lib/domain/types"

/** Within this many px of the bottom still counts as pinned, so small overscrolls keep following. */
const PINNED_SLACK_PX = 80

/**
 * Keep the chat scrolled to `bottomRef`: open each chat at its newest message, follow `messages` while the reader is at the bottom; `pin` re-pins it, e.g. on send.
 *
 * - Until the reader scrolls, clicks or types in the chat, late layout (composer padding, diagrams, plots) keeps it at the end
 * - Another `chatId` opens at the end again, the container is reused across chats
 * - A `focus` request scrolls to its question instead, settling stops
 */
export function useStickToBottom(scrollRef: RefObject<HTMLDivElement | null>, bottomRef: RefObject<HTMLDivElement | null>, chatId: string, messages: Message[], focus?: ChatFocus): { pin: () => void } {
  // Refs, not state: updated on every scroll and read by the effects without re-rendering
  const pinnedRef = useRef(true)
  const settlingRef = useRef(true)
  const previousCount = useRef(0)
  const openChatId = useRef(chatId)

  useEventListener(scrollRef, "scroll", () => {
    const el = scrollRef.current
    if (el) pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < PINNED_SLACK_PX
  }, { passive: true })

  const stopSettling = () => { settlingRef.current = false }
  useEventListener(scrollRef, "wheel", stopSettling, { passive: true })
  useEventListener(scrollRef, "touchstart", stopSettling, { passive: true })
  useEventListener(scrollRef, "pointerdown", stopSettling)
  useEventListener(scrollRef, "keydown", stopSettling)

  useEffect(() => {
    if (openChatId.current !== chatId) {
      openChatId.current = chatId
      previousCount.current = 0
      pinnedRef.current = true
      settlingRef.current = true
    }
    const wasEmpty = previousCount.current === 0
    previousCount.current = messages.length
    if (!pinnedRef.current) return
    // Restored histories jump straight to the end, avoiding work in every preview along a smooth scroll.
    bottomRef.current?.scrollIntoView({ behavior: wasEmpty ? "instant" : "smooth" })
  }, [chatId, messages, bottomRef])

  const focusQaIndex = focus?.qaIndex ?? null
  const focusKey = focus?.key
  useEffect(() => {
    if (focusQaIndex != null) settlingRef.current = false
  }, [focusQaIndex, focusKey])

  useEffect(() => {
    const el = scrollRef.current
    const content = bottomRef.current?.parentElement
    if (!el || !content) return
    // Not gated on `pinnedRef`: scroll anchoring shifts the view when content above grows, which is no reader scroll
    const keepAtEnd = new ResizeObserver(() => {
      if (!settlingRef.current) return
      el.scrollTop = el.scrollHeight
      pinnedRef.current = true
    })
    keepAtEnd.observe(el)
    keepAtEnd.observe(content)
    return () => keepAtEnd.disconnect()
  }, [scrollRef, bottomRef])

  const pin = useCallback(() => { pinnedRef.current = true }, [])
  return { pin }
}
