import type React from "react"
import type { Message } from "./types"

const FLUSH_MS = 60

// Coalesces in-place mutations of the trailing assistant `msg` into setMessages
// calls at most every FLUSH_MS. Both useChatStream and useChat.morphicSearch
// accumulate tokens onto that message and call schedule() per event; final()
// guarantees the last state lands. Single home for the flush plumbing both
// callers used to duplicate verbatim.
export function createFlushBatcher(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  msg: Message,
) {
  let lastFlush = 0
  let pending = false
  let timer: ReturnType<typeof setTimeout> | null = null

  const flush = () => {
    pending = false
    lastFlush = performance.now()
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (!last || last.role !== "assistant") return prev
      const copy = [...prev]
      copy[copy.length - 1] = { ...msg }
      return copy
    })
  }

  const schedule = () => {
    if (pending) return
    const delay = Math.max(0, FLUSH_MS - (performance.now() - lastFlush))
    if (delay <= 0) {
      flush()
      return
    }
    pending = true
    timer = setTimeout(flush, delay)
  }

  const final = () => {
    if (timer) clearTimeout(timer)
    flush()
  }

  return { schedule, final }
}
