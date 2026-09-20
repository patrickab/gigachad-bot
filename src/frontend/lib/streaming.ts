import type React from "react"
import type { SSEEvent } from "./sse"
import type { Message, ToolCallRecord, ToolCallResult, ToolCallStarted, Usage } from "./types"

const FLUSH_MS = 60

// Coalesces in-place mutations of the trailing assistant `msg` into setMessages
// calls at most every FLUSH_MS. Both useChatStream and useChat.webSearch
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
  // The trailing assistant message this batcher owns: `msg` itself until the first
  // flush, then the copy that flush published. A newer turn replaces the tail with a
  // message this batcher never published, so a late flush from an abandoned stream
  // finds a foreign tail and publishes nothing.
  let owned: Message = msg

  const flush = () => {
    pending = false
    lastFlush = performance.now()
    const published = { ...msg }
    const expected = owned
    // Advance before the updater runs: React invokes it later, and may invoke it twice.
    owned = published
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (!last || last.role !== "assistant" || last !== expected) return prev
      const copy = [...prev]
      copy[copy.length - 1] = published
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

/** A wire event decoded into the shape the chat reducer consumes. Unknown event
 *  names decode to null so new backend events stay inert instead of breaking the fold. */
export type ChatStreamEvent =
  | { kind: "token"; text: string }
  | { kind: "tool_call"; started: ToolCallStarted }
  | { kind: "tool_result"; result: ToolCallResult }
  | { kind: "usage"; usage: Usage }
  | { kind: "done" }
  | { kind: "error"; message: string }

// Malformed JSON throws here exactly as the inline parses did: the stream loop's
// catch treats it as a failed turn rather than a silently skipped event.
export function decodeChatStreamEvent(event: SSEEvent): ChatStreamEvent | null {
  switch (event.event) {
    case "token":
      return { kind: "token", text: event.data }
    case "tool_call":
      return { kind: "tool_call", started: JSON.parse(event.data) as ToolCallStarted }
    case "tool_result":
      return { kind: "tool_result", result: JSON.parse(event.data) as ToolCallResult }
    case "usage":
      return { kind: "usage", usage: JSON.parse(event.data) as Usage }
    case "done":
      return { kind: "done" }
    case "error":
      return { kind: "error", message: event.data }
    default:
      return null
  }
}

function withToolCallStarted(
  calls: ToolCallRecord[] | undefined,
  started: ToolCallStarted,
): ToolCallRecord[] {
  return [...(calls ?? []), { ...started, status: "running" }]
}

// Matches on call id only; a result for an unknown id lands nowhere, which is how
// out-of-order or post-abort results have always been dropped.
function withToolCallResult(
  calls: ToolCallRecord[] | undefined,
  result: ToolCallResult,
): ToolCallRecord[] {
  return (calls ?? []).map((call) =>
    call.id === result.id ? { ...call, ...result, status: result.error ? "error" : "done" } : call
  )
}

/** The events that fold into the trailing assistant message. `usage` and `done`
 *  stay with the caller: one lands in React state, the other only ends the loop. */
export type MessageStreamEvent = Extract<
  ChatStreamEvent,
  { kind: "token" | "tool_call" | "tool_result" | "error" }
>

// Mutates `msg` in place; createFlushBatcher publishes the copy.
export function applyChatStreamEvent(msg: Message, event: MessageStreamEvent): void {
  switch (event.kind) {
    case "token":
      msg.content += event.text
      break
    case "tool_call":
      msg.tool_calls = withToolCallStarted(msg.tool_calls, event.started)
      break
    case "tool_result":
      msg.tool_calls = withToolCallResult(msg.tool_calls, event.result)
      break
    case "error":
      msg.content += `\n\nError: ${event.message}`
      break
  }
}

/** Terminal sweep for tool records the turn never resolved — abort, transport failure,
 *  SSE error or a stream that simply ended. Cancelled work is an `error` record rather
 *  than a fourth status, so a saved chat reloads a stated failure instead of a spinner.
 *  Mutates `msg` in place; createFlushBatcher publishes the copy. */
export function settleRunningToolCalls(msg: Message, reason: string): void {
  const calls = msg.tool_calls
  if (!calls?.some((call) => call.status === "running")) return
  msg.tool_calls = calls.map((call) =>
    call.status === "running" ? { ...call, status: "error", summary: reason, error: reason } : call
  )
}

export function addUsage(prev: Usage, turn: Usage): Usage {
  return {
    prompt_tokens: prev.prompt_tokens + turn.prompt_tokens,
    completion_tokens: prev.completion_tokens + turn.completion_tokens,
    total_tokens: prev.total_tokens + turn.total_tokens,
  }
}
