import type React from "react"
import type { SSEEvent } from "./sse"
import type { Message, ToolCallProgress, ToolCallRecord, ToolCallResult, ToolCallStarted, Usage } from "./types"

// Publish every SSE event immediately. Delaying state updates turns short and fast
// responses into a single final render, which is indistinguishable from no streaming.
export function createFlushBatcher(
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>,
  msg: Message,
) {
  // `msg` remains the mutable accumulator. `owned` tracks the copy React last
  // received so an abandoned stream cannot overwrite a newer assistant message.
  let owned: Message = msg

  const flush = () => {
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
    flush()
  }

  const final = () => {
    flush()
  }

  return { schedule, final }
}

/** A wire event decoded into the shape the chat reducer consumes. Unknown event
 *  names decode to null so new backend events stay inert instead of breaking the fold. */
export type ChatStreamEvent =
  | { kind: "token"; text: string }
  | { kind: "tool_call"; started: ToolCallStarted }
  | { kind: "tool_progress"; progress: ToolCallProgress }
  | { kind: "tool_result"; result: ToolCallResult }
  | { kind: "usage"; usage: Usage }
  | { kind: "done" }
  | { kind: "error"; message: string }

// Malformed JSON throws: the stream loop's catch treats it as a failed turn, not a skipped event.
export function decodeChatStreamEvent(event: SSEEvent): ChatStreamEvent | null {
  switch (event.event) {
    case "token":
      return { kind: "token", text: event.data }
    case "tool_call":
      return { kind: "tool_call", started: JSON.parse(event.data) as ToolCallStarted }
    case "tool_result":
      return { kind: "tool_result", result: JSON.parse(event.data) as ToolCallResult }
    case "tool_progress":
      return { kind: "tool_progress", progress: JSON.parse(event.data) as ToolCallProgress }
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

// Matches on call id only. A result for an unknown id is dropped (out-of-order or post-abort).
function withToolCallResult(
  calls: ToolCallRecord[] | undefined,
  result: ToolCallResult,
): ToolCallRecord[] {
  return (calls ?? []).map((call) =>
    call.id === result.id ? { ...call, ...result, status: result.error ? "error" : "done" } : call
  )
}

function withToolCallProgress(
  calls: ToolCallRecord[] | undefined,
  progress: ToolCallProgress,
): ToolCallRecord[] {
  return (calls ?? []).map((call) => call.id === progress.id ? { ...call, stages: progress.stages } : call)
}

/** The events that fold into the trailing assistant message. `usage` and `done`
 *  stay with the caller: one lands in React state, the other only ends the loop. */
export type MessageStreamEvent = Extract<
  ChatStreamEvent,
  { kind: "token" | "tool_call" | "tool_progress" | "tool_result" | "error" }
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
    case "tool_progress":
      msg.tool_calls = withToolCallProgress(msg.tool_calls, event.progress)
      break
    case "error":
      msg.content += `\n\nError: ${event.message}`
      break
  }
}

/** Terminal sweep for tool records the turn never resolved (abort, transport failure,
 *  SSE error or stream end). Marked `error` rather than a new status so a saved chat
 *  reloads a stated failure instead of a spinner. Mutates `msg` in place. */
export function settleRunningToolCalls(msg: Message, reason: string): void {
  const calls = msg.tool_calls
  if (!calls?.some((call) => call.status === "running")) return
  // Stage timestamps are in seconds
  const now = Date.now() / 1_000
  msg.tool_calls = calls.map((call) =>
    call.status === "running"
      ? {
          ...call,
          status: "error",
          summary: reason,
          error: reason,
          stages: call.stages?.map((stage) => stage.status === "running"
            ? { ...stage, status: "error", duration: Math.max(stage.duration, now - stage.started_at) }
            : stage),
        }
      : call
  )
}

