import { render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import type { Message, ToolCallRecord } from "@/lib/domain/types"
import type { ToolRun } from "@/features/conversation/tools/types"

const rows = vi.hoisted(() => [] as { id: string; run: ToolRun | undefined }[])
vi.mock("@/features/conversation/tools/ToolCallRow", () => ({
  ToolCallRow: ({ call, run }: { call: { id: string; name: string }; run?: ToolRun }) => {
    rows.push({ id: call.id, run })
    return <div>{call.name}</div>
  },
}))
vi.mock("@/features/chat/messages/ChatMessage", () => ({
  AssistantMessageContent: () => null,
  ChatMessage: ({ content }: { content: string }) => <div>{content}</div>,
}))

import { ChatPairBody } from "@/features/chat/ChatPair"
import { chatRecordOf, chatToolRun } from "@/features/chat/toolRun"
import { applyChatStreamEvent } from "@/lib/backend/streaming"

describe("chatToolRun", () => {
  it("maps a record to a Pi call and run, the record in the run's details", () => {
    const record: ToolCallRecord = { id: "t1", name: "legacy_tool", arguments: { a: 1 }, status: "error", error: "boom" }
    const { call, run } = chatToolRun(record)
    expect(call).toEqual({ type: "toolCall", id: "t1", name: "legacy_tool", arguments: { a: 1 } })
    expect(run).toEqual({ running: false, isError: true, result: { content: [{ type: "text", text: "boom" }], details: record } })
    expect(chatRecordOf(run)).toBe(record)
    expect(chatRecordOf({ running: false, isError: false, result: { content: [], details: { ...record } } })).toBeUndefined()
  })

  it("builds one run per record and status", () => {
    const record: ToolCallRecord = { id: "t1", name: "web_search", arguments: {}, status: "running" }
    expect(chatToolRun(record)).toBe(chatToolRun(record))
    expect(chatToolRun(record).run.running).toBe(true)
    const done = { ...record, status: "done" as const, summary: "3 sources" }
    expect(chatToolRun(done).run).toEqual({ running: false, isError: false, result: { content: [{ type: "text", text: "3 sources" }], details: done } })
  })
})

describe("a live pair's tool rows while a reply streams", () => {
  it("build no new run and re-render no settled row on a streamed frame", () => {
    const assistant: Message = { role: "assistant", content: "" }
    const user: Message = { role: "user", content: "Question" }
    applyChatStreamEvent(assistant, { kind: "tool_call", started: { id: "settled", name: "web_search", arguments: { query: "a" } } })
    applyChatStreamEvent(assistant, { kind: "tool_result", result: { id: "settled", name: "web_search", summary: "done", sources: [], detail: {}, error: null, sandbox: null } })
    applyChatStreamEvent(assistant, { kind: "tool_call", started: { id: "live", name: "workspace_agent", arguments: {} } })
    const frame = () => <ChatPairBody user={user} assistant={{ ...assistant }} globalIndex={0} qaIndex={0} isStreaming />
    const { rerender } = render(frame())
    const settledRuns = () => rows.filter((row) => row.id === "settled").map((row) => row.run)
    expect(settledRuns()).toHaveLength(1)

    applyChatStreamEvent(assistant, { kind: "token", text: "Answer" })
    rerender(frame())
    applyChatStreamEvent(assistant, { kind: "tool_progress", progress: { id: "live", stages: [{ id: "s", label: "Running", status: "running", started_at: 1, duration: 0 }] } })
    rerender(frame())

    expect(settledRuns()).toHaveLength(1)
    expect(chatToolRun(assistant.tool_calls![0]).run).toBe(settledRuns()[0])
    // The live call's record was replaced by its progress, so its row re-renders
    expect(rows.filter((row) => row.id === "live")).toHaveLength(2)
  })
})
