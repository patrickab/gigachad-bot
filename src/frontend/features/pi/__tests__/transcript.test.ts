import { describe, expect, it } from "vitest"
import { messagesOfEntries, reduceTranscript, transcriptOfSnapshot, type SessionEntry } from "../transcript"
import type { AgentMessage, AssistantMessage, UserMessage } from "../types"

const prompt: UserMessage = { role: "user", content: "Fix the bug", timestamp: 1_000 }
const reply: AssistantMessage = { role: "assistant", content: [{ type: "text", text: "Done" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 2_000 }
const followUp: UserMessage = { role: "user", content: "Thanks", timestamp: 3_000 }

function messageEntry(id: string, parentId: string | null, message: AgentMessage): SessionEntry {
  return { type: "message", id, parentId, timestamp: new Date(message.timestamp).toISOString(), message }
}

const EMPTY_QUEUE = { steering: [], followUp: [] }

describe("messagesOfEntries", () => {
  it("follows the active branch from the leaf, skipping abandoned ones", () => {
    const abandoned: UserMessage = { role: "user", content: "Other idea", timestamp: 2_500 }
    const entries = [
      messageEntry("a", null, prompt),
      messageEntry("b", "a", reply),
      messageEntry("x", "b", abandoned),
      messageEntry("c", "b", followUp),
    ]
    expect(messagesOfEntries({ entries, leafId: "c" })).toEqual([prompt, reply, followUp])
    expect(messagesOfEntries({ entries, leafId: "x" })).toEqual([prompt, reply, abandoned])
  })

  it("shows compactions as summaries in place and drops entries that are not messages", () => {
    const entries: SessionEntry[] = [
      messageEntry("a", null, prompt),
      { type: "model_change", id: "m", parentId: "a", timestamp: "2026-01-01T00:00:00.000Z", provider: "openai", modelId: "gpt" },
      { type: "compaction", id: "k", parentId: "m", timestamp: "2026-01-01T00:00:01.000Z", summary: "Earlier work", firstKeptEntryId: "a", tokensBefore: 900 },
      messageEntry("c", "k", followUp),
    ]
    expect(messagesOfEntries({ entries, leafId: "c" }).map((message) => message.role)).toEqual(["user", "compactionSummary", "user"])
  })

  it("is empty without a leaf and survives a parent cycle", () => {
    expect(messagesOfEntries({ entries: [], leafId: null })).toEqual([])
    const cycle = [messageEntry("a", "b", prompt), messageEntry("b", "a", reply)]
    expect(messagesOfEntries({ entries: cycle, leafId: "b" })).toHaveLength(2)
  })
})

describe("snapshot plus replay", () => {
  it("rebuilds the running turn without duplicating messages the snapshot holds", () => {
    // Snapshot taken after the prompt was persisted, mid reply
    const snapshot = transcriptOfSnapshot({ entries: [messageEntry("a", null, prompt)], leafId: "a" }, true, EMPTY_QUEUE)
    const replay = [
      { type: "message_start", message: prompt },
      { type: "message_end", message: prompt },
      { type: "message_start", message: { ...reply, content: [] } },
      { type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } },
      { type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "Do" } },
    ]
    const replayed = replay.reduce(reduceTranscript, snapshot)
    expect(replayed.messages).toEqual([prompt])
    expect(replayed.running).toBe(true)
    expect(replayed.streaming?.content).toEqual([{ type: "text", text: "Do" }])

    const live = [
      { type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "ne" } },
      { type: "message_end", message: reply },
      { type: "agent_settled", aborted: false },
    ].reduce(reduceTranscript, replayed)
    expect(live.messages).toEqual([prompt, reply])
    expect(live.streaming).toBeNull()
    expect(live.running).toBe(false)
  })

  it("drops a replayed message_end the snapshot already committed, keeps the order of new ones", () => {
    const snapshot = transcriptOfSnapshot({ entries: [messageEntry("a", null, prompt), messageEntry("b", "a", reply)], leafId: "b" }, true, EMPTY_QUEUE)
    const toolResult: AgentMessage = { role: "toolResult", toolCallId: "call-1", toolName: "bash", content: [], isError: false, timestamp: 2_000 }
    const after = [
      { type: "message_end", message: reply },
      { type: "message_end", message: toolResult },
    ].reduce(reduceTranscript, snapshot)
    expect(after.messages).toEqual([prompt, reply, toolResult])
  })

  it("starts from the snapshot's queue and run state, dropping live tool runs", () => {
    const before = reduceTranscript(transcriptOfSnapshot({ entries: [], leafId: null }, false, EMPTY_QUEUE), { type: "tool_execution_start", toolCallId: "call-1" })
    expect(Object.keys(before.runs)).toEqual(["call-1"])
    const queue = { steering: ["Stop"], followUp: [] }
    const fresh = transcriptOfSnapshot({ entries: [], leafId: null }, true, queue)
    expect(fresh).toEqual({ messages: [], streaming: null, runs: {}, running: true, queue })
  })
})
