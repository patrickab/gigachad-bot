import { describe, expect, it } from "vitest"
import { EMPTY_TRANSCRIPT, messagesOfEntries, reduceTranscript, shownMessages, transcriptReducer, type SessionEntry, type Transcript, type TranscriptAction } from "../transcript"
import type { AgentMessage, AssistantMessage, UserMessage } from "../types"

const prompt: UserMessage = { role: "user", content: "Fix the bug", timestamp: 1_000 }
const reply: AssistantMessage = { role: "assistant", content: [{ type: "text", text: "Done" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 2_000 }
const followUp: UserMessage = { role: "user", content: "Thanks", timestamp: 3_000 }

function messageEntry(id: string, parentId: string | null, message: AgentMessage): SessionEntry {
  return { type: "message", id, parentId, timestamp: new Date(message.timestamp).toISOString(), message }
}

const EMPTY_QUEUE = { steering: [], followUp: [] }

function snapshotOf(entries: SessionEntry[], running: boolean, queue: Transcript["queue"] = EMPTY_QUEUE, from: Transcript = EMPTY_TRANSCRIPT): Transcript {
  return transcriptReducer(from, { kind: "snapshot", entries: { entries, leafId: entries.at(-1)?.id ?? null }, running, queue })
}

function events(transcript: Transcript, records: { type: string; [field: string]: unknown }[]): Transcript {
  return records.reduce(reduceTranscript, transcript)
}

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
    const snapshot = snapshotOf([messageEntry("a", null, prompt)], true)
    const replayed = events(snapshot, [
      { type: "message_start", message: prompt },
      { type: "message_end", message: prompt },
      { type: "message_start", message: { ...reply, content: [] } },
      { type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } },
      { type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "Do" } },
    ])
    expect(shownMessages(replayed)).toEqual([prompt])
    expect(replayed.running).toBe(true)
    expect(replayed.streaming?.content).toEqual([{ type: "text", text: "Do" }])

    const live = events(replayed, [
      { type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "ne" } },
      { type: "message_end", message: reply },
      { type: "agent_settled", aborted: false },
    ])
    expect(live.messages).toEqual([prompt])
    expect(shownMessages(live)).toEqual([prompt, reply])
    expect(live.streaming).toBeNull()
    expect(live.running).toBe(false)
  })

  it("drops a replayed message_end the snapshot already committed, keeps the order of new ones", () => {
    const snapshot = snapshotOf([messageEntry("a", null, prompt), messageEntry("b", "a", reply)], true)
    const toolResult: AgentMessage = { role: "toolResult", toolCallId: "call-1", toolName: "bash", content: [], isError: false, timestamp: 2_000 }
    const after = events(snapshot, [
      { type: "message_end", message: reply },
      { type: "message_end", message: toolResult },
      { type: "message_end", message: toolResult },
    ])
    expect(after.unconfirmed).toEqual([toolResult])
    expect(shownMessages(after)).toEqual([prompt, reply, toolResult])
  })

  it("starts from the snapshot's queue and run state, dropping live tool runs and live messages", () => {
    const before = events(snapshotOf([], false), [{ type: "tool_execution_start", toolCallId: "call-1" }, { type: "message_end", message: reply }])
    expect(Object.keys(before.runs)).toEqual(["call-1"])
    const queue = { steering: ["Stop"], followUp: [] }
    const fresh = snapshotOf([], true, queue, before)
    expect(fresh).toEqual({ ...EMPTY_TRANSCRIPT, running: true, queue })
  })
})

describe("resync", () => {
  const base = snapshotOf([messageEntry("a", null, prompt)], true)

  it("appends the entries after the cursor and replaces the whole branch", () => {
    const appended = transcriptReducer(base, { kind: "resync", resync: { mode: "append", entries: [messageEntry("b", "a", reply)], leafId: "b" } })
    expect(appended.messages).toEqual([prompt, reply])
    expect(appended.running).toBe(true)

    // A fork: the branch now ends with another prompt after the reply
    const branch = [messageEntry("a", null, prompt), messageEntry("b", "a", reply), messageEntry("c", "b", followUp)]
    const replaced = transcriptReducer(appended, { kind: "resync", resync: { mode: "replace", entries: branch, leafId: "c" }, running: false, queue: EMPTY_QUEUE })
    expect(replaced.messages).toEqual([prompt, reply, followUp])
    expect(replaced.running).toBe(false)
  })

  it("never shows a live message_end twice when the resync brings its entry, in either order", () => {
    const bringReply: TranscriptAction = { kind: "resync", resync: { mode: "append", entries: [messageEntry("b", "a", reply)], leafId: "b" } }
    const eventFirst = transcriptReducer(events(base, [{ type: "message_end", message: reply }]), bringReply)
    expect(eventFirst.unconfirmed).toEqual([])
    expect(shownMessages(eventFirst)).toEqual([prompt, reply])

    const resyncFirst = events(transcriptReducer(base, bringReply), [{ type: "message_end", message: reply }])
    expect(shownMessages(resyncFirst)).toEqual([prompt, reply])
  })

  it("keeps live state for the replay and drops the streaming reply when the replay starts", () => {
    const streaming = events(base, [{ type: "message_start", message: { ...reply, content: [] } }, { type: "tool_execution_start", toolCallId: "call-1" }])
    const resynced = transcriptReducer(streaming, { kind: "resync", resync: { mode: "append", entries: [], leafId: "a" } })
    expect(resynced.streaming).not.toBeNull()
    const replaying = transcriptReducer(resynced, { kind: "replay_start" })
    expect(replaying.streaming).toBeNull()
    expect(Object.keys(replaying.runs)).toEqual(["call-1"])
  })

  it("drops the live state when the stream changes, keeping confirmed messages", () => {
    const live = events(transcriptReducer(base, { kind: "prompt", prompt: followUp }), [
      { type: "message_start", message: { ...reply, content: [] } },
      { type: "tool_execution_start", toolCallId: "call-1" },
      { type: "message_end", message: reply },
    ])
    const reset = transcriptReducer(live, { kind: "stream_changed" })
    expect(reset).toEqual({ ...base, unconfirmed: [], optimistic: null, streaming: null, runs: {} })
  })
})

describe("optimistic prompt", () => {
  const settled = snapshotOf([messageEntry("a", null, prompt), messageEntry("b", "a", reply)], false)
  const sent: UserMessage = { role: "user", content: "Thanks", timestamp: 9_999 }

  it("shows after the confirmed messages and goes when the confirmed prompt arrives", () => {
    const optimistic = transcriptReducer(settled, { kind: "prompt", prompt: sent })
    expect(shownMessages(optimistic)).toEqual([prompt, reply, sent])
    const confirmed = transcriptReducer(optimistic, { kind: "resync", resync: { mode: "append", entries: [messageEntry("c", "b", followUp)], leafId: "c" } })
    expect(confirmed.optimistic).toBeNull()
    expect(shownMessages(confirmed)).toEqual([prompt, reply, followUp])
  })

  it("hands over to Pi's live prompt, which the resync then confirms", () => {
    const optimistic = transcriptReducer(settled, { kind: "prompt", prompt: sent })
    const live = events(optimistic, [{ type: "message_start", message: followUp }, { type: "message_end", message: followUp }])
    expect(live.optimistic).toBeNull()
    expect(shownMessages(live)).toEqual([prompt, reply, followUp])
  })

  it("stays while a replace brings only older prompts with the same text", () => {
    const repeated = transcriptReducer(settled, { kind: "prompt", prompt: { ...prompt, timestamp: 9_999 } })
    const replaced = transcriptReducer(repeated, { kind: "resync", resync: { mode: "replace", entries: [messageEntry("a", null, prompt), messageEntry("b", "a", reply)], leafId: "b" } })
    expect(replaced.optimistic).not.toBeNull()
    expect(transcriptReducer(replaced, { kind: "prompt", prompt: null }).optimistic).toBeNull()
  })
})
