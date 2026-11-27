import { describe, expect, it } from "vitest"
import { committedRunsOf, EMPTY_TRANSCRIPT, reduceTranscript, shownMessages, transcriptOfEntries, transcriptReducer, withLiveRuns, type SessionEntry, type Transcript, type TranscriptAction } from "../transcript"
import type { AgentMessage, AssistantMessage, CustomMessage, UserMessage } from "../types"

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

describe("transcriptOfEntries", () => {
  it("follows the active branch from the leaf, skipping abandoned ones", () => {
    const abandoned: UserMessage = { role: "user", content: "Other idea", timestamp: 2_500 }
    const entries = [
      messageEntry("a", null, prompt),
      messageEntry("b", "a", reply),
      messageEntry("x", "b", abandoned),
      messageEntry("c", "b", followUp),
    ]
    expect(transcriptOfEntries({ entries, leafId: "c" }).messages).toEqual([prompt, reply, followUp])
    expect(transcriptOfEntries({ entries, leafId: "x" }).messages).toEqual([prompt, reply, abandoned])
  })

  it("shows compactions as summaries in place and drops entries that are not messages", () => {
    const entries: SessionEntry[] = [
      messageEntry("a", null, prompt),
      { type: "model_change", id: "m", parentId: "a", timestamp: "2026-01-01T00:00:00.000Z", provider: "openai", modelId: "gpt" },
      { type: "compaction", id: "k", parentId: "m", timestamp: "2026-01-01T00:00:01.000Z", summary: "Earlier work", firstKeptEntryId: "a", tokensBefore: 900 },
      messageEntry("c", "k", followUp),
    ]
    expect(transcriptOfEntries({ entries, leafId: "c" }).messages.map((message) => message.role)).toEqual(["user", "compactionSummary", "user"])
  })

  it("is empty without a leaf and survives a parent cycle", () => {
    expect(transcriptOfEntries({ entries: [], leafId: null }).messages).toEqual([])
    const cycle = [messageEntry("a", "b", prompt), messageEntry("b", "a", reply)]
    expect(transcriptOfEntries({ entries: cycle, leafId: "b" }).messages).toHaveLength(2)
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

  it("places a live custom message by its entry, although Pi stamps the entry when it persists it", () => {
    const notice: CustomMessage = { role: "custom", customType: "factory-notice", content: "Task merged", display: true, timestamp: 2_500 }
    const live = events(base, [{ type: "message_end", message: notice }])
    expect(live.unconfirmed).toEqual([notice])

    const answer: AssistantMessage = { ...reply, content: [{ type: "text", text: "Nice" }], timestamp: 3_500 }
    const entries: SessionEntry[] = [
      messageEntry("a", null, prompt),
      { type: "custom_message", id: "n", parentId: "a", timestamp: new Date(2_507).toISOString(), customType: "factory-notice", content: "Task merged", display: true },
      messageEntry("c", "n", followUp),
      messageEntry("d", "c", answer),
    ]
    const resynced = transcriptReducer(live, { kind: "resync", resync: { mode: "replace", entries, leafId: "d" } })
    expect(resynced.unconfirmed).toEqual([])
    expect(shownMessages(resynced).map((message) => message.role)).toEqual(["user", "custom", "user", "assistant"])
    // A replayed message_end of the notice stays in place too
    expect(events(resynced, [{ type: "message_end", message: notice }]).unconfirmed).toEqual([])
  })

  it("keeps a second identical custom message apart from the first one's entry", () => {
    const notice: CustomMessage = { role: "custom", customType: "factory-notice", content: "Ping", display: true, timestamp: 1_500 }
    const entry: SessionEntry = { type: "custom_message", id: "n", parentId: "a", timestamp: new Date(1_501).toISOString(), customType: "factory-notice", content: "Ping", display: true }
    const confirmed = transcriptReducer(base, { kind: "resync", resync: { mode: "append", entries: [entry], leafId: "n" } })
    const again = { ...notice, timestamp: 1_800 }
    expect(events(confirmed, [{ type: "message_end", message: again }]).unconfirmed).toEqual([again])
  })

  it("drops live messages older than a message the resync brings", () => {
    const lost: UserMessage = { role: "user", content: "Lost", timestamp: 1_500 }
    const live = events(base, [{ type: "message_end", message: lost }])
    const resynced = transcriptReducer(live, { kind: "resync", resync: { mode: "append", entries: [messageEntry("b", "a", reply)], leafId: "b" } })
    expect(resynced.unconfirmed).toEqual([])
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

describe("entry edits", () => {
  const edit = (id: string, parentId: string, targetId: string, replacement: unknown): SessionEntry => ({ type: "context_edit", id, parentId, timestamp: "2026-01-01T00:00:05.000Z", targetId, replacement })
  const label = (id: string, parentId: string, targetId: string, text: string | undefined): SessionEntry => ({ type: "label", id, parentId, timestamp: "2026-01-01T00:00:06.000Z", targetId, label: text })
  const base = [messageEntry("a", null, prompt), messageEntry("b", "a", reply)]

  it("applies context_edit and label entries of a snapshot to their target instead of showing them", () => {
    const edited = snapshotOf([...base, edit("e1", "b", "b", "Shorter"), edit("e2", "e1", "a", null), label("l1", "e2", "b", "checkpoint")], false)
    expect(edited.messages.map((message) => message.role)).toEqual(["user", "assistant"])
    expect(edited.entryIds).toEqual(["a", "b"])
    expect((edited.messages[1] as AssistantMessage).content).toEqual([{ type: "text", text: "Shorter" }])
    expect(edited.messages[0]).toEqual(prompt)
    expect(edited.marks).toEqual({ a: { context: "omitted" }, b: { context: "replaced", label: "checkpoint" } })
  })

  it("applies edits that arrive via a resync append, the latest one winning", () => {
    const snapshot = snapshotOf(base, false)
    const appended = transcriptReducer(snapshot, {
      kind: "resync",
      resync: { mode: "append", entries: [edit("e1", "b", "a", "Fix it"), messageEntry("c", "e1", followUp), edit("e2", "c", "a", null), label("l1", "e2", "a", "start")], leafId: "l1" },
    })
    expect(shownMessages(appended)).toEqual([{ ...prompt, content: "Fix it" }, reply, followUp])
    expect(appended.entryIds).toEqual(["a", "b", "c"])
    expect(appended.marks).toEqual({ a: { context: "omitted", label: "start" } })

    const cleared = transcriptReducer(appended, { kind: "resync", resync: { mode: "append", entries: [label("l2", "l1", "a", undefined)], leafId: "l2" } })
    expect(cleared.marks).toEqual({ a: { context: "omitted" } })
  })
})

describe("live tool output", () => {
  const partial = (text: string) => ({ content: [{ type: "text", text }] })
  const text = (run: { result?: { content: unknown[] } }) => (run.result?.content[0] as { text: string }).text

  it("keeps only the tail of partial output and ignores chunks after the end", () => {
    const lines = Array.from({ length: 500 }, (_, i) => `line ${i}`).join("\n")
    const running = events(EMPTY_TRANSCRIPT, [{ type: "tool_execution_start", toolCallId: "c" }, { type: "tool_execution_update", toolCallId: "c", partialResult: partial(lines) }])
    const kept = text(running.runs.c).split("\n")
    expect(kept).toHaveLength(200)
    expect(kept.at(-1)).toBe("line 499")
    const ended = events(running, [{ type: "tool_execution_end", toolCallId: "c", result: partial("final") }, { type: "tool_execution_update", toolCallId: "c", partialResult: partial("late") }])
    expect(ended.runs.c.running).toBe(false)
    expect(text(ended.runs.c)).toBe("final")
  })

  it("lets a committed result win over a stale running one after a resync, without duplicates", () => {
    const toolResult: AgentMessage = { role: "toolResult", toolCallId: "c", toolName: "bash", content: [{ type: "text", text: "done" }], isError: false, timestamp: 2_500 }
    const live = events(EMPTY_TRANSCRIPT, [{ type: "tool_execution_start", toolCallId: "c" }, { type: "tool_execution_update", toolCallId: "c", partialResult: partial("part") }])
    const resynced = transcriptReducer(live, { kind: "resync", resync: { mode: "append", entries: [messageEntry("t", null, toolResult)], leafId: "t" } })
    const runs = withLiveRuns(committedRunsOf(shownMessages(resynced)), resynced.runs)
    expect(Object.keys(runs)).toEqual(["c"])
    expect(runs.c.running).toBe(false)
    expect(text(runs.c)).toBe("done")
  })

  it("lays live runs over the committed ones without touching them", () => {
    const result = (id: string): AgentMessage => ({ role: "toolResult", toolCallId: id, toolName: "bash", content: [{ type: "text", text: `done ${id}` }], isError: false, timestamp: 1 })
    const messages = [result("a"), result("b")]
    const committed = committedRunsOf(messages)
    expect(committedRunsOf(messages).a).toBe(committed.a)
    const runs = withLiveRuns(committed, { a: { running: true, isError: false, durationMs: 5 }, b: { running: false, isError: true, durationMs: 7 }, c: { running: true, isError: false } })
    expect(runs.a).toMatchObject({ running: false, durationMs: 5 })
    expect(text(runs.a)).toBe("done a")
    expect(runs.b).toMatchObject({ running: false, isError: true, durationMs: 7 })
    expect(runs.c.running).toBe(true)
    expect(Object.keys(committed)).toEqual(["a", "b"])
    expect(committed.a.durationMs).toBeUndefined()
  })
})
