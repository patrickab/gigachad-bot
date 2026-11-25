import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { FactoryTimeline, FactoryTimelineData, FactoryTimelineEntry, FactoryTimelineKind } from "@/lib/domain/factory"

const core = vi.hoisted(() => ({ timeline: vi.fn<(id: string, since: number) => Promise<FactoryTimeline>>() }))
const api = vi.hoisted(() => ({
  callFactory: vi.fn(async (action: string, params: Record<string, unknown>) => {
    if (action !== "tasks.timeline") throw new Error(`unexpected ${action}`)
    return core.timeline(params.id as string, params.since as number)
  }),
}))

vi.mock("@/lib/backend/api", () => api)

import { groupTimeline, TIMELINE_POLL_MS, useTaskTimeline } from "@/features/factory/useTaskTimeline"

const USAGE = { input: 10, output: 5, cacheRead: 3, cacheWrite: 2, cost: 0.25 }

function entry<Kind extends FactoryTimelineKind>(kind: Kind, data: FactoryTimelineData[Kind]): FactoryTimelineEntry {
  return { at: "2025-01-01T00:00:00Z", kind, data } as FactoryTimelineEntry
}

const gate = (round: number, exitCode: number) =>
  entry("gate", { round, phase: "gates", argv: [`gates-${round}`], exitCode, killed: false, durationMs: 1200, outputTail: exitCode ? "boom" : "" })

const ENTRIES: FactoryTimelineEntry[] = [
  entry("task", { title: "T", request: "R", settings: { merge: "review", maxRounds: 3, review: true, models: {}, branch: "master" }, branch: "factory/t", mainBranch: "master", baseSha: "abc" }),
  entry("round", { round: 1, trigger: "start" }),
  entry("state", { state: "running", round: 1, reason: "start" }),
  gate(1, 1),
  entry("round", { round: 2, trigger: "gates" }),
  entry("state", { state: "running", round: 2, reason: "gates" }),
  gate(2, 0),
  entry("review", { round: 2, status: "ok", approved: true, findings: [], model: "haiku", sessionFile: "/r.jsonl", usage: USAGE, wallMs: 4000 }),
  entry("state", { state: "waiting_approve", round: 2, reason: "approved" }),
  entry("state", { state: "merging", round: 2, reason: "approved" }),
  entry("merge", { ok: true, commit: "0123456789abcdef", error: "", depsSynced: false }),
  entry("state", { state: "done", round: 2, reason: "merged" }),
]

function stepName(step: FactoryTimelineEntry): string {
  if (step.kind === "gate") return step.data.argv[0]
  return step.kind
}

beforeEach(() => {
  core.timeline.mockReset()
})

afterEach(() => {
  vi.useRealTimers()
})

describe("groupTimeline", () => {
  it("groups gates, reviews and the merge under their round with its trigger and newest state", () => {
    const { rounds, review } = groupTimeline(ENTRIES)

    expect(rounds.map((round) => [round.round, round.trigger, round.state?.state, round.steps.map(stepName)])).toEqual([
      [1, "start", "running", ["gates-1"]],
      [2, "gates", "done", ["gates-2", "review", "merge"]],
    ])
    expect(review.cost).toBe(0.25)
  })

  it("puts a task the coder started in the newest round", () => {
    const { rounds } = groupTimeline([...ENTRIES, entry("spawn", { id: "t9", title: "fix: follow-up" })])
    expect(rounds.at(-1)?.steps.map(stepName)).toEqual(["gates-2", "review", "merge", "spawn"])
  })

  it("sums the usage of every review as the core counts it: uncached input", () => {
    const review = (round: number) => entry("review", { round, status: "ok", approved: false, findings: [], model: "haiku", sessionFile: "", usage: USAGE, wallMs: 1 })
    expect(groupTimeline([review(1), review(2)]).review).toEqual({ input: 2 * USAGE.input, output: 2 * USAGE.output, cost: 0.5 })
  })

  it("puts each coder and user message in the newest round, in order", () => {
    const timeline = groupTimeline([
      entry("round", { round: 1, trigger: "start" }),
      entry("coder", { text: "Looking." }),
      entry("coder", { text: "Done." }),
      entry("user", { text: "Also the README." }),
      entry("round", { round: 2, trigger: "message" }),
      entry("coder", { text: "Fixed." }),
    ])

    expect(timeline.rounds.map((round) => [round.round, round.messages.map((message) => `${message.kind}: ${message.data.text}`)])).toEqual([
      [1, ["coder: Looking.", "coder: Done.", "user: Also the README."]],
      [2, ["coder: Fixed."]],
    ])
  })

  it("puts the coder's question and its answer in the round they continue", () => {
    const timeline = groupTimeline([
      entry("round", { round: 1, trigger: "start" }),
      entry("question", { text: "Keep the old flag?" }),
      entry("state", { state: "waiting_answer", round: 1, reason: "question" }),
      entry("user", { text: "Drop it." }),
      entry("coder", { text: "Dropped." }),
    ])

    expect(timeline.rounds.map((round) => [round.round, round.messages.map((message) => `${message.kind}: ${message.data.text}`)])).toEqual([
      [1, ["question: Keep the old flag?", "user: Drop it.", "coder: Dropped."]],
    ])
  })

  it("reads a merge request's source and target from its task entry, none for a coder's task", () => {
    const settings = { merge: "auto", maxRounds: 3, review: true, models: {}, branch: "master" } as const
    const request = entry("task", { title: "M", request: "R", settings, branch: "wave/w1", mainBranch: "master", baseSha: "abc", kind: "merge_request", source: "wave/w1" })

    expect(groupTimeline([request]).mergeRequest).toEqual({ source: "wave/w1", target: "master" })
    expect(groupTimeline(ENTRIES).mergeRequest).toBeNull()
  })

  it("keeps a step whose round entry was not read, without a trigger", () => {
    expect(groupTimeline([gate(3, 0)]).rounds).toEqual([{ round: 3, trigger: null, steps: [gate(3, 0)], messages: [], state: null }])
  })
})

describe("useTaskTimeline", () => {
  it("reads from 0 once, then on from the last next when the revision changes, and appends", async () => {
    core.timeline
      .mockResolvedValueOnce({ entries: ENTRIES.slice(0, 4), next: 400 })
      .mockResolvedValueOnce({ entries: ENTRIES.slice(4, 7), next: 700 })
      .mockResolvedValueOnce({ entries: [], next: 700 })
    const hook = renderHook(({ revision }) => useTaskTimeline("t1", revision, false), { initialProps: { revision: "a" } })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(4))
    expect(core.timeline).toHaveBeenLastCalledWith("t1", 0)

    hook.rerender({ revision: "b" })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(7))
    expect(core.timeline).toHaveBeenLastCalledWith("t1", 400)
    expect(hook.result.current.entries).toEqual(ENTRIES.slice(0, 7))

    // Nothing new: the cursor and the entries stay
    hook.rerender({ revision: "c" })
    await waitFor(() => expect(core.timeline).toHaveBeenCalledTimes(3))
    expect(core.timeline).toHaveBeenLastCalledWith("t1", 700)
    expect(hook.result.current.entries).toHaveLength(7)

    hook.rerender({ revision: "c" })
    await act(async () => {})
    expect(core.timeline).toHaveBeenCalledTimes(3)
  })

  it("reads a change that arrives during a read after it, from the new cursor", async () => {
    let answerFirst: (timeline: FactoryTimeline) => void = () => {}
    core.timeline.mockReturnValueOnce(new Promise((resolve) => { answerFirst = resolve })).mockResolvedValueOnce({ entries: ENTRIES.slice(2, 4), next: 400 })
    const hook = renderHook(({ revision }) => useTaskTimeline("t1", revision, false), { initialProps: { revision: "a" } })
    await waitFor(() => expect(core.timeline).toHaveBeenCalledTimes(1))
    hook.rerender({ revision: "b" })
    await Promise.resolve()
    expect(core.timeline).toHaveBeenCalledTimes(1)

    answerFirst({ entries: ENTRIES.slice(0, 2), next: 200 })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(4))
    expect(core.timeline).toHaveBeenLastCalledWith("t1", 200)
  })

  it("polls only while a runner holds the task", async () => {
    vi.useFakeTimers()
    core.timeline.mockResolvedValue({ entries: [], next: 0 })
    const hook = renderHook(({ running }) => useTaskTimeline("t1", "a", running), { initialProps: { running: false } })
    await act(async () => { await vi.advanceTimersByTimeAsync(5 * TIMELINE_POLL_MS) })
    expect(core.timeline).toHaveBeenCalledTimes(1)

    hook.rerender({ running: true })
    await act(async () => { await vi.advanceTimersByTimeAsync(2 * TIMELINE_POLL_MS) })
    expect(core.timeline).toHaveBeenCalledTimes(3)

    hook.rerender({ running: false })
    await act(async () => { await vi.advanceTimersByTimeAsync(5 * TIMELINE_POLL_MS) })
    expect(core.timeline).toHaveBeenCalledTimes(3)
  })

  it("keeps the entries and reports a failed read, the next read clears the error", async () => {
    core.timeline
      .mockResolvedValueOnce({ entries: ENTRIES.slice(0, 2), next: 200 })
      .mockRejectedValueOnce(new Error("pi-host is down"))
      .mockResolvedValueOnce({ entries: [], next: 200 })
    const hook = renderHook(({ revision }) => useTaskTimeline("t1", revision, false), { initialProps: { revision: "a" } })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(2))

    hook.rerender({ revision: "b" })
    await waitFor(() => expect(hook.result.current.error).toBe("pi-host is down"))
    expect(hook.result.current.entries).toHaveLength(2)

    hook.rerender({ revision: "c" })
    await waitFor(() => expect(hook.result.current.error).toBe(""))
    expect(core.timeline).toHaveBeenLastCalledWith("t1", 200)
  })
})
