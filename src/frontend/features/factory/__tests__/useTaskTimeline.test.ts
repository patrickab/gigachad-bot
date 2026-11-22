import { beforeEach, describe, expect, it, vi } from "vitest"
import { renderHook, waitFor } from "@testing-library/react"
import type { FactoryAnswer, FactoryEntry } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  listFactoryEntries: vi.fn<(sessionId: string, since?: string) => Promise<FactoryEntry[]>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { groupTimeline, useTaskTimeline } from "@/features/factory/useTaskTimeline"

const USAGE = { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, cost: 0.25 }

function entry<Kind extends FactoryEntry["customType"]>(id: string, customType: Kind, data: Extract<FactoryEntry, { customType: Kind }>["data"]): FactoryEntry {
  return { id, timestamp: "2025-01-01T00:00:00Z", customType, data } as FactoryEntry
}

const gate = (id: string, round: number, exitCode: number) =>
  entry(id, "factory-gate", { round, phase: "gates", argv: ["factory/gates.sh"], exitCode, killed: false, durationMs: 1200, outputTail: exitCode ? "boom" : "" })

const ENTRIES: FactoryEntry[] = [
  entry("e1", "factory-task", { title: "T", request: "R", settings: { merge: "review", maxRounds: 3, review: true, models: {}, branch: "master" }, branch: "factory/t", mainBranch: "master", baseSha: "abc" }),
  entry("e2", "factory-round", { round: 1, trigger: "start" }),
  entry("e3", "factory-state", { state: "running", round: 1, reason: "start" }),
  gate("e4", 1, 1),
  entry("e5", "factory-round", { round: 2, trigger: "gates" }),
  entry("e6", "factory-state", { state: "running", round: 2, reason: "gates" }),
  gate("e7", 2, 0),
  entry("e8", "factory-review", { round: 2, status: "ok", approved: true, findings: [], model: "haiku", sessionFile: "/r.jsonl", usage: USAGE, wallMs: 4000 }),
  entry("e9", "factory-state", { state: "waiting_approve", round: 2, reason: "approved" }),
  entry("e10", "factory-state", { state: "merging", round: 2, reason: "approved" }),
  entry("e11", "factory-merge", { ok: true, commit: "0123456789abcdef", error: "", depsSynced: false }),
  entry("e12", "factory-state", { state: "done", round: 2, reason: "merged" }),
]

beforeEach(() => {
  api.listFactoryEntries.mockReset()
})

describe("groupTimeline", () => {
  it("groups gates, reviews and the merge under their round with its trigger and newest state", () => {
    const { rounds, review } = groupTimeline(ENTRIES)

    expect(rounds.map((round) => [round.round, round.trigger, round.state?.state, round.steps.map((step) => step.id)])).toEqual([
      [1, "start", "running", ["e4"]],
      [2, "gates", "done", ["e7", "e8", "e11"]],
    ])
    expect(review.cost).toBe(0.25)
  })

  it("sums the usage of every review, cache reads and writes as input", () => {
    const review = (id: string, round: number) => entry(id, "factory-review", { round, status: "ok", approved: false, findings: [], model: "haiku", sessionFile: "", usage: USAGE, wallMs: 1 })
    expect(groupTimeline([review("a", 1), review("b", 2)]).review).toEqual({
      input: 2 * (USAGE.input + USAGE.cacheRead + USAGE.cacheWrite),
      output: 2 * USAGE.output,
      cost: 0.5,
    })
  })

  it("puts each coder answer in the newest round and sums the coder's usage", () => {
    const answer = (id: string, text: string, cost: number): FactoryAnswer => ({
      id, timestamp: "", type: "message", message: { role: "assistant", text, model: "sonnet", stopReason: "stop", usage: { input: 100, output: 10, cost } },
    })
    const timeline = groupTimeline([
      entry("r1", "factory-round", { round: 1, trigger: "start" }),
      answer("a1", "Looking.", 0.5),
      answer("a2", "Done.", 0.25),
      entry("r2", "factory-round", { round: 2, trigger: "gates" }),
      answer("a3", "Fixed.", 0.25),
    ])

    expect(timeline.rounds.map((round) => [round.round, round.answers.map((each) => each.message.text)])).toEqual([[1, ["Looking.", "Done."]], [2, ["Fixed."]]])
    expect(timeline.coder).toEqual({ input: 300, output: 30, cost: 1 })
  })

  it("keeps a step whose round entry was not read, without a trigger", () => {
    expect(groupTimeline([gate("e4", 3, 0)]).rounds).toEqual([{ round: 3, trigger: null, steps: [expect.objectContaining({ id: "e4" })], answers: [], state: null }])
  })
})

describe("useTaskTimeline", () => {
  it("reads all entries once, then only those after the last one when the revision changes, and appends them", async () => {
    api.listFactoryEntries.mockResolvedValueOnce(ENTRIES.slice(0, 4)).mockResolvedValueOnce(ENTRIES.slice(4, 7)).mockResolvedValueOnce([])
    const hook = renderHook(({ revision }) => useTaskTimeline("s1", revision), { initialProps: { revision: "a" } })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(4))
    expect(api.listFactoryEntries).toHaveBeenCalledTimes(1)
    expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", undefined)

    hook.rerender({ revision: "b" })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(7))
    expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", "e4")
    expect(hook.result.current.entries.map((item) => item.id)).toEqual(["e1", "e2", "e3", "e4", "e5", "e6", "e7"])

    // Nothing new: the cursor and the entries stay
    hook.rerender({ revision: "c" })
    await waitFor(() => expect(api.listFactoryEntries).toHaveBeenCalledTimes(3))
    expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", "e7")
    expect(hook.result.current.entries).toHaveLength(7)
  })

  it("does not read again while the revision stays, nor before the session is known", async () => {
    api.listFactoryEntries.mockResolvedValue([])
    const hook = renderHook(({ sessionId }: { sessionId?: string }) => useTaskTimeline(sessionId, "a"), { initialProps: {} })
    expect(api.listFactoryEntries).not.toHaveBeenCalled()

    hook.rerender({ sessionId: "s1" })
    hook.rerender({ sessionId: "s1" })
    await waitFor(() => expect(api.listFactoryEntries).toHaveBeenCalledTimes(1))
  })

  it("reads a change that arrives during a read after it, from the new cursor", async () => {
    let answerFirst: (entries: FactoryEntry[]) => void = () => {}
    api.listFactoryEntries.mockReturnValueOnce(new Promise((resolve) => { answerFirst = resolve })).mockResolvedValueOnce(ENTRIES.slice(2, 4))
    const hook = renderHook(({ revision }) => useTaskTimeline("s1", revision), { initialProps: { revision: "a" } })
    await waitFor(() => expect(api.listFactoryEntries).toHaveBeenCalledTimes(1))
    hook.rerender({ revision: "b" })
    await Promise.resolve()
    expect(api.listFactoryEntries).toHaveBeenCalledTimes(1)

    answerFirst(ENTRIES.slice(0, 2))
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(4))
    expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", "e2")
  })

  it("keeps the entries and reports a failed read, the next read clears the error", async () => {
    api.listFactoryEntries.mockResolvedValueOnce(ENTRIES.slice(0, 2)).mockRejectedValueOnce(new Error("pi-host is down")).mockResolvedValueOnce([])
    const hook = renderHook(({ revision }) => useTaskTimeline("s1", revision), { initialProps: { revision: "a" } })
    await waitFor(() => expect(hook.result.current.entries).toHaveLength(2))

    hook.rerender({ revision: "b" })
    await waitFor(() => expect(hook.result.current.error).toBe("pi-host is down"))
    expect(hook.result.current.entries).toHaveLength(2)

    hook.rerender({ revision: "c" })
    await waitFor(() => expect(hook.result.current.error).toBe(""))
    expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", "e2")
  })
})
