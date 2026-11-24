import { describe, expect, it } from "vitest"
import {
  currentRows, dismissFinished, dismissRows, elapsedMs, EMPTY_ROSTER, inspectCommand, mergeRoster, parseInspectReply, rowsFromAsyncWidget, rowsFromToolDetails,
  type SubagentRow,
} from "../subagents"

const snapshot = (runs: unknown[]) => [`PI_SUBAGENT_ASYNC_JSON:${JSON.stringify({ kind: "pi-subagents.async-status-snapshot", version: 1, runs })}`]

describe("rowsFromAsyncWidget", () => {
  it("makes one row per background child, with its tool and times", () => {
    const rows = rowsFromAsyncWidget(snapshot([
      { id: "run1", kind: "subagent", label: "reviewer", state: "running", startedAt: 1000, progressText: "Reading the diff", activity: { currentTool: "read" }, agent: "reviewer" },
      { id: "wf", kind: "workflow", label: "wf", state: "complete", children: [
        { id: "s0", kind: "step", label: "scout", agent: "scout", state: "complete", startedAt: 5, endedAt: 9 },
        { id: "h", kind: "host-step", label: "gate", state: "complete" },
        { id: "s1", kind: "step", label: "worker", state: "failed" },
      ] },
    ]))
    expect(rows).toEqual([
      { id: "async:run1:", agent: "reviewer", task: "Reading the diff", status: "running", tool: "read", startedAt: 1000, endedAt: undefined, inspect: { asyncId: "run1" } },
      { id: "async:wf:s0", agent: "scout", task: undefined, status: "done", tool: undefined, startedAt: 5, endedAt: 9, inspect: { asyncId: "wf", childId: "s0" } },
      { id: "async:wf:s1", agent: "worker", task: undefined, status: "failed", tool: undefined, startedAt: undefined, endedAt: undefined, inspect: { asyncId: "wf", childId: "s1" } },
    ])
  })

  it("ignores missing, foreign or malformed widget lines", () => {
    expect(rowsFromAsyncWidget(undefined)).toEqual([])
    expect(rowsFromAsyncWidget(["other"])).toEqual([])
    expect(rowsFromAsyncWidget(["PI_SUBAGENT_ASYNC_JSON:{nope"])).toEqual([])
  })
})

describe("rowsFromToolDetails", () => {
  it("reads live progress of a running foreground call", () => {
    const rows = rowsFromToolDetails("call1", { mode: "parallel", progress: [
      { index: 0, agent: "scout", task: "Map the repo", status: "running", currentTool: "grep", tokens: 12_345, durationMs: 4000, recentOutput: ["a", "b"] },
      { index: 1, agent: "worker", task: "Fix it", status: "pending" },
    ] }, true)
    expect(rows).toMatchObject([
      { id: "fg:call1:0", agent: "scout", status: "running", tool: "grep", tokens: 12_345, durationMs: 4000, output: "a\nb" },
      { id: "fg:call1:1", agent: "worker", status: "queued" },
    ])
  })

  it("reads tokens, cost and output of finished children, a failed one fails", () => {
    const rows = rowsFromToolDetails("call2", { mode: "chain", results: [
      { index: 0, agent: "scout", task: "t", exitCode: 0, usage: { input: 1000, output: 500, cost: 0.0312 }, finalOutput: "found" },
      { index: 1, agent: "worker", task: "u", exitCode: 1, error: "boom", usage: { input: 1, output: 1, cost: 0 } },
      { index: 2, agent: "late", task: "v", exitCode: 0, interrupted: true },
    ] }, false)
    expect(rows).toMatchObject([
      { agent: "scout", status: "done", tokens: 1500, cost: 0.0312, output: "found" },
      { agent: "worker", status: "failed", output: "boom" },
      { agent: "late", status: "stopped" },
    ])
  })

  it("ignores management and unrelated details", () => {
    expect(rowsFromToolDetails("c", { mode: "management", results: [{ agent: "x" }] }, false)).toEqual([])
    expect(rowsFromToolDetails("c", { files: [] }, false)).toEqual([])
    expect(rowsFromToolDetails("c", undefined, false)).toEqual([])
  })
})

const row = (id: string, status: SubagentRow["status"]): SubagentRow => ({ id, agent: id, status })

describe("roster", () => {
  it("updates rows in place, keeps finished ones and counts a vanished active child as done", () => {
    let roster = mergeRoster(EMPTY_ROSTER, [row("a", "running"), row("b", "done")])
    roster = mergeRoster(roster, [row("a", "done"), row("c", "running")])
    expect(roster.rows.map((entry) => [entry.id, entry.status])).toEqual([["a", "done"], ["b", "done"], ["c", "running"]])
    roster = mergeRoster(roster, [], 99)
    expect(roster.rows.find((entry) => entry.id === "c")).toMatchObject({ status: "done", endedAt: 99 })
  })

  it("keeps dismissed children out even when the source still reports them", () => {
    let roster = mergeRoster(EMPTY_ROSTER, [row("a", "done"), row("b", "running")])
    roster = dismissRows(roster, ["a"])
    roster = mergeRoster(roster, [row("a", "done"), row("b", "running")])
    expect(roster.rows.map((entry) => entry.id)).toEqual(["b"])
  })

  it("dismisses only the finished rows", () => {
    const roster = dismissFinished(mergeRoster(EMPTY_ROSTER, [row("a", "done"), row("b", "running"), row("c", "failed")]))
    expect(roster.rows.map((entry) => entry.id)).toEqual(["b"])
    expect([...roster.dismissed].sort()).toEqual(["a", "c"])
  })

  it("combines both sources", () => {
    const rows = currentRows({ call: { running: true, isError: false, result: { content: [], details: { mode: "single", progress: [{ index: 0, agent: "w", status: "running" }] } } } }, {
      "subagent-async": snapshot([{ id: "r", kind: "subagent", label: "x", state: "queued" }]),
    })
    expect(rows.map((entry) => entry.id)).toEqual(["async:r:", "fg:call:0"])
  })
})

describe("helpers", () => {
  it("measures elapsed time from timestamps, or the reported duration", () => {
    expect(elapsedMs({ ...row("a", "running"), startedAt: 1000 }, 4000)).toBe(3000)
    expect(elapsedMs({ ...row("a", "done"), startedAt: 1000, endedAt: 2500 }, 9000)).toBe(1500)
    expect(elapsedMs({ ...row("a", "running"), durationMs: 700 }, 9000)).toBe(700)
    expect(elapsedMs(row("a", "running"), 9000)).toBeUndefined()
  })

  it("builds the inspect command and parses its reply", () => {
    expect(inspectCommand("r1", { asyncId: "run", childId: "s0" })).toBe("/subagents-inspect-rpc r1 run s0")
    expect(inspectCommand("r1", { asyncId: "run" })).toBe("/subagents-inspect-rpc r1 run")
    expect(parseInspectReply(['PI_SUBAGENT_INSPECT_JSON:{"requestId":"r1","task":"t"}'])).toMatchObject({ requestId: "r1", task: "t" })
    expect(parseInspectReply(['PI_SUBAGENT_INSPECT_JSON:{"task":"t"}'])).toBeNull()
  })
})
