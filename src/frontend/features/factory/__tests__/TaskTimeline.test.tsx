import { describe, expect, it } from "vitest"
import { render, within } from "@testing-library/react"
import type { FactoryEntry } from "@/lib/domain/factory"
import { TaskTimeline } from "@/features/factory/TaskTimeline"

const USAGE = { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, cost: 0.25 }

const ENTRIES = [
  { id: "e2", timestamp: "", customType: "factory-round", data: { round: 1, trigger: "start" } },
  { id: "e3", timestamp: "", customType: "factory-state", data: { state: "running", round: 1, reason: "start" } },
  { id: "e4", timestamp: "", customType: "factory-gate", data: { round: 1, phase: "hooks", argv: ["ruff", "format"], exitCode: 0, killed: false, durationMs: 400, outputTail: "" } },
  { id: "e5", timestamp: "", customType: "factory-gate", data: { round: 1, phase: "gates", argv: ["scripts/check.sh"], exitCode: 2, killed: false, durationMs: 12000, outputTail: "1 failed" } },
  { id: "e6", timestamp: "", customType: "factory-round", data: { round: 2, trigger: "gates" } },
  { id: "e7", timestamp: "", customType: "factory-state", data: { state: "running", round: 2, reason: "gates" } },
  { id: "e8", timestamp: "", customType: "factory-gate", data: { round: 2, phase: "gates", argv: ["scripts/check.sh"], exitCode: null, killed: true, durationMs: 60000, outputTail: "" } },
  {
    id: "e9", timestamp: "", customType: "factory-review",
    data: { round: 2, status: "ok", approved: false, findings: [{ file: "src/a.ts", line: 12, severity: "blocker", text: "Leaks the handle" }], model: "haiku", sessionFile: "", usage: USAGE, wallMs: 4000 },
  },
  { id: "e10", timestamp: "", customType: "factory-state", data: { state: "blocked", round: 2, reason: "max_rounds" } },
] as unknown as FactoryEntry[]

describe("TaskTimeline", () => {
  it("shows the totals and each round with its trigger and status", () => {
    const view = render(<TaskTimeline entries={ENTRIES} error="" />)

    expect(view.getByLabelText("Totals").textContent).toBe("2 rounds · review $0.25")
    const first = within(view.getByLabelText("Round 1"))
    expect(first.getByText("start")).toBeTruthy()
    // Round 1 was ended by round 2: it is not shown as still working
    expect(first.queryByText("Working")).toBeNull()
    const second = within(view.getByLabelText("Round 2"))
    expect(second.getByText("after failed gates")).toBeTruthy()
    expect(second.getByText("Blocked · max_rounds")).toBeTruthy()
  })

  it("shows each gate with its result and duration, a failed one's output collapsed", () => {
    const view = render(<TaskTimeline entries={ENTRIES} error="" />)
    const first = within(view.getByLabelText("Round 1"))

    expect(first.getByText("ruff format")).toBeTruthy()
    expect(first.getByText("passed")).toBeTruthy()
    expect(first.getByText("0.4s")).toBeTruthy()
    expect(first.getByText("exit 2")).toBeTruthy()
    expect(first.getByText("12s")).toBeTruthy()
    const output = first.getByText("1 failed").closest("details")!
    expect(output.open).toBe(false)
    // A killed command has no exit code
    expect(within(view.getByLabelText("Round 2")).getByText("killed")).toBeTruthy()
  })

  it("shows a command that ended without an exit code as failed", () => {
    const entries = [
      { id: "e1", timestamp: "", customType: "factory-gate", data: { round: 1, phase: "after_merge", argv: ["uv", "sync"], exitCode: null, killed: false, durationMs: 100, outputTail: "spawn ENOENT" } },
    ] as unknown as FactoryEntry[]
    const round = within(render(<TaskTimeline entries={entries} error="" />).getByLabelText("Round 1"))

    expect(round.getByText("after merge")).toBeTruthy()
    expect(round.getByText("failed")).toBeTruthy()
    expect(round.getByText("spawn ENOENT").closest("details")!.open).toBe(false)
  })

  it("shows the review with verdict, cost and findings, open for the current round", () => {
    const view = render(<TaskTimeline entries={ENTRIES} error="" />)
    const second = within(view.getByLabelText("Round 2"))

    expect(second.getByText("rejected")).toBeTruthy()
    expect(second.getByText("$0.25 · 4.0s")).toBeTruthy()
    expect(second.getByText("src/a.ts:12")).toBeTruthy()
    expect(second.getByText("blocker")).toBeTruthy()
    expect(second.getByText("Leaks the handle").closest("details")!.open).toBe(true)
  })

  it("shows the merge result in the newest round", () => {
    const merged = [
      ...ENTRIES,
      { id: "e11", timestamp: "", customType: "factory-merge", data: { ok: false, kind: "merge", commit: "", error: "rebase conflict", depsSynced: false } },
    ] as unknown as FactoryEntry[]
    const second = within(render(<TaskTimeline entries={merged} error="" />).getByLabelText("Round 2"))

    expect(second.getByText("merge")).toBeTruthy()
    expect(second.getByText("failed")).toBeTruthy()
    expect(second.getByText("rebase conflict").closest("details")!.open).toBe(false)
  })

  it("shows nothing but the error for a task without entries", () => {
    const view = render(<TaskTimeline entries={[]} error="Could not load the timeline" />)

    expect(view.queryByLabelText("Totals")).toBeNull()
    expect(view.getByRole("alert").textContent).toBe("Could not load the timeline")
  })
})
