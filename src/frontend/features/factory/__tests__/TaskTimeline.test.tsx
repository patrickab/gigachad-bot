import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, within } from "@testing-library/react"
import type { FactoryTimelineEntry } from "@/lib/domain/factory"
import { TaskTimeline } from "@/features/factory/TaskTimeline"

const USAGE = { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, cost: 0.25 }

const ENTRIES = [
  { at: "", kind: "round", data: { round: 1, trigger: "start" } },
  { at: "", kind: "state", data: { state: "running", round: 1, reason: "start" } },
  { at: "", kind: "gate", data: { round: 1, phase: "hooks", argv: ["ruff", "format"], exitCode: 0, killed: false, durationMs: 400, outputTail: "" } },
  { at: "", kind: "gate", data: { round: 1, phase: "gates", argv: ["factory/gates.sh"], exitCode: 2, killed: false, durationMs: 12000, outputTail: "1 failed" } },
  { at: "", kind: "round", data: { round: 2, trigger: "gates" } },
  { at: "", kind: "state", data: { state: "running", round: 2, reason: "gates" } },
  { at: "", kind: "gate", data: { round: 2, phase: "gates", argv: ["factory/gates.sh"], exitCode: null, killed: true, durationMs: 60000, outputTail: "" } },
  {
    at: "", kind: "review",
    data: { round: 2, status: "ok", approved: false, findings: [{ file: "src/a.ts", line: 12, severity: "blocker", text: "Leaks the handle" }], model: "haiku", sessionFile: "", usage: USAGE, wallMs: 4000 },
  },
  { at: "", kind: "state", data: { state: "blocked", round: 2, reason: "max_rounds" } },
] as unknown as FactoryTimelineEntry[]

describe("TaskTimeline", () => {
  it("shows the coder's turn in its round: its last message, the earlier ones collapsed, and what the user wrote", () => {
    const coder = (text: string) => ({ at: "", kind: "coder", data: { text } })
    const entries = [
      ENTRIES[0], coder("Reading the code."), coder("  "), coder("Added the toggle and a test."),
      { at: "", kind: "user", data: { text: "Also update the README." } },
      ...ENTRIES.slice(1),
    ] as unknown as FactoryTimelineEntry[]

    const view = render(<TaskTimeline entries={entries} usage={{ input: 4500, output: 605, cost: 1.75 }} error="" onOpenTask={vi.fn()} />)

    const first = within(view.getByLabelText("Round 1"))
    expect(first.getByText("coder")).toBeTruthy()
    expect(first.getByText("Added the toggle and a test.")).toBeTruthy()
    // A message with only tool calls says nothing: one earlier message, not two
    expect(first.getByText("1 earlier message")).toBeTruthy()
    expect(first.getByText("you")).toBeTruthy()
    expect(first.getByText("Also update the README.")).toBeTruthy()
    expect(view.getByLabelText("Totals").textContent).toBe("2 rounds · 4.5k in / 605 out · $1.75 (review $0.25)")
  })

  it("shows the coder's question in its round, then the answer", () => {
    const entries = [
      ENTRIES[0],
      { at: "", kind: "question", data: { text: "Keep the old flag?" } },
      { at: "", kind: "state", data: { state: "waiting_answer", round: 1, reason: "question" } },
      { at: "", kind: "user", data: { text: "Drop it." } },
    ] as unknown as FactoryTimelineEntry[]

    const first = within(render(<TaskTimeline entries={entries} usage={null} error="" onOpenTask={vi.fn()} />).getByLabelText("Round 1"))

    expect(first.getByText("question")).toBeTruthy()
    expect(first.getByText("Keep the old flag?")).toBeTruthy()
    expect(first.getByText("Drop it.")).toBeTruthy()
    expect(first.getByText("Waiting for supervisor")).toBeTruthy()
  })

  it("names a merge request's source and target branch, nothing for a coder's task", () => {
    const settings = { merge: "auto", maxRounds: 3, review: true, models: {}, branch: "master" }
    const task = { at: "", kind: "task", data: { title: "M", request: "", settings, branch: "wave/w1", mainBranch: "master", baseSha: "a", kind: "merge_request", source: "wave/w1" } }
    const view = render(<TaskTimeline entries={[task] as unknown as FactoryTimelineEntry[]} usage={null} error="" onOpenTask={vi.fn()} />)

    expect(view.getByLabelText("Merge request")).toHaveTextContent("merge requestwave/w1 → master")
    view.unmount()
    expect(render(<TaskTimeline entries={ENTRIES} usage={null} error="" onOpenTask={vi.fn()} />).queryByLabelText("Merge request")).toBeNull()
  })

  it("shows the totals and each round with its trigger and status", () => {
    const view = render(<TaskTimeline entries={ENTRIES} usage={{ input: 10, output: 5, cost: 0.25 }} error="" onOpenTask={vi.fn()} />)

    expect(view.getByLabelText("Totals").textContent).toBe("2 rounds · 10 in / 5 out · $0.25 (review $0.25)")
    const first = within(view.getByLabelText("Round 1"))
    expect(first.getByText("start")).toBeTruthy()
    // Round 1 was ended by round 2: it is not shown as still working
    expect(first.queryByText("Working")).toBeNull()
    const second = within(view.getByLabelText("Round 2"))
    expect(second.getByText("after failed gates")).toBeTruthy()
    expect(second.getByText("Blocked · max_rounds")).toBeTruthy()
  })

  it("shows each gate with its result and duration, a failed one's output collapsed", () => {
    const view = render(<TaskTimeline entries={ENTRIES} usage={null} error="" onOpenTask={vi.fn()} />)
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
      { at: "", kind: "gate", data: { round: 1, phase: "after_merge", argv: ["uv", "sync"], exitCode: null, killed: false, durationMs: 100, outputTail: "spawn ENOENT" } },
    ] as unknown as FactoryTimelineEntry[]
    const round = within(render(<TaskTimeline entries={entries} usage={null} error="" onOpenTask={vi.fn()} />).getByLabelText("Round 1"))

    expect(round.getByText("after merge")).toBeTruthy()
    expect(round.getByText("failed")).toBeTruthy()
    expect(round.getByText("spawn ENOENT").closest("details")!.open).toBe(false)
  })

  it("shows the review with verdict, cost and findings, open for the current round", () => {
    const view = render(<TaskTimeline entries={ENTRIES} usage={null} error="" onOpenTask={vi.fn()} />)
    const second = within(view.getByLabelText("Round 2"))

    expect(second.getByText("rejected")).toBeTruthy()
    expect(second.getByText("10 in / 5 out · $0.25 · 4.0s")).toBeTruthy()
    expect(second.getByText("src/a.ts:12")).toBeTruthy()
    expect(second.getByText("blocker")).toBeTruthy()
    expect(second.getByText("Leaks the handle").closest("details")!.open).toBe(true)
  })

  it("shows the merge result in the newest round", () => {
    const merged = [
      ...ENTRIES,
      { at: "", kind: "merge", data: { ok: false, commit: "", error: "rebase conflict", depsSynced: false } },
    ] as unknown as FactoryTimelineEntry[]
    const second = within(render(<TaskTimeline entries={merged} usage={null} error="" onOpenTask={vi.fn()} />).getByLabelText("Round 2"))

    expect(second.getByText("merge")).toBeTruthy()
    expect(second.getByText("failed")).toBeTruthy()
    expect(second.getByText("rebase conflict").closest("details")!.open).toBe(false)
  })

  it("names a task the coder started in the newest round and opens it on click", () => {
    const onOpenTask = vi.fn()
    const spawned = [
      ...ENTRIES,
      { at: "", kind: "spawn", data: { id: "t9", title: "fix(ui): align the header" } },
    ] as unknown as FactoryTimelineEntry[]
    const second = within(render(<TaskTimeline entries={spawned} usage={null} error="" onOpenTask={onOpenTask} />).getByLabelText("Round 2"))

    expect(second.getByText("new task")).toBeTruthy()
    fireEvent.click(second.getByRole("button", { name: "fix(ui): align the header" }))
    expect(onOpenTask).toHaveBeenCalledWith("t9")
  })

  it("leaves the usage out of the totals before a model answered", () => {
    const view = render(<TaskTimeline entries={ENTRIES.slice(0, 2)} usage={null} error="" onOpenTask={vi.fn()} />)
    expect(view.getByLabelText("Totals").textContent).toBe("1 round")
  })

  it("shows nothing but the error for a task without entries", () => {
    const view = render(<TaskTimeline entries={[]} usage={null} error="Could not load the timeline" onOpenTask={vi.fn()} />)

    expect(view.queryByLabelText("Totals")).toBeNull()
    expect(view.getByRole("alert").textContent).toBe("Could not load the timeline")
  })
})
