import { describe, it, expect } from "vitest"
import { BOARD_COLUMNS, STATE_LABELS, shownState, statusTone, taskStatusLine, taskTone } from "@/features/factory/states"
import type { FactoryTaskState } from "@/lib/domain/factory"

function task(state: FactoryTaskState | null, { running = false, interrupted = false, round = 1 }: { running?: boolean; interrupted?: boolean; round?: number } = {}) {
  return { state, round: state === null ? null : round, running, interrupted }
}

describe("shownState", () => {
  it("shows a task without a factory state as starting", () => {
    expect(shownState(task(null))).toBe("starting")
    expect(shownState(task(null, { running: true }))).toBe("starting")
  })

  it.each<FactoryTaskState>(["running", "merging"])("shows a %s task the core reports interrupted as interrupted", (state) => {
    expect(shownState(task(state, { interrupted: true }))).toBe("interrupted")
    expect(shownState(task(state, { running: true }))).toBe(state)
  })

  it.each<FactoryTaskState>(["waiting_approve", "waiting_answer", "blocked", "reviewing", "done", "cancelled"])("shows a %s task as it is", (state) => {
    expect(shownState(task(state))).toBe(state)
  })
})

describe("board columns", () => {
  it("gives every shown state a label and exactly one column", () => {
    const inColumns = BOARD_COLUMNS.flatMap((column) => column.states)
    expect(new Set(inColumns).size).toBe(inColumns.length)
    expect([...inColumns].sort()).toEqual(Object.keys(STATE_LABELS).sort())
  })
})

describe("task status", () => {
  it("names the state, and the round once one started", () => {
    expect(taskStatusLine(task(null))).toBe("Launching session")
    expect(taskStatusLine(task("running", { round: 0 }))).toBe("Working")
    expect(taskStatusLine(task("waiting_approve", { round: 2 }))).toBe("Ready for review · round 2")
    expect(taskStatusLine(task("merging", { round: 3, interrupted: true }))).toBe("Interrupted · round 3")
    expect(taskStatusLine(task("waiting_answer", { round: 1 }))).toBe("Waiting for supervisor · round 1")
  })

  it("pulses the dot only while a runner holds the task", () => {
    expect(statusTone("running")).toBe("active")
    expect(taskTone(task("running", { running: true }))).toBe("active")
    expect(taskTone(task(null, { running: true }))).toBe("active")
    expect(taskTone(task(null))).toBe("ready")
    expect(taskTone(task("running", { interrupted: true }))).toBe("error")
    expect(taskTone(task("blocked"))).toBe("error")
    expect(taskTone(task("waiting_approve"))).toBe("attention")
    expect(taskTone(task("waiting_answer"))).toBe("attention")
    expect(taskTone(task("done"))).toBe("done")
    expect(taskTone(task("cancelled"))).toBe("idle")
  })
})
