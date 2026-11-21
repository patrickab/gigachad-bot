import { describe, it, expect } from "vitest"
import { BOARD_COLUMNS, LOST_AFTER_MS, STATE_LABELS, STATE_ORDER, allowedActions, shownState, statusTone, taskStatusLine, taskTone, type FactoryShownState } from "@/features/factory/states"
import type { FactoryLive, FactoryTaskState } from "@/lib/domain/factory"

const LIVE: FactoryLive = { running: false, waiting: false, attached: 0 }

function task(state: FactoryTaskState | null, live: FactoryLive | null = LIVE, round = 1) {
  return { factory: state === null ? null : { state, round, reason: "", at: "2026-01-01T00:00:00Z" }, live }
}

describe("shownState", () => {
  it("shows a task without a factory state as starting", () => {
    expect(shownState(task(null, null))).toBe("starting")
    expect(shownState(task(null))).toBe("starting")
  })

  it.each<FactoryTaskState>(["running", "merging"])("shows a %s task without a live child as interrupted", (state) => {
    expect(shownState(task(state, null))).toBe("interrupted")
    expect(shownState(task(state))).toBe(state)
  })

  it.each<FactoryTaskState>(["waiting_approve", "blocked", "done", "cancelled"])("shows a %s task as it is, with or without a live child", (state) => {
    expect(shownState(task(state, null))).toBe(state)
    expect(shownState(task(state))).toBe(state)
  })
})

describe("board order", () => {
  it("puts what waits on the user before what runs, and that before what is finished", () => {
    expect(STATE_ORDER).toEqual(["waiting_approve", "blocked", "interrupted", "starting", "running", "merging", "done", "cancelled"])
    expect(BOARD_COLUMNS.map((column) => column.title)).toEqual(["Needs you", "Working", "Merging", "Finished"])
  })

  it("gives every shown state a label and exactly one column, in the same order", () => {
    expect(BOARD_COLUMNS.flatMap((column) => column.states)).toEqual(STATE_ORDER)
    expect(Object.keys(STATE_LABELS).sort()).toEqual([...STATE_ORDER].sort())
  })
})

describe("allowedActions", () => {
  const none = { approve: false, retry: false, cancel: false, message: false, delete: false }
  it.each<[FactoryShownState, Partial<typeof none>]>([
    ["starting", {}],
    ["running", { cancel: true, message: true }],
    ["interrupted", { retry: true, message: true, delete: true }],
    ["waiting_approve", { approve: true, cancel: true, message: true }],
    ["blocked", { retry: true, message: true, delete: true }],
    ["merging", { cancel: true, message: true }],
    ["done", { message: true, delete: true }],
    ["cancelled", { retry: true, message: true, delete: true }],
  ])("%s", (state, allowed) => {
    expect(allowedActions(state)).toEqual({ ...none, ...allowed })
  })

  it("deletes a starting task only once it is older than a minute", () => {
    const createdAt = "2026-01-01T00:00:00Z"
    const created = Date.parse(createdAt)
    expect(allowedActions("starting", createdAt, created + LOST_AFTER_MS).delete).toBe(false)
    expect(allowedActions("starting", createdAt, created + LOST_AFTER_MS + 1)).toEqual({ ...none, delete: true })
    expect(allowedActions("running", createdAt, created + 2 * LOST_AFTER_MS).delete).toBe(false)
  })
})

describe("task status", () => {
  it("names the state, and the round once one started", () => {
    expect(taskStatusLine(task(null, null))).toBe("Starting")
    expect(taskStatusLine(task("running", LIVE, 0))).toBe("Working")
    expect(taskStatusLine(task("waiting_approve", LIVE, 2))).toBe("Ready for review · round 2")
    expect(taskStatusLine(task("merging", null, 3))).toBe("Interrupted · round 3")
  })

  it("pulses the dot only while the Pi child runs a turn", () => {
    expect(statusTone("running")).toBe("active")
    expect(taskTone(task("running", { ...LIVE, running: true }))).toBe("active")
    expect(taskTone(task("running"))).toBe("ready")
    expect(taskTone(task(null, null))).toBe("ready")
    expect(taskTone(task("running", null))).toBe("error")
    expect(taskTone(task("blocked", null))).toBe("error")
    expect(taskTone(task("waiting_approve"))).toBe("attention")
    expect(taskTone(task("done", null))).toBe("done")
    expect(taskTone(task("cancelled", null))).toBe("idle")
  })
})
