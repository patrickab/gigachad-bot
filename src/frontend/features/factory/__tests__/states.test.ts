import { describe, it, expect } from "vitest"
import { allowedActions } from "@/features/factory/states"
import type { TaskState } from "@/lib/domain/types"

describe("allowedActions", () => {
  const none = { start: false, approve: false, retry: false, cancel: false, message: false, restartPreview: false }
  it.each<[TaskState, Partial<typeof none>]>([
    ["backlog", { start: true, cancel: true }],
    ["human_review", { approve: true, cancel: true, message: true, restartPreview: true }],
    ["blocked", { retry: true, cancel: true, message: true }],
    ["todo", { cancel: true }],
    ["in_progress", { cancel: true }],
    ["rework", { cancel: true }],
    ["merging", { cancel: true }],
    ["done", {}],
    ["cancelled", {}],
  ])("offers the right commands for %s", (state, allowed) => {
    expect(allowedActions(state)).toEqual({ ...none, ...allowed })
  })
})
