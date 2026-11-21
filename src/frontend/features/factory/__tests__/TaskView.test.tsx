import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import type { FactoryEntry, FactoryPreview, FactoryTask, FactoryTaskCommand, FactoryTaskDiff, FactoryTaskState } from "@/lib/domain/factory"
import type { PiCatalogProps } from "@/features/pi/PiTopBar"

const api = vi.hoisted(() => {
  class ApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message)
    }
  }
  return {
    ApiError,
    commandFactoryTask: vi.fn<(id: string, command: FactoryTaskCommand) => Promise<unknown>>(),
    sendFactoryTaskMessage: vi.fn<(id: string, text: string) => Promise<unknown>>(),
    getFactoryTaskDiff: vi.fn<(id: string) => Promise<FactoryTaskDiff>>(),
    listFactoryEntries: vi.fn<(sessionId: string, since?: string) => Promise<FactoryEntry[]>>(),
    getFactoryPreview: vi.fn<(id: string) => Promise<FactoryPreview | null>>(),
    startFactoryPreview: vi.fn(),
    stopFactoryPreview: vi.fn(),
  }
})
/** What the mocked stores answer: the task of the repo's list and the Pi session rows. */
const store = vi.hoisted(() => ({
  task: undefined as FactoryTask | undefined,
  sessions: [] as { sessionId: string; modifiedAt: string }[],
  listedRepos: [] as string[],
}))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/factory/useFactoryTasks", () => ({
  useFactoryTasks: (repo: string) => {
    store.listedRepos.push(repo)
    return { tasks: [], error: null, refresh: async () => {} }
  },
  useFactoryTask: () => store.task,
}))
vi.mock("@/features/pi/sessionStore", () => ({ usePiSessions: () => ({ sessions: store.sessions, loading: false, error: null }) }))
vi.mock("@/features/factory/AgentPanel", () => ({
  AgentPanel: ({ entries }: { entries: readonly FactoryEntry[] }) => <div data-testid="agent-panel">{entries.length} entries</div>,
}))

import { TaskView, taskCommands } from "@/features/factory/TaskView"
import { shownState, type FactoryShownState } from "@/features/factory/states"

const LIVE = { running: false, waiting: false, attached: 0 }

function task(state: FactoryTaskState | null, overrides: Partial<FactoryTask> = {}): FactoryTask {
  return {
    id: "t1", repo: "/home/u/git/demo", session_id: "s1", session_path: "/sessions/s1.jsonl", worktree: "/home/u/git/worktrees/demo/t1", branch: "factory/fix-header",
    title: "Fix the header", created_at: "2025-01-01T00:00:00Z",
    factory: state && { state, round: 2, reason: "", at: "2025-01-01T00:01:00Z" },
    live: LIVE,
    ...overrides,
  }
}

const DIFF = (name: string) => `diff --git a/${name} b/${name}\n--- a/${name}\n+++ b/${name}\n@@ -1 +1 @@\n-old\n+new\n`
const ROUND: FactoryEntry = { id: "e1", timestamp: "", customType: "factory-round", data: { round: 1, trigger: "start" } }
const GATE: FactoryEntry = { id: "e2", timestamp: "", customType: "factory-gate", data: { round: 1, phase: "gates", argv: ["check"], exitCode: 0, killed: false, durationMs: 1, outputTail: "" } }

function view() {
  return <TaskView repo="/home/u/git/demo" taskId="t1" onBack={vi.fn()} modelProps={{} as PiCatalogProps} />
}

const buttons = () => screen.queryAllByRole("button").map((button) => button.textContent).filter((label) => ["Approve", "Retry", "Cancel"].includes(label ?? ""))

beforeEach(() => {
  store.task = undefined
  store.sessions = []
  store.listedRepos = []
  api.commandFactoryTask.mockReset().mockResolvedValue({})
  api.getFactoryTaskDiff.mockReset().mockResolvedValue({ base: "abc", diff: "" })
  api.listFactoryEntries.mockReset().mockResolvedValue([])
  api.getFactoryPreview.mockReset().mockResolvedValue(null)
  api.sendFactoryTaskMessage.mockReset().mockResolvedValue({})
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => { stored.set(key, value) },
    removeItem: (key: string) => { stored.delete(key) },
  })
})
afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
  delete (Range.prototype as Partial<Range>).getBoundingClientRect
  vi.unstubAllGlobals()
})

describe("taskCommands", () => {
  it("offers the commands that fit each shown state", () => {
    const offered = (state: FactoryShownState) => Object.entries(taskCommands(state)).filter(([, allowed]) => allowed).map(([command]) => command)

    expect(offered("starting")).toEqual([])
    expect(offered("running")).toEqual(["cancel"])
    expect(offered("merging")).toEqual(["cancel"])
    expect(offered("waiting_approve")).toEqual(["approve", "cancel"])
    expect(offered("blocked")).toEqual(["retry"])
    expect(offered("cancelled")).toEqual(["retry"])
    expect(offered("interrupted")).toEqual(["retry"])
    expect(offered("done")).toEqual([])
  })
})

describe("TaskView", () => {
  it("keeps the repo's task list read and shows a loading state until it has the task", () => {
    render(view())

    expect(store.listedRepos).toContain("/home/u/git/demo")
    expect(screen.getByText("Loading task")).toBeTruthy()
    expect(screen.queryByTestId("agent-panel")).toBeNull()
    expect(api.listFactoryEntries).not.toHaveBeenCalled()
  })

  it("shows title, state, round and branch, or the current checkout", async () => {
    store.task = task("running")
    const first = render(view())
    expect(screen.getByRole("heading", { name: "Fix the header" })).toBeTruthy()
    expect(screen.getByLabelText("Task state").textContent).toBe("Working · round 2 · factory/fix-header")
    await waitFor(() => expect(screen.getByText("No changes")).toBeTruthy())
    first.unmount()

    store.task = task("done", { worktree: null, branch: null })
    render(view())
    expect(screen.getByLabelText("Task state").textContent).toBe("Done · round 2 · current checkout")
  })

  it("approves a task that waits for approval", async () => {
    store.task = task("waiting_approve")
    render(view())
    expect(buttons()).toEqual(["Approve", "Cancel"])

    fireEvent.click(screen.getByRole("button", { name: "Approve" }))
    await waitFor(() => expect(api.commandFactoryTask).toHaveBeenCalledWith("t1", "approve"))
  })

  it("retries a blocked or cancelled task", async () => {
    store.task = task("blocked")
    const first = render(view())
    expect(buttons()).toEqual(["Retry"])
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(api.commandFactoryTask).toHaveBeenCalledWith("t1", "retry"))
    first.unmount()

    store.task = task("cancelled")
    render(view())
    expect(buttons()).toEqual(["Retry"])
  })

  it("cancels a running or merging task after a confirmation", async () => {
    store.task = task("merging")
    const first = render(view())
    expect(buttons()).toEqual(["Cancel"])
    first.unmount()

    store.task = task("running")
    render(view())
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    expect(api.commandFactoryTask).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Yes" }))

    await waitFor(() => expect(api.commandFactoryTask).toHaveBeenCalledWith("t1", "cancel"))
    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy())
  })

  it("offers nothing for a finished task, and shows why a command failed", async () => {
    store.task = task("done")
    const first = render(view())
    expect(buttons()).toEqual([])
    first.unmount()

    api.commandFactoryTask.mockRejectedValue(new Error("prompt_failed"))
    store.task = task("blocked")
    render(view())
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("prompt_failed"))
  })

  it("shows a task without a factory state as starting, without commands", () => {
    store.task = task(null)
    render(view())

    expect(shownState(store.task)).toBe("starting")
    expect(screen.getByLabelText("Task state").textContent).toBe("Starting · factory/fix-header")
    expect(buttons()).toEqual([])
  })

  it("shows a running task without a Pi child as interrupted and offers a retry", () => {
    store.task = task("running", { live: null })
    render(view())

    expect(screen.getByLabelText("Task state").textContent).toBe("Interrupted · round 2 · factory/fix-header")
    expect(buttons()).toEqual(["Retry"])
  })

  it("reads the timeline again when the session file grows and the diff again when the timeline grew", async () => {
    store.task = task("running")
    store.sessions = [{ sessionId: "s1", modifiedAt: "2025-01-01T00:01:00Z" }, { sessionId: "other", modifiedAt: "2025-01-01T00:00:00Z" }]
    api.listFactoryEntries.mockResolvedValueOnce([ROUND])
    api.getFactoryTaskDiff.mockResolvedValue({ base: "abc", diff: DIFF("a.ts") })
    const rendered = render(view())
    await waitFor(() => expect(screen.getByTestId("agent-panel").textContent).toBe("1 entries"))
    await waitFor(() => expect(screen.getByLabelText("Diff for a.ts")).toBeTruthy())
    expect(api.listFactoryEntries).toHaveBeenCalledTimes(1)
    const diffReads = api.getFactoryTaskDiff.mock.calls.length

    // Another session changed: neither the timeline nor the diff is read
    store.sessions = [store.sessions[0], { sessionId: "other", modifiedAt: "2025-01-01T00:05:00Z" }]
    rendered.rerender(view())
    expect(api.listFactoryEntries).toHaveBeenCalledTimes(1)

    // The task's session changed without a new factory entry: the timeline is read, the diff stays
    api.listFactoryEntries.mockResolvedValueOnce([])
    store.sessions = [{ sessionId: "s1", modifiedAt: "2025-01-01T00:02:00Z" }]
    rendered.rerender(view())
    await waitFor(() => expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", "e1"))
    expect(api.getFactoryTaskDiff).toHaveBeenCalledTimes(diffReads)

    // A gate ran: the timeline grows and the diff follows
    api.listFactoryEntries.mockResolvedValueOnce([GATE])
    api.getFactoryTaskDiff.mockResolvedValue({ base: "abc", diff: DIFF("a.ts") + DIFF("b.ts") })
    store.sessions = [{ sessionId: "s1", modifiedAt: "2025-01-01T00:03:00Z" }]
    rendered.rerender(view())
    await waitFor(() => expect(screen.getByTestId("agent-panel").textContent).toBe("2 entries"))
    await waitFor(() => expect(screen.getByLabelText("Diff for b.ts")).toBeTruthy())
    expect(api.getFactoryTaskDiff).toHaveBeenCalledTimes(diffReads + 1)
  })

  it("reads the timeline again when the task's factory state moves", async () => {
    store.task = task("running")
    api.listFactoryEntries.mockResolvedValueOnce([ROUND])
    const rendered = render(view())
    await waitFor(() => expect(api.listFactoryEntries).toHaveBeenCalledTimes(1))

    store.task = { ...store.task, factory: { state: "blocked", round: 2, reason: "stopped", at: "2025-01-01T00:09:00Z" } }
    rendered.rerender(view())
    await waitFor(() => expect(api.listFactoryEntries).toHaveBeenLastCalledWith("s1", "e1"))
  })

  it("shows the preview on demand instead of the diff", async () => {
    store.task = task("waiting_approve")
    render(view())
    expect(api.getFactoryPreview).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("tab", { name: "Preview" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "Start preview" })).toBeTruthy())
    expect(api.getFactoryPreview).toHaveBeenCalledWith("t1")
  })
})

/** Selects the diff line showing `text` and comments `comment` on it, as the user does. */
async function comment(text: string, note: string) {
  const code = await within(screen.getByLabelText("Diff")).findByText(text)
  const range = document.createRange()
  range.selectNodeContents(code)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
  fireEvent(document, new Event("selectionchange"))
  fireEvent.click(screen.getByRole("button", { name: "Add comment to selection" }))
  const editor = screen.getByLabelText(/^Note on /)
  fireEvent.change(editor, { target: { value: note } })
  fireEvent.keyDown(editor, { key: "Enter" })
}

const sendReview = () => screen.queryByRole("button", { name: /^Send review/ })

describe("TaskView review", () => {
  beforeEach(() => {
    Object.defineProperty(Range.prototype, "getBoundingClientRect", { configurable: true, value: () => ({ left: 100, bottom: 100, height: 10 }) })
    api.getFactoryTaskDiff.mockResolvedValue({ base: "abc", diff: DIFF("a.txt") })
  })

  it("shows a comment under its line and sends all pending comments as one message, then clears them", async () => {
    store.task = task("waiting_approve")
    render(view())
    expect(sendReview()).toBeNull()
    await comment("new", "Use a constant")
    expect(within(screen.getByLabelText("Diff")).getByText("Use a constant")).toBeTruthy()
    expect(sendReview()?.textContent).toBe("Send review 1")

    fireEvent.click(sendReview()!)
    await waitFor(() => expect(sendReview()).toBeNull())
    expect(api.sendFactoryTaskMessage).toHaveBeenCalledWith("t1", "Review of the diff:\n\na.txt:1\n```\nnew\n```\nUse a constant")
    expect(screen.queryByText("Use a constant")).toBeNull()
  })

  it("keeps the comments and shows the error when the send fails", async () => {
    api.sendFactoryTaskMessage.mockRejectedValue(new Error("task_busy"))
    store.task = task("blocked")
    render(view())
    await comment("new", "Rename it")
    fireEvent.click(sendReview()!)
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("task_busy"))
    expect(sendReview()?.textContent).toBe("Send review 1")
    expect(screen.getByText("Rename it")).toBeTruthy()
  })

  it("does not send while the task merges", async () => {
    store.task = task("merging")
    render(view())
    await comment("new", "Later")
    expect(sendReview()).toBeDisabled()
    fireEvent.click(sendReview()!)
    expect(api.sendFactoryTaskMessage).not.toHaveBeenCalled()
  })

  it("keeps a task's comments across leaving and reopening it, and sends one whose line left the diff with its quoted line", async () => {
    store.task = task("waiting_approve")
    const first = render(view())
    await comment("new", "Keep this")
    first.unmount()

    api.getFactoryTaskDiff.mockResolvedValue({ base: "abc", diff: DIFF("b.txt") })
    render(view())
    expect(await screen.findByText("a.txt · no longer in diff")).toBeTruthy()
    expect(screen.getByText("Keep this")).toBeTruthy()
    fireEvent.click(sendReview()!)
    await waitFor(() => expect(api.sendFactoryTaskMessage).toHaveBeenCalledWith("t1", "Review of the diff:\n\na.txt:1\n```\nnew\n```\nKeep this"))
  })

  it("discards the comments when the shown task finishes", async () => {
    store.task = task("waiting_approve")
    const shown = render(view())
    await comment("new", "Too late")
    store.task = task("done")
    shown.rerender(view())
    await waitFor(() => expect(sendReview()).toBeNull())
  })
})
