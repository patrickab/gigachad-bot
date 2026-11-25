import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import type { FactoryPreview, FactoryTask, FactoryTaskAction, FactoryTaskState, FactoryTimeline, FactoryTimelineEntry } from "@/lib/domain/factory"
import { factoryTask } from "./factoryTask"

const core = vi.hoisted(() => ({
  command: vi.fn<(action: string, id: string) => Promise<unknown>>(),
  message: vi.fn<(id: string, text: string) => Promise<unknown>>(),
  diff: vi.fn<(id: string) => Promise<string>>(),
  timeline: vi.fn<(id: string, since: number) => Promise<FactoryTimeline>>(),
}))
const api = vi.hoisted(() => ({
  callFactory: vi.fn(async (action: string, params: Record<string, unknown>) => {
    const id = params.id as string
    if (action === "tasks.message") return core.message(id, params.text as string)
    if (action === "tasks.diff") {
      const diff = await core.diff(id)
      return { base: "abc", diff, files: [...diff.matchAll(/^\+\+\+ b\/(.+)$/gm)].map(([, path]) => ({ path, added: 1, removed: 1 })) }
    }
    if (action === "tasks.timeline") return core.timeline(id, params.since as number)
    if (["tasks.approve", "tasks.retry", "tasks.cancel"].includes(action)) return core.command(action, id)
    throw new Error(`unexpected ${action}`)
  }),
  getFactoryPreview: vi.fn<(id: string) => Promise<FactoryPreview | null>>(),
  startFactoryPreview: vi.fn(),
  stopFactoryPreview: vi.fn(),
}))
/** What the mocked task store answers: the task of the repo's list. */
const store = vi.hoisted(() => ({
  task: undefined as FactoryTask | undefined,
  listedRepos: [] as string[],
  refreshes: 0,
}))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/factory/useFactoryTasks", () => ({
  useFactoryTasks: (repo: string) => {
    store.listedRepos.push(repo)
    return { tasks: [], error: null, refresh: async () => { store.refreshes += 1 } }
  },
  useFactoryTask: () => store.task,
}))
vi.mock("@/features/factory/AgentPanel", () => ({
  AgentPanel: ({ entries, canMessage }: { entries: readonly FactoryTimelineEntry[]; canMessage: boolean }) =>
    <div data-testid="agent-panel" data-can-message={String(canMessage)}>{entries.length} entries</div>,
}))

import { TaskView, taskCommands } from "@/features/factory/TaskView"
import { shownState } from "@/features/factory/states"

/** What the core lists as `actions` for a task in each state, as the fixture's server answers. */
const CORE_ACTIONS: Record<FactoryTaskState | "none", FactoryTaskAction[]> = {
  none: ["tasks.cancel", "tasks.remove"],
  running: ["tasks.cancel", "tasks.message"],
  merging: ["tasks.cancel"],
  waiting_approve: ["tasks.approve", "tasks.cancel", "tasks.remove", "tasks.message"],
  waiting_answer: ["tasks.message", "tasks.cancel"],
  blocked: ["tasks.retry", "tasks.cancel", "tasks.remove", "tasks.message"],
  done: ["tasks.remove", "tasks.message"],
  cancelled: ["tasks.retry", "tasks.remove", "tasks.message"],
}

function task(state: FactoryTaskState | null, overrides: Partial<FactoryTask> = {}): FactoryTask {
  return factoryTask("t1", {
    repo: "/home/u/git/demo", worktree: "/home/u/git/worktrees/demo/t1", branch: "factory/fix-header", title: "Fix the header",
    state, round: state === null ? null : 2, at: state === null ? null : "2025-01-01T00:01:00Z", running: state === "running" || state === "merging",
    actions: CORE_ACTIONS[state ?? "none"],
    ...overrides,
  })
}

const DIFF = (name: string) => `diff --git a/${name} b/${name}\n--- a/${name}\n+++ b/${name}\n@@ -1 +1 @@\n-old\n+new\n`
const ROUND: FactoryTimelineEntry = { at: "", kind: "round", data: { round: 1, trigger: "start" } }
const GATE: FactoryTimelineEntry = { at: "", kind: "gate", data: { round: 1, phase: "gates", argv: ["check"], exitCode: 0, killed: false, durationMs: 1, outputTail: "" } }

function view() {
  return <TaskView repo="/home/u/git/demo" taskId="t1" onBack={vi.fn()} onOpenTask={vi.fn()} />
}

const buttons = () => screen.queryAllByRole("button").map((button) => button.textContent).filter((label) => ["Approve", "Retry", "Cancel"].includes(label ?? ""))

beforeEach(() => {
  store.task = undefined
  store.listedRepos = []
  store.refreshes = 0
  api.callFactory.mockClear()
  core.command.mockReset().mockResolvedValue({})
  core.diff.mockReset().mockResolvedValue("")
  core.timeline.mockReset().mockResolvedValue({ entries: [], next: 0 })
  api.getFactoryPreview.mockReset().mockResolvedValue(null)
  core.message.mockReset().mockResolvedValue({})
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
  it("offers exactly the commands among the core's actions", () => {
    const offered = (actions: FactoryTaskAction[]) => Object.entries(taskCommands(actions)).filter(([, allowed]) => allowed).map(([command]) => command)

    expect(offered([])).toEqual([])
    expect(offered(["tasks.remove", "tasks.message"])).toEqual([])
    expect(offered(["tasks.cancel"])).toEqual(["cancel"])
    expect(offered(["tasks.approve", "tasks.cancel", "tasks.remove"])).toEqual(["approve", "cancel"])
    expect(offered(["tasks.retry", "tasks.cancel"])).toEqual(["retry", "cancel"])
  })
})

describe("TaskView", () => {
  it("keeps the repo's task list read and shows a loading state until it has the task", async () => {
    render(view())

    expect(store.listedRepos).toContain("/home/u/git/demo")
    expect(screen.getByText("Loading task")).toBeTruthy()
    expect(screen.queryByTestId("agent-panel")).toBeNull()
    // The timeline is read by the task's id, from the start
    await waitFor(() => expect(core.timeline).toHaveBeenCalledWith("t1", 0))
  })

  it("shows title, state, round and branch", async () => {
    store.task = task("running")
    render(view())
    expect(screen.getByRole("heading", { name: "Fix the header" })).toBeTruthy()
    expect(screen.getByLabelText("Task state").textContent).toBe("Working · round 2 · factory/fix-header")
    await waitFor(() => expect(screen.getByText("No changes")).toBeTruthy())
  })

  it("approves a task that waits for approval", async () => {
    store.task = task("waiting_approve")
    render(view())
    expect(buttons()).toEqual(["Approve", "Cancel"])

    fireEvent.click(screen.getByRole("button", { name: "Approve" }))
    await waitFor(() => expect(core.command).toHaveBeenCalledWith("tasks.approve", "t1"))
    await waitFor(() => expect(store.refreshes).toBe(1))
  })

  it("retries a blocked or cancelled task", async () => {
    store.task = task("blocked")
    const first = render(view())
    expect(buttons()).toEqual(["Retry", "Cancel"])
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(core.command).toHaveBeenCalledWith("tasks.retry", "t1"))
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
    expect(core.command).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Yes" }))

    await waitFor(() => expect(core.command).toHaveBeenCalledWith("tasks.cancel", "t1"))
    await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy())
  })

  it("offers nothing for a finished task, and shows why a command failed", async () => {
    store.task = task("done")
    const first = render(view())
    expect(buttons()).toEqual([])
    first.unmount()

    core.command.mockRejectedValue(new Error("the runner of task t1 is still ending, try again in a few seconds"))
    store.task = task("blocked")
    render(view())
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("the runner of task t1 is still ending, try again in a few seconds"))
  })

  it("shows a task without a factory state as starting, with the core's commands, without a diff read or a message box", async () => {
    store.task = task(null)
    render(view())

    expect(shownState(store.task)).toBe("starting")
    expect(screen.getByLabelText("Task state").textContent).toBe("Launching session · factory/fix-header")
    expect(buttons()).toEqual(["Cancel"])
    expect(screen.getByTestId("agent-panel").dataset.canMessage).toBe("false")
    await waitFor(() => expect(screen.getByText("No changes")).toBeTruthy())
    expect(core.diff).not.toHaveBeenCalled()
  })

  it("shows a task the core reports interrupted as such and offers retry and cancel", () => {
    store.task = task("running", { running: false, interrupted: true, actions: ["tasks.retry", "tasks.cancel", "tasks.message"] })
    render(view())

    expect(screen.getByLabelText("Task state").textContent).toBe("Interrupted · round 2 · factory/fix-header")
    expect(buttons()).toEqual(["Retry", "Cancel"])
  })

  it("offers what the core lists whatever the state, and no message the core does not accept", () => {
    store.task = task("running", { actions: ["tasks.approve"] })
    render(view())
    expect(buttons()).toEqual(["Approve"])
    expect(screen.getByTestId("agent-panel").dataset.canMessage).toBe("false")
  })

  it("lets the user message the coder unless the task merges", () => {
    store.task = task("running")
    const first = render(view())
    expect(screen.getByTestId("agent-panel").dataset.canMessage).toBe("true")
    first.unmount()

    store.task = task("merging")
    render(view())
    expect(screen.getByTestId("agent-panel").dataset.canMessage).toBe("false")
  })

  it("reads the timeline on from its cursor when the task moves, and the diff again when the timeline grew", async () => {
    store.task = task("blocked")
    core.timeline.mockResolvedValueOnce({ entries: [ROUND], next: 100 })
    core.diff.mockResolvedValue(DIFF("a.ts"))
    const rendered = render(view())
    await waitFor(() => expect(screen.getByTestId("agent-panel").textContent).toBe("1 entries"))
    await waitFor(() => expect(screen.getByLabelText("Diff for a.ts")).toBeTruthy())
    expect(core.timeline).toHaveBeenCalledTimes(1)
    const diffReads = core.diff.mock.calls.length

    // The list answered the same task again: nothing is read
    store.task = { ...store.task! }
    rendered.rerender(view())
    expect(core.timeline).toHaveBeenCalledTimes(1)

    // The task's usage moved without a new entry: the timeline is read, the diff stays
    core.timeline.mockResolvedValueOnce({ entries: [], next: 100 })
    store.task = { ...store.task, usage: { input: 1, output: 2, cost: 0.1 } }
    rendered.rerender(view())
    await waitFor(() => expect(core.timeline).toHaveBeenLastCalledWith("t1", 100))
    expect(core.diff).toHaveBeenCalledTimes(diffReads)

    // A gate ran: the timeline grows and the diff follows
    core.timeline.mockResolvedValueOnce({ entries: [GATE], next: 200 })
    core.diff.mockResolvedValue(DIFF("a.ts") + DIFF("b.ts"))
    store.task = { ...store.task, state: "waiting_approve", at: "2025-01-01T00:09:00Z" }
    rendered.rerender(view())
    await waitFor(() => expect(screen.getByTestId("agent-panel").textContent).toBe("2 entries"))
    await waitFor(() => expect(screen.getByLabelText("Diff for b.ts")).toBeTruthy())
    expect(core.diff).toHaveBeenCalledTimes(diffReads + 1)
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
    core.diff.mockResolvedValue(DIFF("a.txt"))
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
    expect(core.message).toHaveBeenCalledWith("t1", "Review of the diff:\n\na.txt:1\n```\nnew\n```\nUse a constant")
    expect(screen.queryByText("Use a constant")).toBeNull()
  })

  it("keeps the comments and shows the error when the send fails", async () => {
    core.message.mockRejectedValue(new Error("task t1 is merging, send the message when the merge has ended"))
    store.task = task("blocked")
    render(view())
    await comment("new", "Rename it")
    fireEvent.click(sendReview()!)
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("task t1 is merging, send the message when the merge has ended"))
    expect(sendReview()?.textContent).toBe("Send review 1")
    expect(screen.getByText("Rename it")).toBeTruthy()
  })

  it("does not send while the task merges", async () => {
    store.task = task("merging")
    render(view())
    await comment("new", "Later")
    expect(sendReview()).toBeDisabled()
    fireEvent.click(sendReview()!)
    expect(core.message).not.toHaveBeenCalled()
  })

  it("keeps a task's comments across leaving and reopening it, and sends one whose line left the diff with its quoted line", async () => {
    store.task = task("waiting_approve")
    const first = render(view())
    await comment("new", "Keep this")
    first.unmount()

    core.diff.mockResolvedValue(DIFF("b.txt"))
    render(view())
    expect(await screen.findByText("a.txt · no longer in diff")).toBeTruthy()
    expect(screen.getByText("Keep this")).toBeTruthy()
    fireEvent.click(sendReview()!)
    await waitFor(() => expect(core.message).toHaveBeenCalledWith("t1", "Review of the diff:\n\na.txt:1\n```\nnew\n```\nKeep this"))
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
