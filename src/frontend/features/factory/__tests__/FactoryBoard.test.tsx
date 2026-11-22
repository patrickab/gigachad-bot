import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor, within } from "@testing-library/react"
import type { FactoryLive, FactorySettings, FactoryTask, FactoryTaskState } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  deleteFactoryTask: vi.fn<(id: string) => Promise<void>>(),
  getFactorySettings: vi.fn<(repo: string) => Promise<FactorySettings>>(),
  createFactoryTask: vi.fn<(body: unknown) => Promise<unknown>>(),
  listFactoryTasks: vi.fn<(repo: string) => Promise<FactoryTask[]>>(),
  fetchModels: vi.fn<() => Promise<never>>(),
  listFactoryBranches: vi.fn<(repo: string) => Promise<string[]>>(),
  getFactoryConfig: vi.fn<(repo: string) => Promise<{ exists: boolean; draft: string | null }>>(),
  putFactoryConfig: vi.fn<(repo: string, content: string) => Promise<unknown>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { FactoryBoard } from "@/features/factory/FactoryBoard"
import { TopBarSlot, TopBarSlotProvider } from "@/features/navigation/TopBarSlots"

const REPO = "/home/me/git/gigachad-bot"

const IDLE: FactoryLive = { running: false, waiting: false, attached: 0 }

let minute = 0

/** A task named after its state, each one's state newer than the one made before it. */
function task(title: string, state: FactoryTaskState | null, extra: Partial<FactoryTask> & { reason?: string; round?: number } = {}): FactoryTask {
  const { reason = "", round = 1, ...fields } = extra
  minute += 1
  const at = `2026-01-01T00:${String(minute).padStart(2, "0")}:00Z`
  return {
    id: `id${title.toLowerCase().replace(/[^a-z0-9]/g, "")}000000`, repo: REPO, session_id: `session-${title}`, session_path: "", worktree: null, branch: null,
    title, created_at: at, factory: state === null ? null : { state, round, reason, at }, live: IDLE, usage: null, ...fields,
  }
}

function renderBoard(tasks: FactoryTask[], props: Partial<Parameters<typeof FactoryBoard>[0]> = {}) {
  const refresh = vi.fn(async () => {})
  const onOpenTask = vi.fn()
  // The title and the buttons render into the app's top bar, so the board is mounted under its slots
  const view = render(
    <TopBarSlotProvider>
      <TopBarSlot name="title">{(ref) => <div ref={ref} />}</TopBarSlot>
      <TopBarSlot name="center">{(ref) => <div ref={ref} />}</TopBarSlot>
      <FactoryBoard repo={REPO} tasks={tasks} error={null} refresh={refresh} onOpenTask={onOpenTask} {...props} />
    </TopBarSlotProvider>,
  )
  const titles = (column: string) => within(view.getByRole("region", { name: column })).queryAllByRole("button", { name: /^Task / }).map((card) => within(card).getByText(/^Task /).textContent)
  const card = (title: string) => view.getByText(title).closest("button")!.parentElement!
  return { view, refresh, onOpenTask, titles, card }
}

beforeEach(() => {
  minute = 0
  api.deleteFactoryTask.mockReset().mockResolvedValue()
  api.getFactorySettings.mockReset().mockResolvedValue({ merge: "review", max_rounds: 3, review: true, preview: false, models: {}, branch: null })
  api.fetchModels.mockReset().mockReturnValue(new Promise(() => {}))
  api.createFactoryTask.mockReset().mockResolvedValue({})
  api.listFactoryTasks.mockReset().mockResolvedValue([])
  api.listFactoryBranches.mockReset().mockResolvedValue(["main"])
  api.getFactoryConfig.mockReset().mockResolvedValue({ exists: true, draft: null })
  api.putFactoryConfig.mockReset().mockResolvedValue({ exists: true, draft: null })
})

describe("FactoryBoard", () => {
  it("names the repo by its folder and orders the columns from what waits on the user to what is finished", () => {
    const { view } = renderBoard([])
    expect(view.getByRole("heading", { level: 2 })).toHaveTextContent("Factory · gigachad-bot")
    expect(view.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent)).toEqual(["Needs you", "Working", "Merging", "Finished"])
  })

  it("sorts every task into the column of its shown state, a column by state and then by the newest change", () => {
    const { titles } = renderBoard([
      task("Task cancelled", "cancelled"),
      task("Task done", "done"),
      task("Task merging", "merging"),
      task("Task running", "running"),
      task("Task starting", null, { live: null }),
      task("Task lost child", "running", { live: null }),
      task("Task lost merge", "merging", { live: null }),
      task("Task blocked", "blocked"),
      task("Task waiting old", "waiting_approve"),
      task("Task waiting new", "waiting_approve"),
    ])

    expect(titles("Needs you")).toEqual(["Task waiting new", "Task waiting old", "Task blocked", "Task lost merge", "Task lost child"])
    expect(titles("Working")).toEqual(["Task starting", "Task running"])
    expect(titles("Merging")).toEqual(["Task merging"])
    expect(titles("Finished")).toEqual(["Task done", "Task cancelled"])
  })

  it("shows a card's title, state and round, the reason of a blocked task only, and the branch or the current checkout", () => {
    const { card } = renderBoard([
      task("Task blocked", "blocked", { reason: "merge_failed", round: 4, branch: "factory/fix-header", worktree: "/worktrees/fix-header" }),
      task("Task cancelled", "cancelled", { reason: "cancelled by the user", round: 2 }),
      task("Task starting", null, { live: null }),
      task("Task lost child", "running", { live: null, round: 3 }),
    ])

    expect(card("Task blocked")).toHaveTextContent("Blocked · round 4")
    expect(card("Task blocked")).toHaveTextContent("merge_failed")
    expect(card("Task blocked")).toHaveTextContent("factory/fix-header")
    expect(card("Task blocked")).not.toHaveTextContent("Current checkout")
    expect(card("Task cancelled")).toHaveTextContent("Cancelled · round 2")
    expect(card("Task cancelled")).not.toHaveTextContent("cancelled by the user")
    expect(card("Task cancelled")).toHaveTextContent("Current checkout")
    expect(card("Task starting")).toHaveTextContent("Launching session")
    expect(card("Task lost child")).toHaveTextContent("Interrupted · round 3")
  })

  it("shows a card's tokens and cost once a model answered", () => {
    const { card } = renderBoard([
      task("Task used", "running", { usage: { input: 3_000_000, output: 21_000, cost: 2.903 } }),
      task("Task fresh", "running"),
    ])

    expect(within(card("Task used")).getByLabelText("Tokens and cost")).toHaveTextContent("3.0M tok · $2.90")
    expect(within(card("Task fresh")).queryByLabelText("Tokens and cost")).toBeNull()
  })

  it("pulses a card's dot only while its Pi child runs a turn", () => {
    const { card } = renderBoard([
      task("Task streaming", "running", { live: { ...IDLE, running: true } }),
      task("Task between turns", "running"),
      task("Task waiting", "waiting_approve"),
    ])
    const pulsing = (title: string) => card(title).querySelector(".animate-pulse") !== null

    expect(pulsing("Task streaming")).toBe(true)
    expect(pulsing("Task between turns")).toBe(false)
    expect(pulsing("Task waiting")).toBe(false)
  })

  it("opens a task from its card", () => {
    const tasks = [task("Task running", "running")]
    const { view, onOpenTask } = renderBoard(tasks)
    fireEvent.click(view.getByRole("button", { name: /Task running/ }))
    expect(onOpenTask).toHaveBeenCalledWith(tasks[0].id)
  })

  it("offers delete on finished, blocked, interrupted and lost cards only", () => {
    const { view } = renderBoard([
      task("Task blocked", "blocked"), task("Task done", "done"), task("Task cancelled", "cancelled"),
      task("Task waiting", "waiting_approve"), task("Task running", "running"), task("Task merging", "merging"),
      task("Task starting", null, { live: null, created_at: new Date().toISOString() }),
      task("Task lost", null, { live: null, created_at: new Date(Date.now() - 2 * 60_000).toISOString() }),
      task("Task lost child", "running", { live: null }),
    ])
    expect(view.getAllByRole("button", { name: /^Delete / }).map((button) => button.getAttribute("aria-label")).sort()).toEqual([
      "Delete Task blocked", "Delete Task cancelled", "Delete Task done", "Delete Task lost", "Delete Task lost child",
    ])
  })

  it("deletes a task after a confirmation and reads the tasks again", async () => {
    const tasks = [task("Task done", "done"), task("Task blocked", "blocked")]
    const { view, refresh, onOpenTask } = renderBoard(tasks)

    fireEvent.click(view.getByRole("button", { name: "Delete Task done" }))
    expect(api.deleteFactoryTask).not.toHaveBeenCalled()
    fireEvent.click(within(view.getByRole("group", { name: "Delete this task?" })).getByRole("button", { name: "Yes" }))

    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1))
    expect(api.deleteFactoryTask).toHaveBeenCalledTimes(1)
    expect(api.deleteFactoryTask).toHaveBeenCalledWith(tasks[0].id)
    expect(onOpenTask).not.toHaveBeenCalled()
    await waitFor(() => expect(view.queryByRole("group", { name: "Delete this task?" })).toBeNull())
  })

  it("keeps the task when the confirmation is declined", () => {
    const { view, refresh } = renderBoard([task("Task done", "done")])

    fireEvent.click(view.getByRole("button", { name: "Delete Task done" }))
    fireEvent.click(view.getByRole("button", { name: "No" }))

    expect(view.queryByRole("group", { name: "Delete this task?" })).toBeNull()
    expect(view.getByRole("button", { name: "Delete Task done" })).toBeInTheDocument()
    expect(api.deleteFactoryTask).not.toHaveBeenCalled()
    expect(refresh).not.toHaveBeenCalled()
  })

  it("asks in place of the card's last row, so the card keeps its rows", () => {
    const { view, card } = renderBoard([task("Task done", "done")])
    const marker = view.getByText("Current checkout").parentElement!
    expect(marker).not.toHaveClass("invisible")

    fireEvent.click(view.getByRole("button", { name: "Delete Task done" }))

    expect(marker).toHaveClass("invisible")
    expect(card("Task done")).toContainElement(marker)
    expect(view.getByRole("group", { name: "Delete this task?" }).parentElement).toHaveClass("absolute")
  })

  it("shows the backend's refusal to delete and keeps the card", async () => {
    api.deleteFactoryTask.mockRejectedValueOnce(new Error("The task is running"))
    const { view, refresh } = renderBoard([task("Task blocked", "blocked")])

    fireEvent.click(view.getByRole("button", { name: "Delete Task blocked" }))
    fireEvent.click(view.getByRole("button", { name: "Yes" }))

    expect(await view.findByRole("alert")).toHaveTextContent("The task is running")
    expect(refresh).not.toHaveBeenCalled()
    expect(view.getByText("Task blocked")).toBeInTheDocument()
  })

  it("shows why the tasks could not be read", () => {
    const { view } = renderBoard([], { error: "Factory is unreachable" })
    expect(view.getByRole("alert")).toHaveTextContent("Factory is unreachable")
  })

  it("caps the finished column until asked for all", () => {
    const { view, titles } = renderBoard(Array.from({ length: 12 }, (_, index) => task(`Task ${index}`, "done")))
    expect(titles("Finished")).toHaveLength(10)

    fireEvent.click(view.getByRole("button", { name: "Show all 12" }))
    expect(titles("Finished")).toHaveLength(12)
  })

  it("archives a finished task out of the finished count into a separate counter, and brings it back", async () => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
    const { view, titles } = renderBoard([task("Task a", "done"), task("Task b", "cancelled"), task("Task c", "running")])
    expect(view.queryByRole("button", { name: /^Archive Task c/ })).toBeNull()

    fireEvent.click(view.getByRole("button", { name: "Archive Task a" }))
    await waitFor(() => expect(titles("Finished")).toEqual(["Task b"]))
    const counter = view.getByRole("button", { name: "Archived" })
    expect(counter).toHaveTextContent("1")

    fireEvent.click(counter)
    await waitFor(() => expect(titles("Finished")).toEqual(["Task a"]))
    fireEvent.click(view.getByRole("button", { name: "Unarchive Task a" }))
    await waitFor(() => expect(titles("Finished")).toHaveLength(2))
    expect(view.queryByRole("button", { name: "Archived" })).toBeNull()
    vi.unstubAllGlobals()
  })

  it("closes the dialog at once and shows the start as launching until its task is listed", async () => {
    let answer: (task: unknown) => void = () => {}
    api.createFactoryTask.mockReturnValue(new Promise((resolve) => { answer = resolve }))
    const { view } = renderBoard([])
    fireEvent.click(view.getByRole("button", { name: "New task" }))
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())
    fireEvent.change(view.getByLabelText("Title"), { target: { value: "Fix" } })
    fireEvent.change(view.getByLabelText("Request"), { target: { value: "Fix it" } })
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(view.queryByLabelText("Title")).toBeNull())
    expect(api.createFactoryTask).toHaveBeenCalledWith({ repo: REPO, title: "Fix", request: "Fix it", overrides: { review: true } })
    const launch = within(view.getByRole("region", { name: "Working" })).getByRole("group", { name: "Fix" })
    expect(launch).toHaveTextContent("Launching session")

    answer({})
    await waitFor(() => expect(view.queryByRole("group", { name: "Fix" })).toBeNull())
    expect(api.listFactoryTasks).toHaveBeenCalledWith(REPO)
  })

  it("moves a refused start to Needs you with its reason, to retry or dismiss", async () => {
    api.createFactoryTask.mockRejectedValueOnce(new Error("The checkout already has a task"))
    const { view } = renderBoard([])
    fireEvent.click(view.getByRole("button", { name: "New task" }))
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())
    fireEvent.change(view.getByLabelText("Title"), { target: { value: "Refused" } })
    fireEvent.change(view.getByLabelText("Request"), { target: { value: "Fix it" } })
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    const needsYou = view.getByRole("region", { name: "Needs you" })
    await waitFor(() => expect(within(needsYou).getByRole("group", { name: "Refused" })).toHaveTextContent("Could not start"))
    expect(within(needsYou).getByRole("group", { name: "Refused" })).toHaveTextContent("The checkout already has a task")

    fireEvent.click(view.getByRole("button", { name: "Retry Refused" }))
    await waitFor(() => expect(view.queryByRole("group", { name: "Refused" })).toBeNull())
    expect(api.createFactoryTask).toHaveBeenCalledTimes(2)

    api.createFactoryTask.mockRejectedValueOnce(new Error("No"))
    fireEvent.click(view.getByRole("button", { name: "New task" }))
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())
    fireEvent.change(view.getByLabelText("Title"), { target: { value: "Dismissed" } })
    fireEvent.change(view.getByLabelText("Request"), { target: { value: "Fix it" } })
    fireEvent.click(view.getByRole("button", { name: "Start" }))
    fireEvent.click(await view.findByRole("button", { name: "Dismiss Dismissed" }))
    await waitFor(() => expect(view.queryByRole("group", { name: "Dismissed" })).toBeNull())
  })
})

describe("a repo without factory.yaml", () => {
  it("shows the draft, keeps New task disabled and enables it once the draft is saved", async () => {
    api.getFactoryConfig.mockResolvedValue({ exists: false, draft: "gates: []\n" })
    const { view } = renderBoard([])
    const editor = await view.findByLabelText("factory.yaml")
    expect(editor).toHaveProperty("value", "gates: []\n")
    const newTask = view.getByRole("button", { name: "New task" })
    expect(newTask).toHaveProperty("disabled", true)

    fireEvent.change(editor, { target: { value: "gates:\n  - [make, lint]\n" } })
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.putFactoryConfig).toHaveBeenCalledWith(REPO, "gates:\n  - [make, lint]\n"))
    await waitFor(() => expect(view.getByRole("button", { name: "New task" })).toHaveProperty("disabled", false))
    expect(view.queryByLabelText("factory.yaml")).toBeNull()
  })

  it("shows no draft for a repo that has the file", async () => {
    const { view } = renderBoard([])
    await waitFor(() => expect(api.getFactoryConfig).toHaveBeenCalled())
    expect(view.queryByLabelText("factory.yaml")).toBeNull()
    expect(view.getByRole("button", { name: "New task" })).toHaveProperty("disabled", false)
  })
})
