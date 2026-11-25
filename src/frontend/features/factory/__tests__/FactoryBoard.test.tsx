import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor, within } from "@testing-library/react"
import type { FactoryRepoSettings, FactoryTask, FactoryTaskState } from "@/lib/domain/factory"

type Handler = (params: Record<string, unknown>) => Promise<unknown>

const api = vi.hoisted(() => {
  class ApiError extends Error {
    constructor(message: string, readonly status: number, readonly code?: string, readonly detail?: Record<string, unknown>) {
      super(message)
    }
  }
  const handlers: Record<string, Handler> = {}
  return {
    ApiError,
    handlers,
    callFactory: vi.fn(async (action: string, params: Record<string, unknown> = {}) => {
      const handler = handlers[action]
      if (!handler) throw new Error(`unexpected ${action}`)
      return handler(params)
    }),
    fetchModels: vi.fn<() => Promise<never>>(),
    getFactoryActions: vi.fn(async () => ({ actions: {} })),
  }
})

vi.mock("@/lib/backend/api", () => api)

import { FactoryBoard, formatAge } from "@/features/factory/FactoryBoard"
import { TopBarSlot, TopBarSlotProvider } from "@/features/navigation/TopBarSlots"
import { factoryTask } from "./factoryTask"

const REPO = "/home/me/git/gigachad-bot"

const SETTINGS: FactoryRepoSettings = {
  stored: {},
  resolved: { merge: "review", maxRounds: 3, review: true, preview: false, models: {}, branch: "main" },
  defaults: { merge: "auto", maxRounds: 2, review: false, preview: false, models: {}, branch: "main" },
}

let minute = 0

/** A task named after its state, each one's state newer than the one made before it; the core accepts its removal. */
function task(title: string, state: FactoryTaskState | null, extra: Partial<FactoryTask> = {}): FactoryTask {
  minute += 1
  const at = `2026-01-01T00:${String(minute).padStart(2, "0")}:00Z`
  return factoryTask(`id${title.toLowerCase().replace(/[^a-z0-9]/g, "")}000000`, {
    repo: REPO, title, createdAt: at, state, round: state === null ? null : 1, at: state === null ? null : at, running: false, actions: ["tasks.remove"], ...extra,
  })
}

/** Calls of `action` to the core, their params. */
function called(action: string): unknown[] {
  return api.callFactory.mock.calls.filter(([name]) => name === action).map(([, params]) => params)
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
  api.callFactory.mockClear()
  for (const action of Object.keys(api.handlers)) delete api.handlers[action]
  Object.assign(api.handlers, {
    "tasks.remove": async (params: Record<string, unknown>) => ({ removed: params.id, dropped: null }),
    "settings.get": async () => SETTINGS,
    "tasks.start": async () => ({}),
    "tasks.list": async () => ({ tasks: [] }),
    "branches.list": async () => ({ branches: [{ name: "main", default: true, hasConfig: true }] }),
    "config.get": async () => ({ exists: true, text: "gates: []\n", draft: null, source: [] }),
    "config.write": async () => ({ path: `${REPO}/factory.yaml` }),
  } satisfies Record<string, Handler>)
  api.fetchModels.mockReset().mockReturnValue(new Promise(() => {}))
})

describe("FactoryBoard", () => {
  it("names the repo by its folder and orders the columns from what waits on the user to what is finished", () => {
    const { view } = renderBoard([])
    expect(view.getByRole("heading", { level: 2 })).toHaveTextContent("Factory · gigachad-bot")
    expect(view.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent)).toEqual(["Needs you", "Working", "Merging", "Finished"])
  })

  it("sorts every task into the column of its shown state, a column newest created first", () => {
    const { titles } = renderBoard([
      task("Task cancelled", "cancelled"),
      task("Task done", "done"),
      task("Task merging", "merging"),
      task("Task running", "running"),
      task("Task starting", null),
      task("Task lost child", "running", { interrupted: true }),
      task("Task lost merge", "merging", { interrupted: true }),
      task("Task blocked", "blocked"),
      task("Task waiting old", "waiting_approve"),
      task("Task waiting new", "waiting_approve"),
    ])

    expect(titles("Needs you")).toEqual(["Task waiting new", "Task waiting old", "Task blocked", "Task lost merge", "Task lost child"])
    expect(titles("Working")).toEqual(["Task starting", "Task running"])
    expect(titles("Merging")).toEqual(["Task merging"])
    expect(titles("Finished")).toEqual(["Task done", "Task cancelled"])
  })

  it("puts a task asking a question in Needs you, its card showing the question", () => {
    const { titles, card } = renderBoard([
      task("Task asking", "waiting_answer", { question: "Keep the old flag?" }),
      task("Task running", "running"),
    ])

    expect(titles("Needs you")).toEqual(["Task asking"])
    expect(card("Task asking")).toHaveTextContent("Question · round 1")
    expect(within(card("Task asking")).getByLabelText("Question")).toHaveTextContent("Keep the old flag?")
    expect(within(card("Task running")).queryByLabelText("Question")).toBeNull()
  })

  it("shows a merge request's kind and source branch on its card", () => {
    const { card } = renderBoard([
      task("Task merge", "running", { kind: "merge_request", branch: "wave/w1" }),
      task("Task coded", "running", { branch: "factory/coded" }),
    ])

    expect(within(card("Task merge")).getByLabelText("Merge request")).toHaveTextContent("merge request · wave/w1")
    expect(within(card("Task coded")).queryByLabelText("Merge request")).toBeNull()
  })

  it("formats a task's age in minutes, hours, days, then weeks", () => {
    const now = Date.parse("2026-03-01T00:00:00Z")
    const ago = (ms: number) => formatAge(new Date(now - ms).toISOString(), now)
    const MIN = 60_000
    expect([0, 59 * MIN, 60 * MIN, 23 * 60 * MIN, 24 * 60 * MIN, 6 * 24 * 60 * MIN, 7 * 24 * 60 * MIN, 15 * 24 * 60 * MIN].map(ago))
      .toEqual(["0min", "59min", "1h", "23h", "1d", "6d", "1w", "2w"])
  })

  it("shows a card's title, state and round, and the reason of a blocked task only", () => {
    const { card } = renderBoard([
      task("Task blocked", "blocked", { reason: "merge_failed", round: 4, branch: "factory/fix-header", worktree: "/worktrees/fix-header" }),
      task("Task cancelled", "cancelled", { reason: "cancelled by the user", round: 2, branch: "factory/other" }),
      task("Task starting", null),
      task("Task lost child", "running", { interrupted: true, round: 3 }),
    ])

    expect(card("Task blocked")).toHaveTextContent("Blocked · round 4")
    expect(card("Task blocked")).toHaveTextContent("merge_failed")
    expect(card("Task blocked")).not.toHaveTextContent("factory/fix-header")
    expect(card("Task cancelled")).toHaveTextContent("Cancelled · round 2")
    expect(card("Task cancelled")).not.toHaveTextContent("cancelled by the user")
    expect(card("Task cancelled")).not.toHaveTextContent("factory/other")
    expect(card("Task starting")).toHaveTextContent("Launching session")
    expect(card("Task lost child")).toHaveTextContent("Interrupted · round 3")
  })

  it("shows a card's coder model when the task has one", () => {
    const settings = { branch: "master", merge: "auto", maxRounds: 3, review: true, models: { coder: "anthropic/claude-opus-4" } } as const
    const { card } = renderBoard([task("Task modelled", "running", { settings }), task("Task default", "running")])

    expect(within(card("Task modelled")).getByLabelText("Coder model")).toHaveTextContent("anthropic/claude-opus-4")
    expect(within(card("Task default")).queryByLabelText("Coder model")).toBeNull()
  })

  it("shows a card's tokens and cost once a model answered", () => {
    const { card } = renderBoard([
      task("Task used", "running", { usage: { input: 3_000_000, output: 21_000, cost: 2.903 } }),
      task("Task fresh", "running"),
    ])

    expect(within(card("Task used")).getByLabelText("Tokens and cost")).toHaveTextContent("3.0M tok · $2.90")
    expect(within(card("Task fresh")).queryByLabelText("Tokens and cost")).toBeNull()
  })

  it("pulses a card's dot only while a runner holds the task", () => {
    const { card } = renderBoard([
      task("Task held", "running", { running: true }),
      task("Task lost child", "running", { interrupted: true }),
      task("Task waiting", "waiting_approve"),
    ])
    const pulsing = (title: string) => card(title).querySelector(".animate-pulse") !== null

    expect(pulsing("Task held")).toBe(true)
    expect(pulsing("Task lost child")).toBe(false)
    expect(pulsing("Task waiting")).toBe(false)
  })

  it("opens a task from its card", () => {
    const tasks = [task("Task running", "running")]
    const { view, onOpenTask } = renderBoard(tasks)
    fireEvent.click(view.getByRole("button", { name: /^Task running/ }))
    expect(onOpenTask).toHaveBeenCalledWith(tasks[0].id)
  })

  it("offers delete exactly on the cards whose task the core would remove now", () => {
    const { view } = renderBoard([
      task("Task blocked", "blocked"), task("Task waiting", "waiting_approve"), task("Task lost child", "running", { interrupted: true }),
      task("Task running", "running", { running: true, actions: ["tasks.cancel", "tasks.message"] }),
      task("Task ending", "done", { running: true, actions: [] }),
    ])
    expect(view.getAllByRole("button", { name: /^Delete / }).map((button) => button.getAttribute("aria-label")).sort()).toEqual([
      "Delete Task blocked", "Delete Task lost child", "Delete Task waiting",
    ])
  })

  it("deletes a task after a confirmation and reads the tasks again", async () => {
    const tasks = [task("Task done", "done"), task("Task blocked", "blocked")]
    const { view, refresh, onOpenTask } = renderBoard(tasks)

    fireEvent.click(view.getByRole("button", { name: "Delete Task done" }))
    expect(called("tasks.remove")).toEqual([])
    fireEvent.click(within(view.getByRole("group", { name: "Delete and revert its code?" })).getByRole("button", { name: "Yes" }))

    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1))
    expect(called("tasks.remove")).toEqual([{ id: tasks[0].id }])
    expect(onOpenTask).not.toHaveBeenCalled()
    await waitFor(() => expect(view.queryByRole("group", { name: "Delete and revert its code?" })).toBeNull())
  })

  it("keeps the task when the confirmation is declined", () => {
    const { view, refresh } = renderBoard([task("Task done", "done")])

    fireEvent.click(view.getByRole("button", { name: "Delete Task done" }))
    fireEvent.click(view.getByRole("button", { name: "No" }))

    expect(view.queryByRole("group", { name: "Delete and revert its code?" })).toBeNull()
    expect(view.getByRole("button", { name: "Delete Task done" })).toBeInTheDocument()
    expect(called("tasks.remove")).toEqual([])
    expect(refresh).not.toHaveBeenCalled()
  })

  it("asks in place of the card's last row, so the card keeps its rows", () => {
    const { view, card } = renderBoard([task("Task done", "done", { usage: { input: 1000, output: 500, cost: 0.1 } })])
    const marker = view.getByLabelText("Tokens and cost").parentElement!
    expect(marker).not.toHaveClass("invisible")

    fireEvent.click(view.getByRole("button", { name: "Delete Task done" }))

    expect(marker).toHaveClass("invisible")
    expect(card("Task done")).toContainElement(marker)
    expect(view.getByRole("group", { name: "Delete and revert its code?" }).parentElement).toHaveClass("absolute")
  })

  it("shows the core's refusal of a task with unmerged work and deletes it with force once confirmed again", async () => {
    const refusal = "task idtaskbl has 1 unmerged commit (not on main), give force: true to remove it anyway"
    api.handlers["tasks.remove"] = async (params) => {
      if (params.force !== true) throw new api.ApiError(refusal, 409, "refused", { force: true })
      return { removed: params.id, dropped: "1 unmerged commit (not on main)" }
    }
    const tasks = [task("Task blocked", "blocked")]
    const { view, refresh, card } = renderBoard(tasks)

    fireEvent.click(view.getByRole("button", { name: "Delete Task blocked" }))
    fireEvent.click(view.getByRole("button", { name: "Yes" }))

    const again = await view.findByRole("group", { name: "Delete anyway?" })
    expect(card("Task blocked")).toHaveTextContent(refusal)
    expect(refresh).not.toHaveBeenCalled()

    fireEvent.click(within(again).getByRole("button", { name: "Yes" }))
    await waitFor(() => expect(refresh).toHaveBeenCalledTimes(1))
    expect(called("tasks.remove")).toEqual([{ id: tasks[0].id }, { id: tasks[0].id, force: true }])
    expect(card("Task blocked")).not.toHaveTextContent(refusal)
  })

  it("keeps a task whose forced delete is declined", async () => {
    api.handlers["tasks.remove"] = async () => { throw new api.ApiError("task x has uncommitted changes", 409, "refused", { force: true }) }
    const { view, refresh } = renderBoard([task("Task blocked", "blocked")])

    fireEvent.click(view.getByRole("button", { name: "Delete Task blocked" }))
    fireEvent.click(view.getByRole("button", { name: "Yes" }))
    fireEvent.click(within(await view.findByRole("group", { name: "Delete anyway?" })).getByRole("button", { name: "No" }))

    expect(view.queryByText("task x has uncommitted changes")).toBeNull()
    expect(called("tasks.remove")).toHaveLength(1)
    expect(refresh).not.toHaveBeenCalled()
  })

  it("shows a refusal that force would not lift as an error, without asking to delete anyway", async () => {
    api.handlers["tasks.remove"] = async () => { throw new api.ApiError("task x is running, cancel it first", 409, "refused") }
    const { view, refresh } = renderBoard([task("Task blocked", "blocked")])

    fireEvent.click(view.getByRole("button", { name: "Delete Task blocked" }))
    fireEvent.click(view.getByRole("button", { name: "Yes" }))

    expect(await view.findByRole("alert")).toHaveTextContent("task x is running, cancel it first")
    expect(view.queryByRole("group", { name: "Delete anyway?" })).toBeNull()
    expect(called("tasks.remove")).toHaveLength(1)
    expect(refresh).not.toHaveBeenCalled()
  })

  it("shows any other failure to delete and keeps the card", async () => {
    api.handlers["tasks.remove"] = async () => { throw new api.ApiError("the runner of task x is still ending, try again in a few seconds", 409, "busy") }
    const { view, refresh } = renderBoard([task("Task blocked", "blocked")])

    fireEvent.click(view.getByRole("button", { name: "Delete Task blocked" }))
    fireEvent.click(view.getByRole("button", { name: "Yes" }))

    expect(await view.findByRole("alert")).toHaveTextContent("the runner of task x is still ending")
    expect(view.queryByRole("group", { name: "Delete anyway?" })).toBeNull()
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
    api.handlers["tasks.start"] = () => new Promise((resolve) => { answer = resolve })
    const { view } = renderBoard([])
    fireEvent.click(view.getByRole("button", { name: "New task" }))
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())
    fireEvent.change(view.getByLabelText("Title"), { target: { value: "Fix" } })
    fireEvent.change(view.getByLabelText("Request"), { target: { value: "Fix it" } })
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(view.queryByLabelText("Title")).toBeNull())
    expect(called("tasks.start")).toEqual([{ repo: REPO, title: "Fix", request: "Fix it" }])
    const launch = within(view.getByRole("region", { name: "Working" })).getByRole("group", { name: "Fix" })
    expect(launch).toHaveTextContent("Launching session")

    answer({})
    await waitFor(() => expect(view.queryByRole("group", { name: "Fix" })).toBeNull())
    expect(called("tasks.list")).toContainEqual({ repo: REPO })
  })

  it("moves a refused start to Needs you with its reason, to retry or dismiss", async () => {
    let refusals = ["the checkout already has a task"]
    api.handlers["tasks.start"] = async () => {
      const refusal = refusals.shift()
      if (refusal) throw new Error(refusal)
      return {}
    }
    const { view } = renderBoard([])
    fireEvent.click(view.getByRole("button", { name: "New task" }))
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())
    fireEvent.change(view.getByLabelText("Title"), { target: { value: "Refused" } })
    fireEvent.change(view.getByLabelText("Request"), { target: { value: "Fix it" } })
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    const needsYou = view.getByRole("region", { name: "Needs you" })
    await waitFor(() => expect(within(needsYou).getByRole("group", { name: "Refused" })).toHaveTextContent("Could not start"))
    expect(within(needsYou).getByRole("group", { name: "Refused" })).toHaveTextContent("the checkout already has a task")

    fireEvent.click(view.getByRole("button", { name: "Retry Refused" }))
    await waitFor(() => expect(view.queryByRole("group", { name: "Refused" })).toBeNull())
    expect(called("tasks.start")).toHaveLength(2)

    refusals = ["No"]
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
    api.handlers["config.get"] = async () => ({ exists: false, text: null, draft: "gates: []\n", source: ["package.json"] })
    const { view } = renderBoard([])
    const editor = await view.findByLabelText("factory.yaml")
    expect(editor).toHaveProperty("value", "gates: []\n")
    const newTask = view.getByRole("button", { name: "New task" })
    expect(newTask).toHaveProperty("disabled", true)

    fireEvent.change(editor, { target: { value: "gates:\n  - [make, lint]\n" } })
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(called("config.write")).toEqual([{ repo: REPO, text: "gates:\n  - [make, lint]\n" }]))
    await waitFor(() => expect(view.getByRole("button", { name: "New task" })).toHaveProperty("disabled", false))
    expect(view.queryByLabelText("factory.yaml")).toBeNull()
  })

  it("shows no draft for a repo that has the file", async () => {
    const { view } = renderBoard([])
    await waitFor(() => expect(called("config.get")).toEqual([{ repo: REPO }]))
    expect(view.queryByLabelText("factory.yaml")).toBeNull()
    expect(view.getByRole("button", { name: "New task" })).toHaveProperty("disabled", false)
  })
})
