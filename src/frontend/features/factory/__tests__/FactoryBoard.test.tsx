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
import { factoryTask, factoryWave, waveTask } from "./factoryTask"

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
      <FactoryBoard repo={REPO} tasks={tasks} waves={[]} error={null} refresh={refresh} onOpenTask={onOpenTask} onOpenSession={vi.fn()} {...props} />
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
    expect(view.getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent)).toEqual(["Working", "Merging", "Finished"])
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

    expect(titles("Waiting")).toEqual(["Task waiting new", "Task waiting old", "Task blocked", "Task lost merge", "Task lost child"])
    expect(titles("Working")).toEqual(["Task waiting new", "Task waiting old", "Task blocked", "Task lost merge", "Task lost child", "Task starting", "Task running"])
    expect(titles("Merging")).toEqual(["Task merging"])
    expect(titles("Finished")).toEqual(["Task done", "Task cancelled"])
  })

  it("puts a task asking a question in Waiting, its card showing the question", () => {
    const { titles, card } = renderBoard([
      task("Task asking", "waiting_answer", { question: "Keep the old flag?" }),
      task("Task running", "running"),
    ])

    expect(titles("Waiting")).toEqual(["Task asking"])
    expect(card("Task asking")).toHaveTextContent("Waiting for supervisor · round 1")
    expect(within(card("Task asking")).getByLabelText("Question")).toHaveTextContent("Keep the old flag?")
    expect(within(card("Task running")).queryByLabelText("Question")).toBeNull()
  })

  /** The count beside a column's title, null when it shows none. */
  const columnCount = (view: ReturnType<typeof renderBoard>["view"], column: string) =>
    within(view.getByRole("region", { name: column })).getByRole("heading", { level: 3 }).nextElementSibling?.textContent ?? null

  it("renders no Waiting block while nothing waits", () => {
    const { view, titles } = renderBoard([task("Task running", "running")])
    expect(view.queryByRole("region", { name: "Waiting" })).toBeNull()
    expect(view.queryByRole("heading", { name: /^Waiting/ })).toBeNull()
    expect(titles("Working")).toEqual(["Task running"])
  })

  it("shows a blocked task and a rejected wave in the Waiting block atop Working, out of Working's count", () => {
    const { view, titles } = renderBoard([task("Task blocked", "blocked"), task("Task running", "running"), task("Task rejected", "done", { wave: "r" })], {
      waves: [factoryWave("r", { tag: "r1", title: "Rejected wave", stage: "rejected", counts: { total: 1, done: 1 } })],
    })
    const waiting = within(view.getByRole("region", { name: "Waiting" }))
    expect(waiting.getByRole("heading", { level: 4 })).toHaveTextContent("Waiting 2")
    // The open wave container holds its done task's card
    expect(titles("Waiting")).toEqual(["Task rejected", "Task blocked"])
    expect(waiting.getByRole("group", { name: "Rejected wave" })).toBeInTheDocument()
    expect(within(view.getByRole("region", { name: "Working" })).getByRole("region", { name: "Waiting" })).toBeInTheDocument()
    expect(columnCount(view, "Working")).toBe("1")
  })

  describe("column gap", () => {
    const stored = new Map<string, string>()
    beforeEach(() => {
      stored.clear()
      vi.stubGlobal("localStorage", {
        getItem: (key: string) => stored.get(key) ?? null,
        setItem: (key: string, value: string) => { stored.set(key, value) },
        removeItem: (key: string) => { stored.delete(key) },
      })
      return () => vi.unstubAllGlobals()
    })

    it("is a separator between every two columns, the arrow keys changing the one gap of all of them", async () => {
      const { view } = renderBoard([])
      const gaps = view.getAllByRole("separator", { name: "Resize column gap" })
      expect(gaps).toHaveLength(2)
      expect(gaps[0]).toHaveAttribute("aria-valuenow", "64")

      fireEvent.keyDown(gaps[1], { key: "ArrowLeft" })
      await waitFor(() => expect(gaps[0]).toHaveAttribute("aria-valuenow", "56"))
      expect(gaps[1]).toHaveAttribute("aria-valuenow", "56")
      expect(gaps[0].parentElement).toHaveStyle({ width: "56px" })
      expect(stored.get("gigachad:factory-column-gap")).toBe("56")

      fireEvent.keyDown(gaps[0], { key: "Home" })
      await waitFor(() => expect(gaps[0]).toHaveAttribute("aria-valuenow", "0"))
      fireEvent.keyDown(gaps[0], { key: "ArrowLeft" })
      fireEvent.keyDown(gaps[0], { key: "End" })
      await waitFor(() => expect(gaps[0]).toHaveAttribute("aria-valuenow", "64"))
      fireEvent.keyDown(gaps[0], { key: "ArrowRight" })
      expect(gaps[0]).toHaveAttribute("aria-valuenow", "64")
    })

    it("restores the gap from storage, clamped to the default", () => {
      stored.set("gigachad:factory-column-gap", "24")
      const { view } = renderBoard([])
      expect(view.getAllByRole("separator")[0]).toHaveAttribute("aria-valuenow", "24")
      view.unmount()
      stored.set("gigachad:factory-column-gap", "500")
      expect(renderBoard([]).view.getAllByRole("separator")[0]).toHaveAttribute("aria-valuenow", "64")
    })

    /** jsdom has no pointer events nor capture: a mouse event carrying `pointerId`, and a spy for the test's duration. */
    function stubCapture() {
      vi.stubGlobal("PointerEvent", class extends MouseEvent {
        readonly pointerId: number
        constructor(type: string, init: PointerEventInit = {}) {
          super(type, init)
          this.pointerId = init.pointerId ?? 0
        }
      })
      const original = Object.getOwnPropertyDescriptor(Element.prototype, "setPointerCapture")
      const capture = vi.fn()
      Object.defineProperty(Element.prototype, "setPointerCapture", { value: capture, configurable: true, writable: true })
      return { capture, restore: () => original ? Object.defineProperty(Element.prototype, "setPointerCapture", original) : Reflect.deleteProperty(Element.prototype, "setPointerCapture") }
    }

    it("follows a pointer drag live, stores the gap once released and restores the body's cursor and selection", async () => {
      const { capture, restore } = stubCapture()
      try {
        const { view } = renderBoard([])
        const gaps = view.getAllByRole("separator", { name: "Resize column gap" })
        fireEvent.pointerDown(gaps[0], { button: 0, pointerId: 1, clientX: 500 })
        expect(capture).toHaveBeenCalledWith(1)
        expect(document.body.style.cursor).toBe("col-resize")
        expect(document.body.style.userSelect).toBe("none")

        fireEvent.pointerMove(gaps[0], { pointerId: 1, clientX: 476 })
        await waitFor(() => expect(gaps[1]).toHaveAttribute("aria-valuenow", "40"))
        fireEvent.pointerMove(gaps[0], { pointerId: 1, clientX: 400 })
        await waitFor(() => expect(gaps[0]).toHaveAttribute("aria-valuenow", "0"))
        expect(stored.has("gigachad:factory-column-gap")).toBe(false)

        fireEvent.pointerMove(gaps[0], { pointerId: 1, clientX: 492 })
        fireEvent(gaps[0], new Event("lostpointercapture"))
        await waitFor(() => expect(stored.get("gigachad:factory-column-gap")).toBe("56"))
        expect(gaps[0]).toHaveAttribute("aria-valuenow", "56")
        expect(document.body.style.cursor).toBe("")
        expect(document.body.style.userSelect).toBe("")

        // Released, the handle follows the pointer no more
        fireEvent.pointerMove(gaps[0], { pointerId: 1, clientX: 400 })
        expect(gaps[0]).toHaveAttribute("aria-valuenow", "56")
      } finally {
        restore()
      }
    })

    it("restores the body's cursor and selection when unmounted mid-drag, storing nothing", () => {
      const { restore } = stubCapture()
      try {
        const { view } = renderBoard([])
        const gap = view.getAllByRole("separator", { name: "Resize column gap" })[0]
        fireEvent.pointerDown(gap, { button: 0, pointerId: 1, clientX: 500 })
        fireEvent.pointerMove(gap, { pointerId: 1, clientX: 480 })
        expect(document.body.style.cursor).toBe("col-resize")
        view.unmount()
        expect(document.body.style.cursor).toBe("")
        expect(document.body.style.userSelect).toBe("")
        expect(stored.has("gigachad:factory-column-gap")).toBe(false)
      } finally {
        restore()
      }
    })
  })

  describe("waves", () => {
    /** The containers of a column by their accessible names. */
    const containers = (view: ReturnType<typeof renderBoard>["view"], column: string) =>
      within(view.getByRole("region", { name: column })).queryAllByRole("group").map((group) => group.getAttribute("aria-label"))

    it("collects a building wave's done tasks in its container in Merging, its running task a card in Working naming the wave", () => {
      const doneOne = task("Task done one", "done", { tag: "w1", wave: "wave1" })
      const running = task("Task running", "running", { tag: "w1", wave: "wave1" })
      const waves = [factoryWave("wave1", { tag: "w1", title: "Board waves", counts: { total: 8, done: 5 }, tasks: [
        waveTask("a", { taskId: doneOne.id, state: "done" }), waveTask("b", { taskId: running.id, state: "running" }), waveTask("c", { title: "Task pending" }),
      ] })]
      const { view, titles, card, onOpenTask } = renderBoard([doneOne, running, task("Task plain done", "done")], { waves })

      expect(titles("Finished")).toEqual(["Task plain done"])
      expect(titles("Working")).toEqual(["Task running"])
      expect(within(card("Task running")).getByLabelText("Wave")).toHaveTextContent("w1")
      const merging = within(view.getByRole("region", { name: "Merging" }))
      const wave = within(merging.getByRole("group", { name: "Board waves" }))
      const header = wave.getAllByRole("button")[0]
      expect(within(header).getByLabelText("Progress")).toHaveTextContent("5/8")
      expect(wave.getByLabelText("Stage")).toHaveTextContent("building")
      expect(header).toHaveTextContent("Board waves")
      expect(view.queryByText("Task pending")).toBeNull()
      fireEvent.click(wave.getByRole("button", { name: /^Task done one/ }))
      expect(onOpenTask).toHaveBeenCalledWith(doneOne.id)
    })

    it("shows a building wave without a done task nowhere but on its tasks' cards", () => {
      const running = task("Task running", "running", { wave: "wave1" })
      const { view } = renderBoard([running], { waves: [factoryWave("wave1", { tag: "w1", title: "Board waves", counts: { total: 2, done: 0 } })] })
      expect(view.queryByRole("group", { name: "Board waves" })).toBeNull()
    })

    it("puts a rejected wave's container in Waiting, a merged one's in Finished closed", () => {
      const merged = task("Task merged", "done", { wave: "m" })
      const { view } = renderBoard([merged, task("Task rejected", "done", { wave: "r" })], { waves: [
        factoryWave("r", { tag: "r1", title: "Rejected wave", stage: "rejected", counts: { total: 1, done: 1 } }),
        factoryWave("m", { tag: "m1", title: "Merged wave", stage: "merged", counts: { total: 1, done: 1 } }),
      ] })
      expect(containers(view, "Waiting")).toEqual(["Rejected wave"])
      expect(containers(view, "Finished")).toEqual(["Merged wave"])
      const wave = within(view.getByRole("group", { name: "Merged wave" }))
      expect(wave.getAllByRole("button")[0]).toHaveAttribute("aria-expanded", "false")
      expect(view.queryByText("Task merged")).toBeNull()
    })

    it("wraps two waves of one milestone in one container counting the merged waves, titled by its name without a document", () => {
      const { view } = renderBoard([], { waves: [
        factoryWave("a", { tag: "plugin/m1/w1", title: "First", stage: "reviewing" }),
        factoryWave("b", { tag: "plugin/m1/w2", title: "Second", stage: "merging" }),
      ] })
      expect(containers(view, "Merging")).toEqual(["plugin", "m1", "First", "Second"])
      const milestone = within(view.getByRole("group", { name: "m1" }))
      expect(within(milestone.getAllByRole("button")[0]).getByLabelText("Progress")).toHaveTextContent("0/2")
    })

    it("keeps a task in the columns whose wave the list does not hold, without a wave line", () => {
      const { titles, card } = renderBoard([task("Task orphan", "done", { wave: "gone" }), task("Task mine", "running", { tag: "plugin/m1/w2" })])
      expect(titles("Finished")).toEqual(["Task orphan"])
      expect(titles("Working")).toEqual(["Task mine"])
      expect(within(card("Task orphan")).queryByLabelText("Wave")).toBeNull()
    })

    it("renders nothing above the columns", () => {
      const { view } = renderBoard([], { waves: [factoryWave("a", { tag: "w1", stage: "merged" })] })
      expect(view.queryByRole("region", { name: "Waves" })).toBeNull()
      expect(view.getAllByRole("region").map((region) => region.getAttribute("aria-label"))).toEqual(["Working", "Merging", "Finished"])
    })
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

  it("shows no model line on a card", () => {
    const settings = { branch: "master", merge: "auto", maxRounds: 3, review: true, models: { coder: "anthropic/claude-opus-4" } } as const
    const { card } = renderBoard([task("Task modelled", "running", { settings })])
    expect(card("Task modelled")).not.toHaveTextContent("anthropic/claude-opus-4")
  })

  it("shows a card's tokens and cost once a model answered", () => {
    const { card } = renderBoard([
      task("Task used", "running", { usage: { input: 3_000_000, output: 21_000, cost: 2.903 } }),
      task("Task fresh", "running"),
    ])

    expect(within(card("Task used")).getByLabelText("Tokens and cost")).toHaveTextContent("↑3.0M ↓21k · $2.90")
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

  it("moves a refused start to Waiting with its reason, to retry or dismiss", async () => {
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

    const needsYou = await view.findByRole("region", { name: "Waiting" })
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
