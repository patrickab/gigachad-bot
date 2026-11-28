import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { SocketObserver } from "@/features/pi/piSocket"
import type { FactoryTask, FactoryWave, FactoryWaveTask } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({ callFactory: vi.fn() }))
const socket = vi.hoisted(() => ({ observers: new Set<SocketObserver>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/pi/piSocket", () => ({
  observePiSocket: (observer: SocketObserver) => {
    socket.observers.add(observer)
    return () => socket.observers.delete(observer)
  },
}))

import { SessionWaves, SessionWavesPanel, taskModels } from "../SessionWaves"

afterEach(() => {
  cleanup()
  api.callFactory.mockReset()
  vi.unstubAllGlobals()
})

function task(key: string, fields: Partial<FactoryWaveTask> = {}): FactoryWaveTask {
  return { key, title: key, after: [], taskId: null, sessionId: null, state: "pending", usage: null, ...fields }
}

function wave(id: string, fields: Partial<FactoryWave> = {}): FactoryWave {
  return {
    id, repo: "/git/repo", tag: `p/${id}`, plan: null, title: `title ${id}`, request: "", branch: `master-p-${id}`, target: "master", base: "abc",
    origin: "me", createdAt: "2026-01-01T00:00:00Z", stage: "building", reason: null, running: true, interrupted: false,
    tasks: [], counts: { total: 0, done: 0 }, reviews: [], gates: [], merge: null, remote: null, usage: null, actions: [], ...fields,
  }
}

function renderPanel(onOpenTask = vi.fn()) {
  render(<SessionWavesPanel sessionId="me" onOpenTask={onOpenTask} />)
  return onOpenTask
}

describe("SessionWaves", () => {
  it("shows the waves it last read at once, before the first read answers, then the fresh ones", async () => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) }, removeItem: (key: string) => { stored.delete(key) } })
    let answer: (value: { waves: FactoryWave[] }) => void = () => undefined
    const others = (action: string) => (action === "decisions.list" ? { decisions: [] } : { tasks: [] })
    api.callFactory.mockImplementation(async (action: string) => (action === "waves.list" ? { waves: [wave("w1")] } : others(action)))
    const first = render(<SessionWavesPanel sessionId="me" onOpenTask={vi.fn()} />)
    await screen.findByText("p/w1")
    first.unmount()
    api.callFactory.mockReset()
    api.callFactory.mockImplementation((action: string) => (action === "waves.list" ? new Promise((resolve) => { answer = resolve }) : Promise.resolve(others(action))))
    renderPanel()
    expect(screen.getByText("p/w1")).toBeTruthy()
    answer({ waves: [wave("w2")] })
    await screen.findByText("p/w2")
    expect(screen.queryByText("p/w1")).toBeNull()
  })

  it("shows only this session's waves, the active ones open", async () => {
    api.callFactory.mockResolvedValue({
      waves: [
        wave("w1", { counts: { total: 8, done: 2 }, usage: { input: 1, output: 1, cost: 4.1 } }),
        wave("w2", { origin: "other" }),
        wave("w3", { origin: null }),
        wave("w6", { stage: "rejected", running: false }),
      ],
    })
    renderPanel()
    const row = (await screen.findByText("w1")).closest("li")!
    expect(api.callFactory).toHaveBeenCalledWith("waves.list", {})
    expect(row.textContent).toContain("title w1")
    expect(row.textContent).toContain("2/8 tasks")
    expect(row.textContent).toContain("building")
    expect(row.textContent).toContain("$4.10")
    const header = screen.getByRole("button", { name: /^Factory/ })
    expect(header.getAttribute("aria-expanded")).toBe("true")
    // Wave states do not count: only the transcript's cards waiting for a ruling do
    expect(header.textContent).toContain("Factory · $4.10 · 2 active")
    expect(header.textContent).not.toContain("needs you")
    expect(screen.getByText("rejected").className).toContain("text-warning")
    expect(screen.getByText("building").className).not.toContain("text-warning")
    expect(screen.queryByText(/finished/)).toBeNull()
    for (const hidden of ["w2", "w3"]) expect(screen.queryByText(hidden)).toBeNull()
    fireEvent.click(header)
    expect(screen.queryByText("w1")).toBeNull()
  })

  it("collapses to the header with only finished waves", () => {
    render(<SessionWaves
      waves={[wave("w4", { stage: "merged", usage: { input: 1, output: 1, cost: 1 } }), wave("w5", { stage: "cancelled", usage: { input: 1, output: 1, cost: 2 } })]}
      onOpenTask={vi.fn()}
    />)
    const header = screen.getByRole("button", { name: /^Factory/ })
    expect(header.textContent).toContain("Factory · $3.00 · 2 finished")
    expect(header.textContent).not.toContain("needs you")
    expect(header.getAttribute("aria-expanded")).toBe("false")
    expect(screen.queryByText("w4")).toBeNull()
    fireEvent.click(header)
    const finished = screen.getByRole("button", { name: /^2 finished/ })
    expect(finished.textContent).toContain("$3.00")
    expect(finished.getAttribute("aria-expanded")).toBe("false")
  })

  it("orders active waves needing the user first and keeps finished ones behind their row", () => {
    render(<SessionWaves
      waves={[
        wave("old", { createdAt: "2026-01-01T00:00:00Z" }),
        wave("new", { createdAt: "2026-01-03T00:00:00Z" }),
        wave("stuck", { stage: "blocked", reason: "gates", running: false, createdAt: "2026-01-02T00:00:00Z" }),
        wave("m1", { stage: "merged", counts: { total: 2, done: 2 }, createdAt: "2026-01-01T00:00:00Z" }),
        wave("m2", { stage: "cancelled", createdAt: "2026-01-04T00:00:00Z" }),
      ]}
      onOpenTask={vi.fn()}
    />)
    expect(screen.getByRole("button", { name: /^Factory/ }).textContent).toContain("Factory · 3 active")
    const names = () => screen.getAllByText(/^(stuck|new|old|m1|m2)$/).map((node) => node.textContent)
    expect(names()).toEqual(["stuck", "new", "old"])
    const finished = screen.getByRole("button", { name: /^2 finished/ })
    fireEvent.click(finished)
    expect(names()).toEqual(["stuck", "new", "old", "m1", "m2"])
    expect(screen.getByText("m1").closest("li")!.textContent).toContain("2/2 tasks · merged")
  })

  it("shows an active wave's tasks in declared order with model, state and cost; a started one opens its task, the wave's row closes them", () => {
    const onOpenTask = vi.fn()
    const tasks = [
      task("a", { title: "feat: running one", state: "running", taskId: "t-a", sessionId: "s-a", usage: { input: 200, output: 3400, cost: 1 } }),
      task("b", { title: "feat: pending one", after: ["a"] }),
      task("c", { title: "feat: done one", state: "done", taskId: "t-c", sessionId: "s-c" }),
    ]
    render(<SessionWaves waves={[wave("w1", { tasks })]} models={{ "t-a": "claude-opus-5-5" }} onOpenTask={onOpenTask} />)
    const rows = screen.getAllByText(/^feat: /).map((node) => node.closest("li")!.textContent)
    expect(rows.map((text) => text?.match(/feat: \w+ one.*/)?.[0])).toEqual(["feat: running one · claude-opus-5-5 · running · $1.00", "feat: pending one · pending · after a", "feat: done one · done"])
    expect(screen.queryByRole("button", { name: /feat: pending one/ })).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /feat: running one/ }))
    expect(onOpenTask).toHaveBeenCalledWith("/git/repo", "t-a")
    fireEvent.click(screen.getByRole("button", { name: /title w1/ }))
    expect(screen.queryByText("feat: running one")).toBeNull()
  })

  it("renders nothing without waves", async () => {
    api.callFactory.mockResolvedValue({ waves: [] })
    const { container } = render(<SessionWavesPanel sessionId="me" onOpenTask={vi.fn()} />)
    await waitFor(() => expect(api.callFactory).toHaveBeenCalled())
    expect(container.innerHTML).toBe("")
  })

  it("reports any other failed read", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {})
    const cause = new Error("invalid: repo must be an absolute path")
    api.callFactory.mockRejectedValue(cause)
    render(<SessionWavesPanel sessionId="me" onOpenTask={vi.fn()} />)
    await waitFor(() => expect(error).toHaveBeenCalledWith("waves.list failed", cause))
    error.mockRestore()
  })

  it("reads again on a factory change", async () => {
    let lists = 0
    api.callFactory.mockImplementation(async (action: string) => {
      if (action === "waves.list") return { waves: lists++ === 0 ? [] : [wave("w1")] }
      return action === "decisions.list" ? { decisions: [] } : { tasks: [] }
    })
    renderPanel()
    await waitFor(() => expect(lists).toBe(1))
    act(() => {
      for (const observer of socket.observers) observer.onHostEvent?.({ type: "sessions_changed", sessionPaths: ["/agent/factory/waves/w1.jsonl"] })
    })
    expect(await screen.findByText("p/w1")).toBeTruthy()
  })
})

describe("SessionWaves above the composer", () => {
  it("caps its height and scrolls inside", () => {
    render(<SessionWaves waves={[wave("w1")]} onOpenTask={vi.fn()} />)
    const cls = screen.getByRole("region", { name: "Waves" }).className
    expect(cls).toContain("max-h-[40vh]")
    expect(cls).toContain("overflow-y-auto")
  })

  const decision = (answers: string, title = "answered the coder") => ({ at: "t", answers, card: { v: 1 as const, kind: "decision" as const, tone: "quiet" as const, title, answers } })

  it("counts the transcript's attention cards no decision answers, not wave or task states", () => {
    const cards = { notices: new Set(["k1", "k2", "k3"]), waiting: ["k1", "k2"], revealNotice: vi.fn() }
    render(<SessionWaves waves={[wave("w1", { stage: "rejected", running: false }), wave("w2", { tasks: [task("a", { state: "waiting_answer" })] })]} decisions={[decision("k2")]} cards={cards} onOpenTask={vi.fn()} />)
    expect(screen.getByRole("button", { name: /^Factory/ }).textContent).toContain("Factory · 2 active · 1 needs you")
  })

  it("shows with decisions but no waves, its Decisions entry jumping through the transcript", () => {
    const cards = { notices: new Set(["k1"]), waiting: [], revealNotice: vi.fn() }
    render(<SessionWaves waves={[]} decisions={[decision("k1"), decision("gone", "answered another")]} cards={cards} onOpenTask={vi.fn()} />)
    const header = screen.getByRole("button", { name: /^Factory/ })
    expect(header.textContent).not.toContain("needs you")
    fireEvent.click(header)
    fireEvent.click(screen.getByRole("button", { name: /Decisions · 2/ }))
    fireEvent.click(screen.getByRole("button", { name: "answered the coder" }))
    expect(cards.revealNotice).toHaveBeenCalledWith("k1")
    // Its card is not in this transcript: no jump offered
    expect(screen.queryByRole("button", { name: "answered another" })).toBeNull()
    expect(screen.getByText("answered another")).toBeTruthy()
  })

  it("lists the decisions from decisions.list", async () => {
    api.callFactory.mockImplementation(async (action: string, params: unknown) => {
      if (action === "decisions.list") {
        expect(params).toEqual({ origin: "me" })
        return { decisions: [decision("k1")] }
      }
      return action === "waves.list" ? { waves: [wave("w1")] } : { tasks: [] }
    })
    renderPanel()
    fireEvent.click(await screen.findByRole("button", { name: /Decisions/ }))
    expect(screen.getByText("answered the coder")).toBeTruthy()
  })

  it("drops a decisions.list entry without a card", async () => {
    api.callFactory.mockImplementation(async (action: string) => {
      if (action === "decisions.list") return { decisions: [{ at: "t", answers: "k0" }, decision("k1")] }
      return action === "waves.list" ? { waves: [wave("w1")] } : { tasks: [] }
    })
    renderPanel()
    fireEvent.click(await screen.findByRole("button", { name: /Decisions · 1/ }))
    expect(screen.getByText("answered the coder")).toBeTruthy()
  })
})

describe("taskModels", () => {
  it("names each task's coder model without its provider", () => {
    const listed = [
      { id: "t-a", settings: { models: { big: "anthropic/claude-opus-5-5" } } },
      { id: "t-b", settings: { models: { big: "x/y/z" } } },
      { id: "t-c", settings: null },
    ] as unknown as FactoryTask[]
    expect(taskModels(listed)).toEqual({ "t-a": "claude-opus-5-5", "t-b": "z" })
  })
})
