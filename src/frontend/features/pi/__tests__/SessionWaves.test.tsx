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
  render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenTask={onOpenTask} />)
  return onOpenTask
}

describe("SessionWaves", () => {
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
    const row = await screen.findByRole("button", { name: /p\/w1/ })
    expect(api.callFactory).toHaveBeenCalledWith("waves.list", { repo: "/git/repo" })
    expect(row.textContent).toContain("title w1")
    expect(row.textContent).toContain("2/8")
    expect(row.textContent).toContain("building")
    expect(row.textContent).toContain("$4.10")
    const header = screen.getByRole("button", { name: /^Factory/ })
    expect(header.getAttribute("aria-expanded")).toBe("true")
    expect(header.textContent).toContain("Factory · 2 active · 1 needs you")
    expect(header.textContent).toContain("$4.10")
    expect(screen.getByText("1 needs you").className).toContain("text-warning")
    expect(screen.getByText("rejected").className).toContain("text-warning")
    expect(screen.getByText("building").className).not.toContain("text-warning")
    expect(screen.queryByText(/finished/)).toBeNull()
    for (const hidden of ["p/w2", "p/w3"]) expect(screen.queryByText(hidden)).toBeNull()
    fireEvent.click(header)
    expect(screen.queryByText("p/w1")).toBeNull()
  })

  it("collapses to the header with only finished waves", () => {
    render(<SessionWaves
      waves={[wave("w4", { stage: "merged", usage: { input: 1, output: 1, cost: 1 } }), wave("w5", { stage: "cancelled", usage: { input: 1, output: 1, cost: 2 } })]}
      onOpenTask={vi.fn()}
    />)
    const header = screen.getByRole("button", { name: /^Factory/ })
    expect(header.textContent).toContain("Factory · 2 finished")
    expect(header.textContent).not.toContain("needs you")
    expect(header.getAttribute("aria-expanded")).toBe("false")
    expect(screen.queryByText("p/w4")).toBeNull()
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
    expect(screen.getByRole("button", { name: /^Factory/ }).textContent).toContain("Factory · 3 active · 1 needs you")
    const tags = screen.getAllByText(/^p\//).map((node) => node.textContent)
    expect(tags).toEqual(["p/stuck", "p/new", "p/old"])
    const finished = screen.getByRole("button", { name: /^2 finished/ })
    expect(screen.queryByText("p/m1")).toBeNull()
    fireEvent.click(finished)
    const list = screen.getByRole("list", { name: "Finished waves" })
    expect(list.textContent).toContain("2/2")
    expect(list.textContent).toContain("merged")
    expect(screen.getAllByText(/^p\//).map((node) => node.textContent)).toEqual(["p/stuck", "p/new", "p/old", "p/m2", "p/m1"])
    expect(screen.getByRole("button", { name: /p\/m1/ })).toBeTruthy()
  })

  it("opens a wave to its tasks with model and tokens, unfinished first, a started one opens its task", () => {
    const onOpenTask = vi.fn()
    const tasks = [
      task("a", { title: "feat: running one", state: "running", taskId: "t-a", sessionId: "s-a", usage: { input: 200, output: 3400, cost: 1, cacheRead: 54800, cacheWrite: 0 } }),
      task("b", { title: "feat: pending one" }),
      task("c", { title: "feat: done one", state: "done", taskId: "t-c", sessionId: "s-c" }),
      task("d", { title: "feat: cancelled one", state: "cancelled", taskId: "t-d", sessionId: "s-d" }),
    ]
    render(<SessionWaves waves={[wave("w1", { tasks })]} models={{ "t-a": "claude-opus-5-5" }} onOpenTask={onOpenTask} />)
    expect(screen.queryByText("feat: running one")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /p\/w1/ }))
    const list = screen.getByRole("list", { name: "p/w1 tasks" })
    expect(list.textContent).toContain("feat: running one")
    expect(list.textContent).toContain("running")
    expect(list.textContent).toContain("feat: pending one")
    expect(list.textContent).toContain("pending")
    expect(Array.from(list.querySelectorAll("li")).map((row) => row.textContent?.match(/feat: \w+ one/)?.[0])).toEqual(["feat: running one", "feat: pending one", "feat: done one", "feat: cancelled one"])
    expect(list.querySelector("li")?.textContent).toContain("claude-opus-5-5 · ↑55.0k ↓3.4k · running")
    const open = screen.getAllByRole("button", { name: "Open task" })
    expect(open).toHaveLength(3)
    fireEvent.click(open[0])
    expect(onOpenTask).toHaveBeenCalledWith("t-a")
  })

  it("renders nothing without waves", async () => {
    api.callFactory.mockResolvedValue({ waves: [] })
    const { container } = render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenTask={vi.fn()} />)
    await waitFor(() => expect(api.callFactory).toHaveBeenCalled())
    expect(container.innerHTML).toBe("")
  })

  it("renders nothing when the factory has no waves.list", async () => {
    api.callFactory.mockRejectedValue(new Error("unknown action: waves.list"))
    const { container } = render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenTask={vi.fn()} />)
    await waitFor(() => expect(api.callFactory).toHaveBeenCalled())
    await act(async () => {})
    expect(container.innerHTML).toBe("")
  })

  it("reports any other failed read", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {})
    const cause = new Error("invalid: repo must be an absolute path")
    api.callFactory.mockRejectedValue(cause)
    render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenTask={vi.fn()} />)
    await waitFor(() => expect(error).toHaveBeenCalledWith("waves.list failed", cause))
    error.mockRestore()
  })

  it("reads again on a factory change", async () => {
    api.callFactory.mockResolvedValueOnce({ waves: [] }).mockResolvedValueOnce({ waves: [wave("w1")] })
    renderPanel()
    await waitFor(() => expect(api.callFactory).toHaveBeenCalledTimes(1))
    act(() => {
      for (const observer of socket.observers) observer.onHostEvent?.({ type: "sessions_changed", sessionPaths: ["/agent/factory/waves/w1.jsonl"] })
    })
    expect(await screen.findByRole("button", { name: /p\/w1/ })).toBeTruthy()
  })
})

describe("taskModels", () => {
  it("names each task's coder model without its provider", () => {
    const listed = [
      { id: "t-a", settings: { models: { coder: "anthropic/claude-opus-5-5" } } },
      { id: "t-b", settings: { models: { big: "x/y/z" } } },
      { id: "t-c", settings: null },
    ] as unknown as FactoryTask[]
    expect(taskModels(listed)).toEqual({ "t-a": "claude-opus-5-5", "t-b": "z" })
  })
})
