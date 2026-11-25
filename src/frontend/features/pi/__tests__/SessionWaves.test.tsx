import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { SocketObserver } from "@/features/pi/piSocket"
import type { FactoryWave, FactoryWaveTask } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({ callFactory: vi.fn() }))
const socket = vi.hoisted(() => ({ observers: new Set<SocketObserver>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/pi/piSocket", () => ({
  observePiSocket: (observer: SocketObserver) => {
    socket.observers.add(observer)
    return () => socket.observers.delete(observer)
  },
}))

import { SessionWaves, SessionWavesPanel } from "../SessionWaves"

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

function renderPanel(onOpenSession = vi.fn()) {
  render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenSession={onOpenSession} />)
  return onOpenSession
}

describe("SessionWaves", () => {
  it("shows only this session's unfinished waves", async () => {
    api.callFactory.mockResolvedValue({
      waves: [
        wave("w1", { counts: { total: 8, done: 2 }, usage: { input: 1, output: 1, cost: 4.1 } }),
        wave("w2", { origin: "other" }),
        wave("w3", { origin: null }),
        wave("w4", { stage: "merged" }),
        wave("w5", { stage: "cancelled" }),
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
    expect(screen.getByText("rejected").className).toContain("text-warning")
    expect(screen.getByText("building").className).not.toContain("text-warning")
    for (const hidden of ["p/w2", "p/w3", "p/w4", "p/w5"]) expect(screen.queryByText(hidden)).toBeNull()
  })

  it("opens a wave to its unfinished tasks, a started one opens its session", () => {
    const onOpenSession = vi.fn()
    const tasks = [
      task("a", { title: "feat: running one", state: "running", taskId: "t-a", sessionId: "s-a" }),
      task("b", { title: "feat: pending one" }),
      task("c", { title: "feat: done one", state: "done", taskId: "t-c", sessionId: "s-c" }),
      task("d", { title: "feat: cancelled one", state: "cancelled", taskId: "t-d", sessionId: "s-d" }),
    ]
    render(<SessionWaves waves={[wave("w1", { tasks })]} onOpenSession={onOpenSession} />)
    expect(screen.queryByText("feat: running one")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: /p\/w1/ }))
    const list = screen.getByRole("list", { name: "p/w1 tasks" })
    expect(list.textContent).toContain("feat: running one")
    expect(list.textContent).toContain("running")
    expect(list.textContent).toContain("feat: pending one")
    expect(list.textContent).toContain("pending")
    expect(list.textContent).not.toContain("done one")
    expect(list.textContent).not.toContain("cancelled one")
    const open = screen.getAllByRole("button", { name: "Open session" })
    expect(open).toHaveLength(1)
    fireEvent.click(open[0])
    expect(onOpenSession).toHaveBeenCalledWith("s-a")
  })

  it("renders nothing without waves", async () => {
    api.callFactory.mockResolvedValue({ waves: [] })
    const { container } = render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenSession={vi.fn()} />)
    await waitFor(() => expect(api.callFactory).toHaveBeenCalled())
    expect(container.innerHTML).toBe("")
  })

  it("renders nothing when the factory has no waves.list", async () => {
    api.callFactory.mockRejectedValue(new Error("unknown action: waves.list"))
    const { container } = render(<SessionWavesPanel repo="/git/repo" sessionId="me" onOpenSession={vi.fn()} />)
    await waitFor(() => expect(api.callFactory).toHaveBeenCalled())
    await act(async () => {})
    expect(container.innerHTML).toBe("")
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
