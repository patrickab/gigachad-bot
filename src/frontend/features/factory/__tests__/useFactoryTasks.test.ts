import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { SocketObserver } from "@/features/pi/piSocket"
import type { FactoryTask, FactoryWave } from "@/lib/domain/factory"
import { factoryTask as task, factoryWave } from "./factoryTask"

const core = vi.hoisted(() => ({
  listTasks: vi.fn<(repo: string) => Promise<FactoryTask[]>>(),
  listWaves: vi.fn<(repo: string) => Promise<FactoryWave[]>>(),
}))
const api = vi.hoisted(() => ({
  callFactory: vi.fn(async (action: string, params: Record<string, unknown> = {}) => {
    if (action === "waves.list") return { waves: await core.listWaves(params.repo as string) }
    if (action !== "tasks.list") throw new Error(`unexpected ${action}`)
    return { tasks: await core.listTasks(params.repo as string) }
  }),
}))
const socket = vi.hoisted(() => ({ observers: new Set<SocketObserver>(), holds: 0 }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/pi/piSocket", () => ({
  observePiSocket: (observer: SocketObserver, { hold = false }: { hold?: boolean } = {}) => {
    socket.observers.add(observer)
    if (hold) socket.holds += 1
    return () => {
      socket.observers.delete(observer)
      if (hold) socket.holds -= 1
    }
  },
}))

import { useFactoryTask, useFactoryTasks } from "@/features/factory/useFactoryTasks"

function emit(type: string, sessionPaths?: string[]) {
  for (const observer of socket.observers) observer.onHostEvent?.(sessionPaths === undefined ? { type } : { type, sessionPaths })
}

/** Each test reads its own repo: the task lists outlive a hook. */
let repo = ""
let count = 0

beforeEach(() => {
  count += 1
  repo = `/git/repo-${count}`
  core.listTasks.mockReset().mockResolvedValue([])
  core.listWaves.mockReset().mockResolvedValue([])
})

afterEach(() => {
  vi.useRealTimers()
})

describe("useFactoryTasks", () => {
  it("reads every task of the repo by path on mount, newest first", async () => {
    core.listTasks.mockResolvedValue([task("old00001"), task("cli00001", { createdAt: "2026-01-02T00:00:00Z" })])
    const { result } = renderHook(() => useFactoryTasks(repo))

    expect(result.current.tasks).toEqual([])
    await waitFor(() => expect(result.current.tasks.map((one) => one.id)).toEqual(["cli00001", "old00001"]))
    expect(api.callFactory).toHaveBeenCalledWith("tasks.list", { repo })
    expect(result.current.error).toBeNull()
  })

  it("reads the repo's waves next to its tasks, and again with them", async () => {
    core.listWaves.mockResolvedValue([factoryWave("wave1")])
    const { result } = renderHook(() => useFactoryTasks(repo))

    await waitFor(() => expect(result.current.waves.map((wave) => wave.id)).toEqual(["wave1"]))
    expect(api.callFactory).toHaveBeenCalledWith("waves.list", { repo })

    core.listWaves.mockResolvedValue([factoryWave("wave1"), factoryWave("wave2")])
    await act(() => result.current.refresh())
    expect(result.current.waves.map((wave) => wave.id)).toEqual(["wave1", "wave2"])
    expect(core.listWaves).toHaveBeenCalledTimes(2)
  })

  it("has no waves and no error with a factory that does not know waves.list", async () => {
    core.listTasks.mockResolvedValue([task("task0010")])
    core.listWaves.mockRejectedValue(Object.assign(new Error('unknown action "waves.list"'), { code: "invalid" }))
    const { result } = renderHook(() => useFactoryTasks(repo))

    await waitFor(() => expect(result.current.tasks).toHaveLength(1))
    expect(result.current.waves).toEqual([])
    expect(result.current.error).toBeNull()
  })

  it("keeps the last waves and shows the error when waves.list fails otherwise", async () => {
    core.listWaves.mockResolvedValue([factoryWave("wave1")])
    const { result } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(result.current.waves).toHaveLength(1))

    core.listWaves.mockRejectedValueOnce(new Error("Factory is down"))
    await act(() => result.current.refresh())
    expect(result.current.error).toBe("Factory is down")
    expect(result.current.waves.map((wave) => wave.id)).toEqual(["wave1"])
  })

  it("reads again on sessions_changed and on a reopened socket, holding the socket while mounted", async () => {
    const { unmount } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(1))
    expect(socket.holds).toBe(1)

    act(() => emit("session_renamed"))
    await act(async () => {})
    expect(core.listTasks).toHaveBeenCalledTimes(1)

    act(() => emit("sessions_changed", ["/home/u/.pi/agent/factory/tasks/t1.lock"]))
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(2))

    act(() => { for (const observer of socket.observers) observer.onOpen?.() })
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(3))

    unmount()
    expect(socket.holds).toBe(0)
  })

  it("reads again only on a notice of a factory path, or of no path at all", async () => {
    renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(1))

    act(() => emit("sessions_changed", ["/home/u/.pi/agent/sessions/--home-u-git-demo--/s1.jsonl"]))
    await act(async () => {})
    expect(core.listTasks).toHaveBeenCalledTimes(1)

    act(() => emit("sessions_changed", ["/home/u/.pi/agent/sessions/--x--/s2.jsonl", "/home/u/.pi/agent/factory/sessions/s3.jsonl"]))
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(2))

    act(() => emit("sessions_changed", []))
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(3))
  })

  it("does not poll", async () => {
    vi.useFakeTimers()
    renderHook(() => useFactoryTasks(repo))
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(core.listTasks).toHaveBeenCalledTimes(1)

    await act(async () => { await vi.advanceTimersByTimeAsync(10 * 60_000) })
    expect(core.listTasks).toHaveBeenCalledTimes(1)
  })

  it("reads again on refresh, resolving once the newer tasks are in", async () => {
    const { result } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(core.listTasks).toHaveBeenCalledTimes(1))

    core.listTasks.mockResolvedValue([task("task0002")])
    await act(() => result.current.refresh())
    expect(result.current.tasks).toEqual([task("task0002")])
  })

  it("answers a refresh asked during a read with a read started after it", async () => {
    let finishFirst: (tasks: FactoryTask[]) => void = () => {}
    core.listTasks.mockReturnValueOnce(new Promise((resolve) => { finishFirst = resolve }))
    const { result } = renderHook(() => useFactoryTasks(repo))
    expect(core.listTasks).toHaveBeenCalledTimes(1)

    core.listTasks.mockResolvedValue([task("task0003")])
    let refreshed: Promise<void> = Promise.resolve()
    act(() => { refreshed = Promise.all([result.current.refresh(), result.current.refresh()]).then(() => {}) })
    expect(core.listTasks).toHaveBeenCalledTimes(1)

    await act(async () => {
      finishFirst([task("stale001")])
      await refreshed
    })
    expect(core.listTasks).toHaveBeenCalledTimes(2)
    expect(result.current.tasks).toEqual([task("task0003")])
  })

  it("keeps the last tasks when a read fails, and clears the error with the next one", async () => {
    core.listTasks.mockResolvedValue([task("task0004")])
    const { result } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(result.current.tasks).toHaveLength(1))

    core.listTasks.mockRejectedValueOnce(new Error("Unknown repo"))
    await act(() => result.current.refresh())
    expect(result.current.error).toBe("Unknown repo")
    expect(result.current.tasks).toEqual([task("task0004")])

    await act(() => result.current.refresh())
    expect(result.current.error).toBeNull()
  })
})

describe("useFactoryTask", () => {
  it("follows the task as the list of its repo has it", async () => {
    core.listTasks.mockResolvedValue([task("task0005")])
    const list = renderHook(() => useFactoryTasks(repo))
    const one = renderHook(() => useFactoryTask("task0005"))
    expect(renderHook(() => useFactoryTask("unknown0")).result.current).toBeUndefined()
    await waitFor(() => expect(one.result.current).toEqual(task("task0005")))

    core.listTasks.mockResolvedValue([task("task0005", { state: "done", round: 2, running: false })])
    await act(() => list.result.current.refresh())
    expect(one.result.current?.state).toBe("done")

    core.listTasks.mockResolvedValue([])
    await act(() => list.result.current.refresh())
    expect(one.result.current).toBeUndefined()
  })
})
