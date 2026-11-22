import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { PiSessionInfo } from "@/lib/backend/api/pi"
import type { FactoryState, FactoryTask } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  listFactoryTasks: vi.fn<(repo: string) => Promise<FactoryTask[]>>(),
}))
const store = vi.hoisted(() => ({ sessions: [] as PiSessionInfo[], filters: [] as unknown[] }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/pi/sessionStore", () => ({
  usePiSessions: (filter: unknown) => {
    store.filters.push(filter)
    return { sessions: store.sessions, loading: false, error: null }
  },
}))

import { useFactoryTask, useFactoryTasks } from "@/features/factory/useFactoryTasks"

const RUNNING: FactoryState = { state: "running", round: 1, reason: "", at: "2026-01-01T00:00:00Z" }

function session(sessionId: string, factory: FactoryState | null = null, running = false): PiSessionInfo {
  return {
    sessionId, sessionPath: `/sessions/${sessionId}.jsonl`, cwd: "/git/repo", machine: "local", createdAt: "", modifiedAt: "", firstMessage: "",
    messageCount: 0, tokens: 0, live: { running, waiting: false, attached: 0 }, pinned: false, archived: false, factory,
  }
}

function task(id: string, factory: FactoryState | null = RUNNING): FactoryTask {
  return { id, repo: "/git/repo", session_id: `s-${id}`, session_path: "", worktree: null, branch: null, title: id, created_at: "2026-01-01T00:00:00Z", factory, live: null, usage: null }
}

/** Each test reads its own repo: the task lists outlive a hook. */
let repo = ""
let count = 0

beforeEach(() => {
  count += 1
  repo = `/git/repo-${count}`
  store.sessions = []
  api.listFactoryTasks.mockReset().mockResolvedValue([])
})

afterEach(() => {
  vi.useRealTimers()
})

describe("useFactoryTasks", () => {
  it("reads the repo's tasks by path on mount", async () => {
    api.listFactoryTasks.mockResolvedValue([task("task0001")])
    const { result } = renderHook(() => useFactoryTasks(repo))

    expect(result.current.tasks).toEqual([])
    await waitFor(() => expect(result.current.tasks).toEqual([task("task0001")]))
    expect(api.listFactoryTasks).toHaveBeenCalledWith(repo)
    expect(result.current.error).toBeNull()
    // The factory's own session list: factory tasks are in no other
    expect(store.filters.at(-1)).toEqual({ factory: true })
  })

  it("reads again when the session store reports a new session, factory state or running child, and not otherwise", async () => {
    store.sessions = [session("one")]
    const { rerender } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(api.listFactoryTasks).toHaveBeenCalledTimes(1))

    // The store read its list again and nothing a task shows changed.
    store.sessions = [{ ...session("one"), modifiedAt: "later", tokens: 5 }]
    rerender()
    await act(async () => {})
    expect(api.listFactoryTasks).toHaveBeenCalledTimes(1)

    store.sessions = [session("one"), session("two")]
    rerender()
    await waitFor(() => expect(api.listFactoryTasks).toHaveBeenCalledTimes(2))

    store.sessions = [session("one"), session("two", RUNNING)]
    rerender()
    await waitFor(() => expect(api.listFactoryTasks).toHaveBeenCalledTimes(3))

    store.sessions = [session("one"), session("two", RUNNING, true)]
    rerender()
    await waitFor(() => expect(api.listFactoryTasks).toHaveBeenCalledTimes(4))

    store.sessions = [session("one"), session("two", { ...RUNNING, state: "waiting_approve", at: "2026-01-01T00:05:00Z" }, true)]
    rerender()
    await waitFor(() => expect(api.listFactoryTasks).toHaveBeenCalledTimes(5))
  })

  it("does not poll", async () => {
    vi.useFakeTimers()
    renderHook(() => useFactoryTasks(repo))
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(api.listFactoryTasks).toHaveBeenCalledTimes(1)

    await act(async () => { await vi.advanceTimersByTimeAsync(10 * 60_000) })
    expect(api.listFactoryTasks).toHaveBeenCalledTimes(1)
  })

  it("reads again on refresh, resolving once the newer tasks are in", async () => {
    const { result } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(api.listFactoryTasks).toHaveBeenCalledTimes(1))

    api.listFactoryTasks.mockResolvedValue([task("task0002")])
    await act(() => result.current.refresh())
    expect(result.current.tasks).toEqual([task("task0002")])
  })

  it("answers a refresh asked during a read with a read started after it", async () => {
    let finishFirst: (tasks: FactoryTask[]) => void = () => {}
    api.listFactoryTasks.mockReturnValueOnce(new Promise((resolve) => { finishFirst = resolve }))
    const { result } = renderHook(() => useFactoryTasks(repo))
    expect(api.listFactoryTasks).toHaveBeenCalledTimes(1)

    api.listFactoryTasks.mockResolvedValue([task("task0003")])
    let refreshed: Promise<void> = Promise.resolve()
    act(() => { refreshed = Promise.all([result.current.refresh(), result.current.refresh()]).then(() => {}) })
    expect(api.listFactoryTasks).toHaveBeenCalledTimes(1)

    await act(async () => {
      finishFirst([task("stale001")])
      await refreshed
    })
    expect(api.listFactoryTasks).toHaveBeenCalledTimes(2)
    expect(result.current.tasks).toEqual([task("task0003")])
  })

  it("keeps the last tasks when a read fails, and clears the error with the next one", async () => {
    api.listFactoryTasks.mockResolvedValue([task("task0004")])
    const { result } = renderHook(() => useFactoryTasks(repo))
    await waitFor(() => expect(result.current.tasks).toHaveLength(1))

    api.listFactoryTasks.mockRejectedValueOnce(new Error("Unknown repo"))
    await act(() => result.current.refresh())
    expect(result.current.error).toBe("Unknown repo")
    expect(result.current.tasks).toEqual([task("task0004")])

    await act(() => result.current.refresh())
    expect(result.current.error).toBeNull()
  })
})

describe("useFactoryTask", () => {
  it("follows the task as the list of its repo has it", async () => {
    api.listFactoryTasks.mockResolvedValue([task("task0005")])
    const list = renderHook(() => useFactoryTasks(repo))
    const one = renderHook(() => useFactoryTask("task0005"))
    expect(renderHook(() => useFactoryTask("unknown0")).result.current).toBeUndefined()
    await waitFor(() => expect(one.result.current).toEqual(task("task0005")))

    const done: FactoryState = { state: "done", round: 2, reason: "", at: "2026-01-01T01:00:00Z" }
    api.listFactoryTasks.mockResolvedValue([task("task0005", done)])
    await act(() => list.result.current.refresh())
    expect(one.result.current?.factory).toEqual(done)

    api.listFactoryTasks.mockResolvedValue([])
    await act(() => list.result.current.refresh())
    expect(one.result.current).toBeUndefined()
  })
})
