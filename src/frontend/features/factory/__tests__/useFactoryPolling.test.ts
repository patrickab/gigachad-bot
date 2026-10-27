import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { renderHook, act } from "@testing-library/react"
import type { FactoryEvent } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  fetchLatestFactoryEventSeq: vi.fn<() => Promise<number>>(),
  fetchFactoryEvents: vi.fn<(repo: string, after: number) => Promise<FactoryEvent[]>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { POLL_INTERVAL_MS, useFactoryPolling } from "@/features/factory/useFactoryPolling"

function event(seq: number, taskId: string): FactoryEvent {
  return { seq, task_id: taskId, kind: "updated", data: {}, created_at: "" }
}

/** Run pending promise callbacks, advancing the fake clock by `ms`. */
async function advance(ms: number) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}

beforeEach(() => {
  vi.useFakeTimers()
  api.fetchLatestFactoryEventSeq.mockReset().mockResolvedValue(5)
  api.fetchFactoryEvents.mockReset().mockResolvedValue([])
})
afterEach(() => vi.useRealTimers())

describe("useFactoryPolling", () => {
  it("reloads everything once and starts the cursor at the latest event", async () => {
    const refresh = vi.fn().mockResolvedValue(undefined)
    renderHook(() => useFactoryPolling("demo", refresh))
    await advance(0)

    expect(refresh).toHaveBeenCalledTimes(1)
    expect(refresh).toHaveBeenCalledWith(null)

    await advance(POLL_INTERVAL_MS)
    expect(api.fetchFactoryEvents).toHaveBeenCalledWith("demo", 5)
  })

  it("refreshes the tasks that had events and advances the cursor past them", async () => {
    api.fetchFactoryEvents.mockResolvedValueOnce([event(6, "a"), event(7, "b"), event(8, "a")])
    const refresh = vi.fn().mockResolvedValue(undefined)
    renderHook(() => useFactoryPolling("demo", refresh))
    await advance(0)

    await advance(POLL_INTERVAL_MS)
    expect(refresh).toHaveBeenLastCalledWith(new Set(["a", "b"]))

    await advance(POLL_INTERVAL_MS)
    expect(api.fetchFactoryEvents).toHaveBeenLastCalledWith("demo", 8)
  })

  it("does not advance the cursor when refresh fails and retries on the next tick", async () => {
    api.fetchFactoryEvents.mockResolvedValue([event(6, "a")])
    const refresh = vi.fn().mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("engine down")).mockResolvedValue(undefined)
    const { result } = renderHook(() => useFactoryPolling("demo", refresh))
    await advance(0)

    await advance(POLL_INTERVAL_MS)
    expect(result.current).toBe("engine down")

    await advance(POLL_INTERVAL_MS)
    expect(api.fetchFactoryEvents).toHaveBeenNthCalledWith(1, "demo", 5)
    expect(api.fetchFactoryEvents).toHaveBeenNthCalledWith(2, "demo", 5)
    expect(refresh).toHaveBeenCalledTimes(3)
    expect(result.current).toBeNull()

    await advance(POLL_INTERVAL_MS)
    expect(api.fetchFactoryEvents).toHaveBeenLastCalledWith("demo", 6)
  })

  it("retries the full reload when the first refresh fails", async () => {
    const refresh = vi.fn().mockRejectedValueOnce(new Error("boom")).mockResolvedValue(undefined)
    renderHook(() => useFactoryPolling("demo", refresh))
    await advance(0)
    expect(api.fetchFactoryEvents).not.toHaveBeenCalled()

    await advance(POLL_INTERVAL_MS)
    expect(refresh).toHaveBeenCalledTimes(2)
    expect(refresh).toHaveBeenLastCalledWith(null)

    await advance(POLL_INTERVAL_MS)
    expect(api.fetchFactoryEvents).toHaveBeenCalledWith("demo", 5)
  })

  it("skips a tick while the previous one is still running", async () => {
    let finishRefresh = () => {}
    const refresh = vi.fn(() => new Promise<void>((resolve) => { finishRefresh = resolve }))
    renderHook(() => useFactoryPolling("demo", refresh))
    await advance(0)

    await advance(POLL_INTERVAL_MS * 3)
    expect(api.fetchLatestFactoryEventSeq).toHaveBeenCalledTimes(1)
    expect(refresh).toHaveBeenCalledTimes(1)

    finishRefresh()
    await advance(POLL_INTERVAL_MS)
    expect(api.fetchFactoryEvents).toHaveBeenCalledWith("demo", 5)
  })
})
