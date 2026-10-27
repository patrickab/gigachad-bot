import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { usePolling } from "@/hooks/usePolling"

function setVisibility(state: "visible" | "hidden") {
  Object.defineProperty(document, "visibilityState", { configurable: true, value: state })
  Object.defineProperty(document, "hidden", { configurable: true, value: state === "hidden" })
  document.dispatchEvent(new Event("visibilitychange"))
}

describe("usePolling", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setVisibility("visible")
  })
  afterEach(() => {
    vi.useRealTimers()
    setVisibility("visible")
  })

  it("runs at once, then per interval, skipping ticks while a run is in flight", async () => {
    let finish = () => {}
    const load = vi.fn(() => {
      const { promise, resolve } = Promise.withResolvers<void>()
      finish = resolve
      return promise
    })
    renderHook(() => usePolling(load, { intervalMs: 1000 }))
    expect(load).toHaveBeenCalledTimes(1)

    await act(async () => { vi.advanceTimersByTime(1000) })
    expect(load).toHaveBeenCalledTimes(1)

    await act(async () => { finish() })
    await act(async () => { vi.advanceTimersByTime(1000) })
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("pauses while the page is hidden and runs again once it shows", async () => {
    const load = vi.fn(async () => {})
    renderHook(() => usePolling(load, { intervalMs: 1000 }))
    await act(async () => { setVisibility("hidden") })
    await act(async () => { vi.advanceTimersByTime(5000) })
    expect(load).toHaveBeenCalledTimes(1)

    await act(async () => { setVisibility("visible") })
    expect(load).toHaveBeenCalledTimes(2)
  })

  it("stops on a null interval and aborts the signal of the stopped run", async () => {
    const signals: AbortSignal[] = []
    const load = vi.fn(async (signal: AbortSignal) => { signals.push(signal) })
    const { rerender } = renderHook(({ intervalMs }) => usePolling(load, { intervalMs }), { initialProps: { intervalMs: 1000 as number | null } })
    rerender({ intervalMs: null })
    await act(async () => { vi.advanceTimersByTime(5000) })
    expect(load).toHaveBeenCalledTimes(1)
    expect(signals[0]!.aborted).toBe(true)
  })
})
