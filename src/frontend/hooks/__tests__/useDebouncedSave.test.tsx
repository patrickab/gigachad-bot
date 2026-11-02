import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import { useDebouncedCall, useDebouncedSave } from "../useDebouncedSave"

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const elapse = (ms: number) => act(() => { vi.advanceTimersByTime(ms) })

describe("useDebouncedCall", () => {
  it("runs the latest scheduled action once after the quiet time", () => {
    const first = vi.fn()
    const second = vi.fn()
    const { result } = renderHook(() => useDebouncedCall(100))
    result.current.schedule(first)
    elapse(60)
    result.current.schedule(second)
    elapse(60)
    expect(second).not.toHaveBeenCalled()
    elapse(40)
    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledTimes(1)
  })

  it("flushes a pending action at once and does nothing without one", () => {
    const action = vi.fn()
    const { result } = renderHook(() => useDebouncedCall(100))
    result.current.flush()
    result.current.schedule(action)
    result.current.flush()
    expect(action).toHaveBeenCalledTimes(1)
    elapse(200)
    expect(action).toHaveBeenCalledTimes(1)
  })

  it("drops a cancelled action", () => {
    const action = vi.fn()
    const { result } = renderHook(() => useDebouncedCall(100))
    result.current.schedule(action)
    result.current.cancel()
    elapse(200)
    expect(action).not.toHaveBeenCalled()
  })

  it("runs once maxWaitMs after the first schedule although the source never goes quiet", () => {
    const actions = [vi.fn(), vi.fn(), vi.fn(), vi.fn()]
    const { result } = renderHook(() => useDebouncedCall(100, 250))
    for (const action of actions.slice(0, 3)) {
      result.current.schedule(action)
      elapse(80)
    }
    result.current.schedule(actions[3])
    elapse(9)
    expect(actions[3]).not.toHaveBeenCalled()
    elapse(1)
    expect(actions[3]).toHaveBeenCalledTimes(1)
    expect(actions.slice(0, 3).every((action) => action.mock.calls.length === 0)).toBe(true)
    result.current.schedule(actions[0])
    elapse(99)
    expect(actions[0]).not.toHaveBeenCalled()
  })
})

describe("useDebouncedSave", () => {
  it("saves a value once it has been quiet, and each new value restarts the wait", () => {
    const save = vi.fn()
    const { rerender } = renderHook(({ value }) => useDebouncedSave(value, save, 100), { initialProps: { value: "a" } })
    elapse(60)
    rerender({ value: "b" })
    elapse(60)
    expect(save).not.toHaveBeenCalled()
    elapse(40)
    expect(save).toHaveBeenCalledOnce()
    expect(save).toHaveBeenCalledWith("b", false)
  })

  it("saves nothing for a null value and the last value once more on unmount", () => {
    const save = vi.fn()
    const empty = renderHook(() => useDebouncedSave<string>(null, save, 100))
    empty.unmount()
    expect(save).not.toHaveBeenCalled()

    const { unmount } = renderHook(() => useDebouncedSave("a", save, 100))
    unmount()
    expect(save).toHaveBeenCalledOnce()
    expect(save).toHaveBeenCalledWith("a", false)
  })

  it("flushes on request, forcing when asked", () => {
    const save = vi.fn()
    const { result } = renderHook(() => useDebouncedSave("a", save, 100))
    result.current(true)
    expect(save).toHaveBeenCalledWith("a", true)
  })
})
