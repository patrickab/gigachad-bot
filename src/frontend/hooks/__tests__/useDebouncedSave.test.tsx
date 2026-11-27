import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import { useDebouncedSave } from "../useDebouncedSave"

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const elapse = (ms: number) => act(() => { vi.advanceTimersByTime(ms) })

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
})
