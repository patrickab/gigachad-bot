import { describe, expect, it, vi } from "vitest"
import { renderHook } from "@testing-library/react"
import { useLatestRequest } from "../useLatestRequest"

describe("useLatestRequest", () => {
  it("hands on only the latest answer, whichever arrives first", async () => {
    const { result } = renderHook(() => useLatestRequest())
    const first = Promise.withResolvers<string>()
    const second = Promise.withResolvers<string>()
    const onResult = vi.fn()

    const firstRun = result.current.run(first.promise, onResult)
    const secondRun = result.current.run(second.promise, onResult)
    second.resolve("second")
    first.resolve("first")

    expect(await secondRun).toBe(true)
    expect(await firstRun).toBe(false)
    expect(onResult).toHaveBeenCalledExactlyOnceWith("second")
  })

  it("drops the answer and the failure of a cancelled request", async () => {
    const { result } = renderHook(() => useLatestRequest())
    const onResult = vi.fn()
    const onFailure = vi.fn()

    const answered = result.current.run(Promise.resolve("late"), onResult, onFailure)
    const failed = result.current.run(Promise.reject(new Error("late")), onResult, onFailure)
    result.current.cancel()

    expect(await answered).toBe(false)
    expect(await failed).toBe(false)
    expect(onResult).not.toHaveBeenCalled()
    expect(onFailure).not.toHaveBeenCalled()
  })

  it("hands the latest failure to onFailure, without one the run rejects with it", async () => {
    const { result } = renderHook(() => useLatestRequest())
    const onFailure = vi.fn()

    expect(await result.current.run(Promise.reject(new Error("No match")), vi.fn(), onFailure)).toBe(false)
    expect(onFailure).toHaveBeenCalledExactlyOnceWith(new Error("No match"))
    await expect(result.current.run(Promise.reject(new Error("Not allowed")), vi.fn())).rejects.toThrow("Not allowed")
  })
})
