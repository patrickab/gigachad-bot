import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import { useDebouncedSearch } from "../useDebouncedSearch"

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

const elapse = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms) })

describe("useDebouncedSearch", () => {
  it("asks nothing for a blank query", async () => {
    const search = vi.fn()
    const { result } = renderHook(() => useDebouncedSearch("", search))
    await elapse(500)
    expect(search).not.toHaveBeenCalled()
    expect(result.current).toEqual({ hits: null, error: "" })
  })

  it("aborts a superseded request and answers the current query", async () => {
    const signals: AbortSignal[] = []
    const pending: ((hits: string[]) => void)[] = []
    const search = (_query: string, signal: AbortSignal) => {
      signals.push(signal)
      return new Promise<string[]>((resolve) => { pending.push(resolve) })
    }
    const { result, rerender } = renderHook(({ query }) => useDebouncedSearch(query, search), { initialProps: { query: "a" } })
    await elapse(200)
    rerender({ query: "ab" })
    expect(signals[0].aborted).toBe(true)
    await elapse(200)
    await act(async () => { pending[0](["stale"]); pending[1](["fresh"]) })
    expect(result.current).toEqual({ hits: ["fresh"], error: "" })
  })

  it("maps a failure to its message", async () => {
    const search = () => Promise.reject(new Error("backend down"))
    const { result } = renderHook(() => useDebouncedSearch("a", search))
    await elapse(200)
    expect(result.current).toEqual({ hits: [], error: "backend down" })
  })
})
