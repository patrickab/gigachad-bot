import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"
import { useLoaded } from "@/hooks/useLoaded"

afterEach(cleanup)

/** A `load` whose reads the test settles by key. */
function deferredLoad() {
  const pending = new Map<string, { resolve: (value: string) => void; reject: (cause: unknown) => void }>()
  const load = (key: string) => new Promise<string>((resolve, reject) => { pending.set(key, { resolve, reject }) })
  return { load, pending }
}

describe("useLoaded", () => {
  it("shows the value of the current key and drops a late answer for an earlier one", async () => {
    const { load, pending } = deferredLoad()
    const { result, rerender } = renderHook(({ key }) => useLoaded(key, load), { initialProps: { key: "a" as string | null } })
    expect(result.current).toEqual({ value: undefined, error: null })

    rerender({ key: "b" })
    await act(async () => { pending.get("a")!.resolve("A") })
    expect(result.current.value).toBeUndefined()

    await act(async () => { pending.get("b")!.resolve("B") })
    expect(result.current).toEqual({ value: "B", error: null })

    rerender({ key: null })
    expect(result.current).toEqual({ value: undefined, error: null })
  })

  it("keys a failure to its key", async () => {
    const { load, pending } = deferredLoad()
    const { result, rerender } = renderHook(({ key }) => useLoaded(key, load), { initialProps: { key: "a" } })
    const cause = new Error("gone")
    await act(async () => { pending.get("a")!.reject(cause) })
    expect(result.current).toEqual({ value: undefined, error: cause })

    rerender({ key: "b" })
    expect(result.current).toEqual({ value: undefined, error: null })
  })
})
