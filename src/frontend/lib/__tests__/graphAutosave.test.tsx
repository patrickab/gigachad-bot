import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { AUTOSAVE_DEBOUNCE_MS, AUTOSAVE_MAX_WAIT_MS, useGraphAutosave } from "@/lib/graphAutosave"

// Each test uses a distinct key: ownership is tracked in a module-level registry.
let counter = 0
const freshKey = () => `graph-${++counter}.architecture.yaml`

function setup(key: string, write = vi.fn(async (_content: string) => {})) {
  const view = renderHook(() => useGraphAutosave({ key, write }))
  return { write, ...view }
}

const advance = async (ms: number) => {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}

describe("useGraphAutosave", () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it("writes once after the debounce settles", async () => {
    const { write, result } = setup(freshKey())

    act(() => result.current.queue("a"))
    await advance(AUTOSAVE_DEBOUNCE_MS - 1)
    expect(write).not.toHaveBeenCalled()

    await advance(2)
    expect(write).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith("a")
  })

  it("still writes during sustained typing, which a pure debounce never would", async () => {
    const { write, result } = setup(freshKey())

    // Keep re-queueing faster than the debounce for longer than the max wait.
    for (let elapsed = 0; elapsed < AUTOSAVE_MAX_WAIT_MS + 200; elapsed += 100) {
      act(() => result.current.queue(`draft-${elapsed}`))
      await advance(100)
    }

    expect(write).toHaveBeenCalled()
    expect(write.mock.calls[0][0]).toMatch(/^draft-/)
  })

  it("does not write when content matches what was last saved", async () => {
    const { write, result } = setup(freshKey())

    act(() => result.current.markSaved("same"))
    act(() => result.current.queue("same"))
    await advance(AUTOSAVE_MAX_WAIT_MS + 100)

    expect(write).not.toHaveBeenCalled()
    expect(result.current.dirty).toBe(false)
  })

  it("cancel drops pending content without writing it", async () => {
    const { write, result } = setup(freshKey())

    act(() => result.current.queue("half-typed"))
    act(() => result.current.cancel())
    await advance(AUTOSAVE_MAX_WAIT_MS + 100)

    expect(write).not.toHaveBeenCalled()
  })

  it("flushes pending content on unmount", async () => {
    const { write, result, unmount } = setup(freshKey())

    act(() => result.current.queue("unmounted"))
    await act(async () => { unmount() })

    expect(write).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith("unmounted")
  })

  it("flushes pending content when the document is hidden", async () => {
    const { write, result } = setup(freshKey())

    act(() => result.current.queue("backgrounded"))
    Object.defineProperty(document, "visibilityState", { value: "hidden", configurable: true })
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")) })

    expect(write).toHaveBeenCalledTimes(1)
    expect(write).toHaveBeenCalledWith("backgrounded")
    Object.defineProperty(document, "visibilityState", { value: "visible", configurable: true })
  })

  it("makes a second view of the same graph flush the first before taking over", async () => {
    const key = freshKey()
    const first = setup(key)
    const second = setup(key, vi.fn(async (_content: string) => {}))

    act(() => first.result.current.queue("from-first"))
    // Second view touches the same file: first must land now, not on a later timer.
    await act(async () => { second.result.current.queue("from-second") })

    expect(first.write).toHaveBeenCalledTimes(1)
    expect(first.write).toHaveBeenCalledWith("from-first")

    await advance(AUTOSAVE_DEBOUNCE_MS + 10)
    expect(second.write).toHaveBeenCalledTimes(1)
    expect(second.write).toHaveBeenCalledWith("from-second")
    // The handed-off view stays quiet rather than waking up with stale content.
    expect(first.write).toHaveBeenCalledTimes(1)
  })

  it("reports dirty only while unsaved content exists", async () => {
    const { result } = setup(freshKey())
    expect(result.current.dirty).toBe(false)

    act(() => result.current.queue("x"))
    expect(result.current.dirty).toBe(true)

    await advance(AUTOSAVE_DEBOUNCE_MS + 10)
    expect(result.current.dirty).toBe(false)
  })
})
