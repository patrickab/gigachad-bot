import { afterEach, describe, expect, it } from "vitest"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { useSlidePosition } from "../useSlidePosition"

afterEach(cleanup)

describe("useSlidePosition", () => {
  it("syncs the slide both ways between windows on one channel, clamped to the deck", async () => {
    const presenter = renderHook(() => useSlidePosition("sync-both-ways", 5))
    const audience = renderHook(() => useSlidePosition("sync-both-ways", 5))

    act(() => presenter.result.current.go(4))
    await waitFor(() => expect(audience.result.current.page).toBe(4))

    act(() => audience.result.current.go(2))
    await waitFor(() => expect(presenter.result.current.page).toBe(2))

    act(() => presenter.result.current.go(99))
    expect(presenter.result.current.page).toBe(5)
    await waitFor(() => expect(audience.result.current.page).toBe(5))

    act(() => audience.result.current.go(-3))
    await waitFor(() => expect(presenter.result.current.page).toBe(1))
  })

  it("starts a joining window on the slide the others show", async () => {
    const presenter = renderHook(() => useSlidePosition("sync-join", 5))
    act(() => presenter.result.current.go(3))

    const audience = renderHook(() => useSlidePosition("sync-join", 5))
    await waitFor(() => expect(audience.result.current.page).toBe(3))
  })

  it("ignores windows on another channel", async () => {
    const presenter = renderHook(() => useSlidePosition("sync-mine", 5))
    const other = renderHook(() => useSlidePosition("sync-other", 5))
    const audience = renderHook(() => useSlidePosition("sync-mine", 5))

    act(() => other.result.current.go(4))
    act(() => presenter.result.current.go(2))
    await waitFor(() => expect(audience.result.current.page).toBe(2))
    expect(other.result.current.page).toBe(4)
  })
})
