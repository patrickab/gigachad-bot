import { afterEach, describe, expect, it, vi } from "vitest"
import type { PiSessionMatch } from "@/lib/backend/api"

const api = vi.hoisted(() => ({ matches: vi.fn() }))
vi.mock("@/lib/backend/api", () => ({ fetchPiSessionMatches: api.matches }))

import { clearPiSearchHit, focusPiSearchHit, stepPiSearchHit, usePiSearchFocus } from "../searchFocus"
import { act, renderHook } from "@testing-library/react"

const hit = (entryId: string, ordinal = 0): PiSessionMatch => ({ entryId, ordinal, quality: 1, before: "", match: "x", after: "" })

afterEach(() => {
  clearPiSearchHit()
  api.matches.mockReset()
})

describe("session search focus", () => {
  it("shows the picked hit at once and swaps in the session's whole ranked list under it", async () => {
    api.matches.mockResolvedValue({ matches: [hit("a"), hit("b", 1), hit("c")], total: 3 })
    const { result } = renderHook(() => usePiSearchFocus("s1"))

    act(() => focusPiSearchHit("s1", "x", hit("b", 1), [hit("b", 1)]))
    const first = result.current
    expect(first).toMatchObject({ query: "x", index: 0, matches: [hit("b", 1)] })

    await act(async () => { await Promise.resolve() })
    expect(result.current).toMatchObject({ index: 1, matches: [hit("a"), hit("b", 1), hit("c")], seq: first!.seq })
    expect(api.matches).toHaveBeenCalledWith("s1", "x", 200)
  })

  it("steps with wrap-around, each step a new jump, and clears", async () => {
    api.matches.mockResolvedValue({ matches: [hit("a"), hit("b")], total: 2 })
    const { result } = renderHook(() => usePiSearchFocus("s1"))
    act(() => focusPiSearchHit("s1", "x", hit("a"), [hit("a"), hit("b")]))
    await act(async () => { await Promise.resolve() })

    const seqs = [result.current!.seq]
    act(() => stepPiSearchHit(-1))
    seqs.push(result.current!.seq)
    expect(result.current!.index).toBe(1)
    act(() => stepPiSearchHit(1))
    seqs.push(result.current!.seq)
    expect(result.current!.index).toBe(0)
    expect(new Set(seqs).size).toBe(3)

    act(() => clearPiSearchHit())
    expect(result.current).toBeNull()
  })

  it("is only for the session it names, and a failed list read keeps the listed hits", async () => {
    api.matches.mockRejectedValue(new Error("down"))
    const other = renderHook(() => usePiSearchFocus("s2"))
    const { result } = renderHook(() => usePiSearchFocus("s1"))
    act(() => focusPiSearchHit("s1", "x", hit("a"), [hit("a"), hit("b")]))
    await act(async () => { await Promise.resolve() })
    expect(other.result.current).toBeNull()
    expect(result.current?.matches).toHaveLength(2)
  })
})
