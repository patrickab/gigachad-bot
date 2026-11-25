import { afterEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import { fetchPiRepositories } from "@/lib/backend/api"
import { usePiFolder } from "../TabContent"

vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiRepositories: vi.fn(),
}))

describe("usePiFolder", () => {
  afterEach(() => vi.clearAllMocks())

  it("takes a checkout under ~/git", async () => {
    vi.mocked(fetchPiRepositories).mockResolvedValue([{ name: "repo", path: "/home/u/git/repo" }])
    const { result } = renderHook(() => usePiFolder())
    act(() => result.current.pickFolder("/home/u/git/repo/"))
    await waitFor(() => expect(result.current.folder).toBe("/home/u/git/repo"))
    expect(result.current.folderError).toBeNull()
  })

  it("keeps the latest of overlapping picks", async () => {
    const first = Promise.withResolvers<{ name: string; path: string }[]>()
    vi.mocked(fetchPiRepositories)
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce([{ name: "b", path: "/home/u/git/b" }])
    const { result } = renderHook(() => usePiFolder())
    act(() => result.current.pickFolder("/home/u/git/a"))
    act(() => result.current.pickFolder("/home/u/git/b"))
    await waitFor(() => expect(result.current.folder).toBe("/home/u/git/b"))
    await act(async () => first.resolve([{ name: "a", path: "/home/u/git/a" }]))
    expect(result.current.folder).toBe("/home/u/git/b")
  })
})
