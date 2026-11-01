import { afterEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { LatexBuild } from "@/lib/domain/latex"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  buildLatex: vi.fn<(path: string) => Promise<LatexBuild>>(),
  folderStamp: vi.fn<(folder: string) => Promise<string>>(),
}))
vi.mock("@/lib/backend/api", () => api)
// The tests tick the folder watch by hand, the hook's own timing is `usePolling`'s business
const watch = vi.hoisted(() => ({
  tick: null as (() => Promise<unknown> | void) | null,
  options: null as { intervalMs: number | null; active?: boolean } | null,
}))
vi.mock("@/hooks/usePolling", () => ({
  usePolling: (load: () => Promise<unknown> | void, options: { intervalMs: number | null; active?: boolean }) => {
    watch.tick = load
    watch.options = options
  },
}))

import { useLatexBuild } from "../useLatexBuild"

const FOLDER = "/data/workbench/latex/proj/paper"
const OLD = `${FOLDER}/paper.tex`
const NEW = `${FOLDER}/thesis.tex`
const built = (pdf: string) => ({ ok: true, pdf_path: pdf, duration_ms: 1, errors: [], log_tail: "" }) as unknown as LatexBuild
const options = (streaming = false) => ({ folder: FOLDER, streaming })
/** One look at the folder. */
const tick = () => act(async () => { await watch.tick?.() })

afterEach(() => {
  vi.clearAllMocks()
  watch.tick = null
  watch.options = null
})

describe("useLatexBuild", () => {
  it("builds nothing and watches nothing without a root", async () => {
    const { result } = renderHook(() => useLatexBuild(null, options()))
    await act(() => result.current.trigger())
    expect(api.buildLatex).not.toHaveBeenCalled()
    expect(watch.options?.active).toBe(false)
  })

  it("starts clean for another root so the old root's PDF never shows", async () => {
    api.folderStamp.mockResolvedValue("s1")
    api.buildLatex.mockResolvedValueOnce(built("/pdf/paper.pdf"))
    const { result, rerender } = renderHook(({ root }) => useLatexBuild(root, options()), { initialProps: { root: OLD as string | null } })
    await act(() => result.current.trigger())
    expect(result.current.pdfPath).toBe("/pdf/paper.pdf")

    rerender({ root: NEW })
    expect(result.current.pdfPath).toBeNull()
    expect(result.current.pdfUrl).toBeNull()
    expect(result.current.result).toBeNull()

    api.buildLatex.mockResolvedValueOnce(built("/pdf/thesis.pdf"))
    await act(() => result.current.trigger())
    expect(api.buildLatex).toHaveBeenLastCalledWith(NEW)
    await waitFor(() => expect(result.current.pdfPath).toBe("/pdf/thesis.pdf"))
  })

  it("drops the answer of a build of the old root that finishes after the root changed", async () => {
    api.folderStamp.mockResolvedValue("s1")
    const slow = Promise.withResolvers<LatexBuild>()
    api.buildLatex.mockReturnValueOnce(slow.promise)
    const { result, rerender } = renderHook(({ root }) => useLatexBuild(root, options()), { initialProps: { root: OLD } })
    let finished: Promise<void> = Promise.resolve()
    act(() => { finished = result.current.trigger() })
    rerender({ root: NEW })
    await act(async () => {
      slow.resolve(built("/pdf/paper.pdf"))
      await finished
    })
    expect(result.current.pdfPath).toBeNull()
    expect(result.current.building).toBe(false)
  })

  it("rebuilds the root when the folder's stamp changes and leaves it alone while the stamp holds", async () => {
    api.folderStamp.mockResolvedValue("s1")
    api.buildLatex.mockResolvedValue(built("/pdf/paper.pdf"))
    const { result } = renderHook(() => useLatexBuild(OLD, options()))
    await act(() => result.current.trigger())
    await tick()
    expect(api.buildLatex).toHaveBeenCalledTimes(1)

    api.folderStamp.mockResolvedValue("s2")
    await tick()
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(2))
    expect(api.buildLatex).toHaveBeenLastCalledWith(OLD)

    await tick()
    expect(api.buildLatex).toHaveBeenCalledTimes(2)
  })

  it("does not build twice for a change a build already saw", async () => {
    api.folderStamp.mockResolvedValue("s1")
    api.buildLatex.mockResolvedValue(built("/pdf/paper.pdf"))
    const { result } = renderHook(() => useLatexBuild(OLD, options()))
    await act(() => result.current.trigger())

    // A save changes the folder and triggers its own build, the next look at the folder finds nothing new
    api.folderStamp.mockResolvedValue("s2")
    await act(() => result.current.trigger())
    await tick()
    expect(api.buildLatex).toHaveBeenCalledTimes(2)
  })

  it("keeps watching when the stamp cannot be read", async () => {
    api.folderStamp.mockResolvedValue("s1")
    api.buildLatex.mockResolvedValue(built("/pdf/paper.pdf"))
    const { result } = renderHook(() => useLatexBuild(OLD, options()))
    await act(() => result.current.trigger())

    api.folderStamp.mockRejectedValueOnce(new Error("offline"))
    await tick()
    api.folderStamp.mockResolvedValue("s2")
    await tick()
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(2))
  })
})
