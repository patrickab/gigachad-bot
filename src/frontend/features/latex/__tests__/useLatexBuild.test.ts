import { afterEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { LatexBuild } from "@/lib/domain/latex"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  buildLatex: vi.fn<(path: string) => Promise<LatexBuild>>(),
}))
vi.mock("@/lib/backend/api", () => api)

import { useLatexBuild } from "../useLatexBuild"

const OLD = "/data/workbench/latex/proj/paper/paper.tex"
const NEW = "/data/workbench/latex/proj/paper/thesis.tex"
const built = (pdf: string) => ({ ok: true, pdf_path: pdf, duration_ms: 1, errors: [], log_tail: "" }) as unknown as LatexBuild

afterEach(() => vi.clearAllMocks())

describe("useLatexBuild", () => {
  it("starts clean for a renamed file so the old file's PDF never shows", async () => {
    api.buildLatex.mockResolvedValueOnce(built("/pdf/paper.pdf"))
    const { result, rerender } = renderHook(({ path }) => useLatexBuild(path), { initialProps: { path: OLD } })
    await act(() => result.current.trigger())
    expect(result.current.pdfPath).toBe("/pdf/paper.pdf")

    rerender({ path: NEW })
    expect(result.current.pdfPath).toBeNull()
    expect(result.current.pdfUrl).toBeNull()
    expect(result.current.result).toBeNull()

    api.buildLatex.mockResolvedValueOnce(built("/pdf/thesis.pdf"))
    await act(() => result.current.trigger())
    expect(api.buildLatex).toHaveBeenLastCalledWith(NEW)
    await waitFor(() => expect(result.current.pdfPath).toBe("/pdf/thesis.pdf"))
  })

  it("drops the answer of a build of the old name that finishes after the rename", async () => {
    const slow = Promise.withResolvers<LatexBuild>()
    api.buildLatex.mockReturnValueOnce(slow.promise)
    const { result, rerender } = renderHook(({ path }) => useLatexBuild(path), { initialProps: { path: OLD } })
    let finished: Promise<void> = Promise.resolve()
    act(() => { finished = result.current.trigger() })
    rerender({ path: NEW })
    await act(async () => {
      slow.resolve(built("/pdf/paper.pdf"))
      await finished
    })
    expect(result.current.pdfPath).toBeNull()
    expect(result.current.building).toBe(false)
  })
})
