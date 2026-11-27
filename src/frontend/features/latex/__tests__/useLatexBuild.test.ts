import { afterEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { LatexBuild } from "@/lib/domain/latex"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  buildLatex: vi.fn<(path: string) => Promise<LatexBuild>>(),
}))
vi.mock("@/lib/backend/api", () => api)
const follow = vi.hoisted(() => ({
  files: null as readonly string[] | null,
  folders: null as readonly string[] | null,
  changed: null as ((path: string) => void) | null,
}))
vi.mock("@/hooks/useFollowFiles", () => ({
  useFollowFiles: (files: readonly string[], onChange: (path: string) => void, folders: readonly string[] = []) => {
    follow.files = files
    follow.folders = folders
    follow.changed = onChange
  },
}))

import { useLatexBuild } from "../useLatexBuild"

const FOLDER = "/data/workbench/latex/proj/paper"
const OLD = `${FOLDER}/paper.tex`
const NEW = `${FOLDER}/thesis.tex`
const built = (pdf: string) => ({ ok: true, pdf_path: pdf, duration_ms: 1, errors: [], log_tail: "" }) as unknown as LatexBuild
const options = { folder: FOLDER }

afterEach(() => {
  vi.clearAllMocks()
  follow.files = null
  follow.folders = null
  follow.changed = null
})

describe("useLatexBuild", () => {
  it("starts clean for another root so the old root's PDF never shows", async () => {
    api.buildLatex.mockResolvedValueOnce(built("/pdf/paper.pdf"))
    const { result, rerender } = renderHook(({ root }) => useLatexBuild(root, options), { initialProps: { root: OLD as string | null } })
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
    const slow = Promise.withResolvers<LatexBuild>()
    api.buildLatex.mockReturnValueOnce(slow.promise)
    const { result, rerender } = renderHook(({ root }) => useLatexBuild(root, options), { initialProps: { root: OLD } })
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
})
