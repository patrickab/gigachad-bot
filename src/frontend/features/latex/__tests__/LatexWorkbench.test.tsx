import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, waitFor } from "@testing-library/react"
import type { WorkbenchProps } from "@/features/workbench/Workbench"
import type { LatexBuild } from "@/lib/domain/latex"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  buildLatex: vi.fn<(path: string) => Promise<LatexBuild>>(),
}))
vi.mock("@/lib/backend/api", () => api)
const workbench = vi.hoisted(() => ({ props: null as WorkbenchProps | null }))
vi.mock("@/features/workbench/Workbench", () => ({
  Workbench: (props: WorkbenchProps) => {
    workbench.props = props
    return null
  },
}))

import { LatexWorkbench } from "../LatexWorkbench"

const PATH = "/data/workbench/latex/paper/paper.tex"

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  workbench.props = null
})

describe("LatexWorkbench", () => {
  it("builds when an agent edit arrives during a run", async () => {
    const built = { ok: true, pdf_path: "paper.pdf", duration_ms: 100, errors: [], log_tail: "" } as unknown as LatexBuild
    api.buildLatex.mockResolvedValue(built)
    const props = { path: PATH } as unknown as WorkbenchProps
    render(<LatexWorkbench {...props} />)
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(1))

    act(() => workbench.props?.onPulled?.("\\section{Agent edit}"))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(2))
    expect(api.buildLatex).toHaveBeenLastCalledWith(PATH)
  })
})
