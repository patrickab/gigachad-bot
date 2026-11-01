import type { ComponentProps } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { WorkbenchProps } from "@/features/workbench/Workbench"
import type { LatexBuild, TexProject } from "@/lib/domain/latex"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  buildLatex: vi.fn<(path: string) => Promise<LatexBuild>>(),
  folderStamp: vi.fn<(folder: string) => Promise<string>>(async () => "stamp"),
  loadTexProject: vi.fn<(path: string) => Promise<TexProject>>(),
  chooseTexRoot: vi.fn<(path: string, root: string) => Promise<TexProject>>(),
}))
vi.mock("@/lib/backend/api", () => api)
vi.mock("@/hooks/usePolling", () => ({ usePolling: () => {} }))
const workbench = vi.hoisted(() => ({ props: null as WorkbenchProps | null }))
vi.mock("@/features/workbench/Workbench", () => ({
  Workbench: (props: WorkbenchProps) => {
    workbench.props = props
    return <div data-testid="workbench">{props.renderPane({ layout: { width: 600, top: 56, bottom: 12, height: 500 }, text: null, loadError: null })}</div>
  },
}))
// The PDF viewer is out of scope, the pane shows what it would say instead of a PDF
vi.mock("../LatexPdfPane", () => ({ LatexPdfPane: ({ emptyMessage }: { emptyMessage: string | null }) => emptyMessage ?? "pdf" }))

import { LatexWorkbench } from "../LatexWorkbench"

type WorkbenchOwnProps = ComponentProps<typeof LatexWorkbench>

const FOLDER = "/home/me/thesis"
const ROOT = `${FOLDER}/main.tex`
const CHAPTER = `${FOLDER}/chapters/intro.tex`
const built = { ok: true, pdf_path: `${FOLDER}/.latex-build/main.pdf`, duration_ms: 100, errors: [], log_tail: "" } as unknown as LatexBuild

/** The workbench of the file at `path`, the parts of its props the tests do not read left out. */
const workbenchOf = (path: string, texFolder?: string, onTexFolder: (folder: string) => void = vi.fn()) =>
  <LatexWorkbench {...{ path, texFolder, onTexFolder } as unknown as WorkbenchOwnProps} />

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  workbench.props = null
})

describe("LatexWorkbench", () => {
  it("builds the project's root, not the open file, and again on an agent edit, after a save of any open file and when a run ends", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(1))
    expect(api.buildLatex).toHaveBeenLastCalledWith(ROOT)

    act(() => workbench.props?.onPulled?.("\\section{Agent edit}", CHAPTER))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(2))

    act(() => workbench.props?.onSaved?.("\\bibliography{refs}", `${FOLDER}/refs.bib`))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(3))

    act(() => workbench.props?.onRunEnd?.())
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(4))
    expect(api.buildLatex).toHaveBeenLastCalledWith(ROOT)
  })

  it("asks again for the project of another file without remounting the workbench", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    const view = render(workbenchOf(ROOT))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(1))
    const mounted = screen.getByTestId("workbench")

    view.rerender(workbenchOf(CHAPTER, FOLDER))
    await waitFor(() => expect(api.loadTexProject).toHaveBeenLastCalledWith(CHAPTER))
    expect(screen.getByTestId("workbench")).toBe(mounted)
  })

  it("tells the item the project folder", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    const unknown = vi.fn()
    render(workbenchOf(ROOT, undefined, unknown))
    await waitFor(() => expect(unknown).toHaveBeenCalledExactlyOnceWith(FOLDER))
  })

  it("lets the user choose the root of a file without one, writes the choice and builds it", async () => {
    const orphan = `${FOLDER}/notes.tex`
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: null, candidates: [ROOT, `${FOLDER}/book.tex`] })
    api.chooseTexRoot.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(orphan))

    fireEvent.click(await screen.findByRole("button", { name: "main.tex" }))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledExactlyOnceWith(ROOT))
    expect(api.chooseTexRoot).toHaveBeenCalledExactlyOnceWith(orphan, ROOT)
    expect(screen.queryByRole("button", { name: "main.tex" })).not.toBeInTheDocument()
  })

  it("keeps the picker and says why when the choice could not be written", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: null, candidates: [ROOT] })
    api.chooseTexRoot.mockRejectedValue(new Error("Not a root"))
    render(workbenchOf(`${FOLDER}/notes.tex`))

    fireEvent.click(await screen.findByRole("button", { name: "main.tex" }))
    expect(await screen.findByRole("alert")).toHaveTextContent("Not a root")
    await waitFor(() => expect(screen.getByRole("button", { name: "main.tex" })).toBeEnabled())
    expect(api.buildLatex).not.toHaveBeenCalled()
  })

  it.each([
    { candidates: [ROOT], expected: () => screen.findByRole("button", { name: "main.tex" }) },
    { candidates: [], expected: () => screen.findByText("No root document") },
  ])("offers the candidates of a file without root, or says there is none ($candidates)", async ({ candidates, expected }) => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: null, candidates })
    render(workbenchOf(`${FOLDER}/refs.bib`))
    expect(await expected()).toBeInTheDocument()
    expect(api.buildLatex).not.toHaveBeenCalled()
  })

  it("offers a retry when the project cannot be read", async () => {
    api.loadTexProject.mockRejectedValueOnce(new Error("Outside the roots"))
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(ROOT))

    fireEvent.click(await screen.findByRole("button", { name: "Retry" }))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledExactlyOnceWith(ROOT))
    expect(screen.getByTestId("workbench")).toBeInTheDocument()
  })
})
