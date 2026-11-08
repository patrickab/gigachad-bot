import type { ComponentProps } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import type { WorkbenchEditorActions, WorkbenchProps } from "@/features/workbench/Workbench"
import type { PdfReveal } from "@/components/content/PdfViewer"
import type { PdfWord } from "@/components/content/pdf/pdfText"
import type { LatexBuild, LatexPrimitive, LatexTemplates, SynctexClickedSource, SynctexSource, SynctexView, TexProject } from "@/lib/domain/latex"

const api = vi.hoisted(() => ({
  ApiError: class ApiError extends Error { status = 500 },
  buildLatex: vi.fn<(path: string) => Promise<LatexBuild>>(),
  loadTexProject: vi.fn<(path: string) => Promise<TexProject>>(),
  chooseTexRoot: vi.fn<(path: string, root: string) => Promise<TexProject>>(),
  loadLatexTemplates: vi.fn<(root: string) => Promise<LatexTemplates>>(async () => ({ theme: null, primitives: [] })),
  findSynctexSource: vi.fn<(root: string, page: number, x: number, y: number, word?: string, offset?: number, occurrence?: number) => Promise<SynctexClickedSource>>(),
  findSynctexView: vi.fn<(root: string, file: string, line: number) => Promise<SynctexView>>(),
  findSlideEnd: vi.fn<(root: string, page: number) => Promise<SynctexSource>>(),
}))
vi.mock("@/lib/backend/api", () => api)
const follow = vi.hoisted(() => ({ folders: [] as readonly string[], changed: null as ((path: string) => void) | null }))
vi.mock("@/hooks/useFollowFiles", () => ({
  useFollowFiles: (_files: readonly string[], onChange: (path: string) => void, folders: readonly string[] = []) => {
    follow.folders = folders
    follow.changed = onChange
  },
}))
const workbench = vi.hoisted(() => ({
  props: null as WorkbenchProps | null,
  send: vi.fn<WorkbenchEditorActions["send"]>(() => true),
  reveal: vi.fn<WorkbenchEditorActions["reveal"]>(),
  focus: vi.fn(),
  neovim: true,
}))
// The workbench shows its pane, its status with the notice and, with Neovim, its `center` controls
vi.mock("@/features/workbench/Workbench", () => ({
  Workbench: (props: WorkbenchProps) => {
    workbench.props = props
    const editor: WorkbenchEditorActions | null = workbench.neovim ? { reveal: workbench.reveal, send: workbench.send, focus: workbench.focus } : null
    return <div data-testid="workbench">
      {props.renderPane({ layout: { width: 600, top: 56, bottom: 12, height: 500 }, text: null, loadError: null, editor })}
      {props.status}
      {props.notices?.notice && <span role="alert">{props.notices.notice}</span>}
      {editor && props.center?.(editor)}
    </div>
  },
}))
// The PDF viewer is out of scope, the pane shows what it would say instead of a PDF, hands out its double-click and slide gap and notes the spot it reveals
type PageDoubleClick = (page: number, x: number, y: number, word: PdfWord) => void
type SlideGap = { marked: number | null; onChoose: (page: number) => void; onClear: () => void }
const NO_WORD: PdfWord = { text: "", offset: 0, occurrence: 0 }
const pdf = vi.hoisted(() => ({ doubleClick: null as PageDoubleClick | null, reveal: undefined as PdfReveal | undefined, slideGap: undefined as SlideGap | undefined }))
vi.mock("../LatexPdfPane", () => ({
  LatexPdfPane: ({ emptyMessage, onPageDoubleClick, reveal, slideGap }: { emptyMessage: string | null; onPageDoubleClick?: PageDoubleClick; reveal?: PdfReveal; slideGap?: SlideGap }) => {
    pdf.doubleClick = onPageDoubleClick ?? null
    pdf.reveal = reveal
    pdf.slideGap = slideGap
    return emptyMessage ?? "pdf"
  },
}))

import { LatexWorkbench } from "../LatexWorkbench"

type WorkbenchOwnProps = ComponentProps<typeof LatexWorkbench>

const FOLDER = "/home/me/thesis"
const ROOT = `${FOLDER}/main.tex`
const CHAPTER = `${FOLDER}/chapters/intro.tex`
const built = { ok: true, pdf_path: `${FOLDER}/.latex-build/main.pdf`, duration_ms: 100, errors: [], log_tail: "" } as unknown as LatexBuild

/** The workbench of the file at `path`, the parts of its props the tests do not read left out. */
const workbenchOf = (path: string, texFolder?: string, onTexFolder: (folder: string) => void = vi.fn()) =>
  <LatexWorkbench {...{ path, texFolder, onTexFolder } as unknown as WorkbenchOwnProps} />

const primitive = (group: string, groupLabel: string, name: string, label: string, preview: number | null = 7, summary = `${label} slide`): LatexPrimitive =>
  ({ id: `${group}/${name}`, group, group_label: groupLabel, label, description: `${summary}, in full`, summary, text: `\\begin{frame}{${label}}\n\\end{frame}\n`, preview })
const PRESENTATION: LatexTemplates = {
  theme: "presentation",
  primitives: [
    primitive("text", "Text", "title", "Title"),
    primitive("text", "Text", "bullets", "Bullets", 7, "Cause and effect points"),
    primitive("numbers", "Key figures", "big-number", "Big number", null, "One big number carries the slide"),
  ],
}

/** What `send` gets for inserting the slide `text`. */
const insertOf = (text: string) => [{ type: "insert_frame", text }, "Neovim is not ready, the slide was not added"] as const

/** The workbench of a presentation root with its "Add slide" list open. */
async function openSlideList() {
  api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
  api.buildLatex.mockResolvedValue(built)
  api.loadLatexTemplates.mockResolvedValueOnce(PRESENTATION)
  render(workbenchOf(ROOT))
  const trigger = await screen.findByRole("button", { name: "Add slide" })
  fireEvent.click(trigger)
  return trigger
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  workbench.props = null
  pdf.doubleClick = null
  pdf.reveal = undefined
  pdf.slideGap = undefined
  workbench.neovim = true
})

describe("LatexWorkbench", () => {
  it("builds the project's root, not the open file, and again for every change in the project folder", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(1))
    expect(api.buildLatex).toHaveBeenLastCalledWith(ROOT)
    expect(follow.folders).toEqual([FOLDER])

    act(() => follow.changed?.(`${FOLDER}/refs.bib`))
    await waitFor(() => expect(api.buildLatex).toHaveBeenCalledTimes(2))
    expect(api.buildLatex).toHaveBeenLastCalledWith(ROOT)
  })

  it("reads the project again after a save of the shown file, a save of another file leaves it", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))
    await waitFor(() => expect(api.loadTexProject).toHaveBeenCalledTimes(1))

    act(() => workbench.props?.onSaved?.("\\bibliography{refs}", `${FOLDER}/refs.bib`))
    expect(api.loadTexProject).toHaveBeenCalledTimes(1)
    act(() => workbench.props?.onSaved?.("% !TEX root = ../main.tex", CHAPTER))
    await waitFor(() => expect(api.loadTexProject).toHaveBeenCalledTimes(2))
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

  it("offers no slides for a root that is not a presentation", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(ROOT))
    await waitFor(() => expect(api.loadLatexTemplates).toHaveBeenCalledExactlyOnceWith(ROOT))
    expect(await screen.findByTestId("workbench")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Add slide" })).not.toBeInTheDocument()
  })

  it("lists the slides of a presentation under their group labels in the order the template set gives, each with its summary", async () => {
    await openSlideList()

    expect(screen.getAllByRole("group")).toEqual([screen.getByRole("group", { name: "Text" }), screen.getByRole("group", { name: "Key figures" })])
    expect(within(screen.getByRole("group", { name: "Text" })).getAllByRole("option").map((item) => item.textContent)).toEqual(["TitleTitle slide", "BulletsCause and effect points"])
    expect(within(screen.getByRole("group", { name: "Key figures" })).getAllByRole("option").map((item) => item.textContent)).toEqual(["Big numberOne big number carries the slide"])
  })

  it("opens on the first slide with the search field focused, moves across groups with the arrows and inserts with Enter", async () => {
    const trigger = await openSlideList()
    const search = screen.getByRole("combobox", { name: "Search slides…" })
    const highlighted = () => screen.getByRole("option", { selected: true }).textContent

    expect(search).toHaveFocus()
    expect(highlighted()).toMatch(/^Title/)
    for (const key of ["ArrowDown", "ArrowDown"]) fireEvent.keyDown(search, { key })
    expect(highlighted()).toMatch(/^Big number/)
    fireEvent.keyDown(search, { key: "ArrowUp" })
    expect(highlighted()).toMatch(/^Bullets/)
    expect(search).toHaveFocus()
    fireEvent.keyDown(search, { key: "Enter" })

    expect(workbench.send).toHaveBeenCalledExactlyOnceWith(...insertOf(PRESENTATION.primitives[1].text))
    expect(workbench.focus).toHaveBeenCalledOnce()
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Slide primitives" })).not.toBeInTheDocument())
    expect(trigger).toHaveFocus()
  })

  it("filters the slides by name, group and summary and says so when none match", async () => {
    await openSlideList()
    const search = screen.getByRole("combobox", { name: "Search slides…" })

    fireEvent.change(search, { target: { value: "carries" } })
    expect(screen.getAllByRole("option").map((item) => item.textContent)).toEqual(["Big numberOne big number carries the slide"])
    fireEvent.change(search, { target: { value: "text" } })
    expect(screen.getAllByRole("option").map((item) => item.textContent)).toEqual(["TitleTitle slide", "BulletsCause and effect points"])
    fireEvent.keyDown(search, { key: "ArrowDown" })
    expect(screen.getByRole("option", { selected: true })).toHaveTextContent(/^Bullets/)
    fireEvent.change(search, { target: { value: "te" } })
    expect(screen.getByRole("option", { selected: true })).toHaveTextContent(/^Title/)
    fireEvent.change(search, { target: { value: "zzz" } })
    expect(screen.queryByRole("option")).not.toBeInTheDocument()
    expect(screen.getByText("No match")).toBeInTheDocument()
    fireEvent.keyDown(search, { key: "Enter" })
    expect(workbench.send).not.toHaveBeenCalled()
    fireEvent.change(search, { target: { value: "big" } })
    fireEvent.keyDown(search, { key: "Enter" })

    expect(workbench.send).toHaveBeenCalledExactlyOnceWith(...insertOf(PRESENTATION.primitives[2].text))
  })

  it("keeps the list open when Neovim cannot take the frame", async () => {
    await openSlideList()
    workbench.send.mockReturnValueOnce(false)

    fireEvent.click(screen.getByRole("option", { name: /^Title/ }))

    expect(workbench.send).toHaveBeenCalledExactlyOnceWith(...insertOf(PRESENTATION.primitives[0].text))
    expect(workbench.focus).not.toHaveBeenCalled()
    expect(screen.getByRole("dialog", { name: "Slide primitives" })).toBeInTheDocument()
  })

  it("closes the list with Escape without inserting, focus back on the button", async () => {
    const trigger = await openSlideList()

    fireEvent.keyDown(screen.getByRole("combobox", { name: "Search slides…" }), { key: "Escape" })

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Slide primitives" })).not.toBeInTheDocument())
    expect(workbench.send).not.toHaveBeenCalled()
    expect(trigger).toHaveFocus()
  })

  it("previews the highlighted slide in a card and shows none for one without a preview or a preview that fails", async () => {
    await openSlideList()
    const preview = () => document.body.querySelector<HTMLImageElement>("[aria-hidden].fixed img")

    expect(preview()?.getAttribute("src")).toMatch(/\/latex\/templates\/presentation\/previews\/text\/title\.png\?v=7$/)
    fireEvent.mouseMove(screen.getByRole("option", { name: /^Big number/ }), { clientX: 10, clientY: 10 })
    expect(preview()).toBeNull()
    fireEvent.mouseMove(screen.getByRole("option", { name: /^Bullets/ }), { clientX: 10, clientY: 20 })
    fireEvent.error(preview()!)
    expect(preview()).toBeNull()
  })
})

describe("LatexWorkbench inverse search", () => {
  /** The workbench of `CHAPTER` with its PDF on screen. */
  async function showPdf() {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))
    await waitFor(() => expect(pdf.doubleClick).not.toBeNull())
  }

  it.each([
    { file: "the shown file", source: CHAPTER },
    { file: "another project file", source: ROOT },
  ])("reveals the source line and column SyncTeX finds for a double-clicked spot and word in $file", async ({ source }) => {
    api.findSynctexSource.mockResolvedValue({ path: source, line: 42, column: 9 })
    await showPdf()

    act(() => pdf.doubleClick?.(2, 72.5, 144, { text: "the", offset: 1, occurrence: 2 }))

    await waitFor(() => expect(workbench.reveal).toHaveBeenCalledExactlyOnceWith(source, 42, 9, { focus: true }))
    expect(api.findSynctexSource).toHaveBeenCalledExactlyOnceWith(ROOT, 2, 72.5, 144, "the", 1, 2)
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("says so when SyncTeX knows no source line for the spot, and clears it once a spot has one", async () => {
    api.findSynctexSource.mockRejectedValueOnce(Object.assign(new api.ApiError("No match"), { status: 404 }))
    await showPdf()

    act(() => pdf.doubleClick?.(1, 10, 10, NO_WORD))

    expect(await screen.findByRole("alert")).toHaveTextContent("No source line for this spot")
    expect(workbench.reveal).not.toHaveBeenCalled()

    api.findSynctexSource.mockResolvedValueOnce({ path: ROOT, line: 3, column: 0 })
    act(() => pdf.doubleClick?.(1, 300, 400, NO_WORD))
    await waitFor(() => expect(workbench.reveal).toHaveBeenCalledExactlyOnceWith(ROOT, 3, 0, { focus: true }))
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("shows why any other failure found no source line", async () => {
    api.findSynctexSource.mockRejectedValueOnce(Object.assign(new api.ApiError("synctex is not installed"), { status: 503 }))
    await showPdf()

    act(() => pdf.doubleClick?.(1, 10, 10, NO_WORD))

    expect(await screen.findByRole("alert")).toHaveTextContent("synctex is not installed")
  })
})

describe("LatexWorkbench forward search", () => {
  /** The workbench of `CHAPTER` with its PDF on screen. */
  async function showPdf() {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))
    await waitFor(() => expect(pdf.doubleClick).not.toBeNull())
  }

  /** `<leader>v` on 1-based `line` of `CHAPTER`, resolving whether the PDF moved to its spot. */
  async function forwardSearch(line: number) {
    let moved: boolean | undefined
    await act(async () => { moved = await workbench.props?.onForwardSearch?.(CHAPTER, line) })
    return moved
  }

  it("reveals the PDF spot and the line's words SyncTeX finds for the line and answers that the PDF moved, again for the same line", async () => {
    api.findSynctexView.mockResolvedValue({ page: 3, x: 72, y: 200, words: ["Growth", "doubled"] })
    await showPdf()

    expect(await forwardSearch(12)).toBe(true)

    expect(pdf.reveal).toMatchObject({ page: 3, y: 200, words: ["Growth", "doubled"] })
    expect(api.findSynctexView).toHaveBeenCalledExactlyOnceWith(ROOT, CHAPTER, 12)
    const firstKey = pdf.reveal?.key

    expect(await forwardSearch(12)).toBe(true)
    expect(pdf.reveal?.key).not.toBe(firstKey)
    expect(pdf.reveal).toMatchObject({ page: 3, y: 200 })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it.each([404, 403])("says so when SyncTeX has no PDF spot for the line (%i) and answers that the PDF stayed", async (status) => {
    api.findSynctexView.mockRejectedValueOnce(Object.assign(new api.ApiError("No match"), { status }))
    await showPdf()

    expect(await forwardSearch(12)).toBe(false)

    expect(await screen.findByRole("alert")).toHaveTextContent("No PDF spot for this line")
    expect(pdf.reveal).toBeUndefined()
  })

  it("shows why any other failure found no PDF spot", async () => {
    api.findSynctexView.mockRejectedValueOnce(Object.assign(new api.ApiError("synctex is not installed"), { status: 503 }))
    await showPdf()

    await forwardSearch(12)

    expect(await screen.findByRole("alert")).toHaveTextContent("synctex is not installed")
  })
})

describe("LatexWorkbench PDF download", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("saves the built PDF under the root's name", async () => {
    const saved: string[] = []
    const fetchPdf = vi.fn<(url: string) => Promise<Response>>(async () => new Response(new Blob(["%PDF"])))
    vi.stubGlobal("fetch", fetchPdf)
    vi.stubGlobal("URL", class extends URL {
      static createObjectURL = () => "blob:pdf"
      static revokeObjectURL = () => {}
    })
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) { saved.push(this.download) })
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))

    const button = await screen.findByRole("button", { name: "Download PDF" })
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.click(button)

    await waitFor(() => expect(saved).toEqual(["main.pdf"]))
    expect(fetchPdf.mock.calls[0][0]).toContain(encodeURIComponent(`${FOLDER}/.latex-build/main.pdf`))
  })
})

describe("LatexWorkbench presentation", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("opens the presenter window of the built PDF from the top bar of a project with a root", async () => {
    const open = vi.spyOn(window, "open").mockReturnValue({} as Window)
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(CHAPTER))

    const button = await screen.findByRole("button", { name: "Present" })
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.click(button)

    expect(open).toHaveBeenCalledTimes(1)
    const query = new URL(String(open.mock.calls[0][0]), window.location.origin).searchParams
    expect(query.get("role")).toBe("presenter")
    expect(query.get("pdf")).toContain(encodeURIComponent(`${FOLDER}/.latex-build/main.pdf`))
  })

  it("offers no presentation without a root", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: null, candidates: [] })
    render(workbenchOf(CHAPTER))

    await screen.findByText("No root document")
    expect(screen.queryByRole("button", { name: "Present" })).toBeNull()
  })
})

describe("LatexWorkbench slide gap", () => {
  /** The workbench of a presentation root with its PDF on screen. */
  async function showPresentation() {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    api.loadLatexTemplates.mockResolvedValueOnce(PRESENTATION)
    render(workbenchOf(ROOT))
    await waitFor(() => expect(pdf.slideGap).toBeDefined())
  }

  /** Picks the "Title" slide from the "Add slide" list. */
  function addTitleSlide() {
    fireEvent.click(screen.getByRole("button", { name: "Add slide" }))
    fireEvent.click(screen.getByRole("option", { name: /^Title/ }))
  }

  it("puts Neovim's cursor after the slide above a clicked gap and marks the gap until a slide is inserted", async () => {
    api.findSlideEnd.mockResolvedValue({ path: CHAPTER, line: 24 })
    await showPresentation()

    act(() => pdf.slideGap?.onChoose(3))

    await waitFor(() => expect(pdf.slideGap?.marked).toBe(3))
    expect(api.findSlideEnd).toHaveBeenCalledExactlyOnceWith(ROOT, 3)
    expect(workbench.reveal).toHaveBeenCalledExactlyOnceWith(CHAPTER, 24)

    addTitleSlide()

    expect(workbench.send).toHaveBeenCalledExactlyOnceWith(...insertOf(PRESENTATION.primitives[0].text))
    await waitFor(() => expect(pdf.slideGap?.marked).toBeNull())
  })

  it("keeps the mark when Neovim cannot take the slide, clearing it drops the mark", async () => {
    api.findSlideEnd.mockResolvedValue({ path: ROOT, line: 20 })
    await showPresentation()
    act(() => pdf.slideGap?.onChoose(2))
    await waitFor(() => expect(pdf.slideGap?.marked).toBe(2))

    workbench.send.mockReturnValueOnce(false)
    addTitleSlide()
    expect(pdf.slideGap?.marked).toBe(2)

    act(() => pdf.slideGap?.onClear())
    expect(pdf.slideGap?.marked).toBeNull()
  })

  it("says so when the page above a gap has no slide, the earlier mark goes", async () => {
    api.findSlideEnd.mockResolvedValueOnce({ path: ROOT, line: 20 })
    api.findSlideEnd.mockRejectedValueOnce(Object.assign(new api.ApiError("No slide"), { status: 404 }))
    await showPresentation()
    act(() => pdf.slideGap?.onChoose(2))
    await waitFor(() => expect(pdf.slideGap?.marked).toBe(2))

    act(() => pdf.slideGap?.onChoose(5))

    expect(await screen.findByRole("alert")).toHaveTextContent("No slide to insert after")
    expect(pdf.slideGap?.marked).toBeNull()
    expect(workbench.reveal).toHaveBeenCalledOnce()
  })

  it("offers no gap and no double-click without Neovim", async () => {
    workbench.neovim = false
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    api.loadLatexTemplates.mockResolvedValueOnce(PRESENTATION)
    render(workbenchOf(ROOT))

    await waitFor(() => expect(api.loadLatexTemplates).toHaveBeenCalledExactlyOnceWith(ROOT))
    await act(async () => {})
    expect(pdf.slideGap).toBeUndefined()
    expect(pdf.doubleClick).toBeNull()
  })

  it("offers no gap for a root that is not a presentation", async () => {
    api.loadTexProject.mockResolvedValue({ folder: FOLDER, root: ROOT, candidates: [] })
    api.buildLatex.mockResolvedValue(built)
    render(workbenchOf(ROOT))

    await waitFor(() => expect(api.loadLatexTemplates).toHaveBeenCalledExactlyOnceWith(ROOT))
    await waitFor(() => expect(pdf.doubleClick).not.toBeNull())
    expect(pdf.slideGap).toBeUndefined()
  })
})
