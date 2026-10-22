import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, waitFor, within } from "@testing-library/react"
import { DiffViewer } from "@/features/editor/diff/DiffViewer"

const api = vi.hoisted(() => ({
  fetchRootDiffFiles: vi.fn<typeof import("@/lib/backend/api").fetchRootDiffFiles>(),
  fetchRootFileDiff: vi.fn<typeof import("@/lib/backend/api").fetchRootFileDiff>(),
  fetchRootDiffContent: vi.fn<typeof import("@/lib/backend/api").fetchRootDiffContent>(),
  writeRootFile: vi.fn(),
}))
vi.mock("@/lib/backend/api", () => api)
const neovim = vi.hoisted(() => ({ enabled: false }))
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => neovim }))
vi.mock("@/features/editor/NeovimEditor", () => ({
  NeovimEditor: () => <div>neovim</div>,
  useNeovimTerminal: () => ({ terminal: null, status: "ready", restart: () => {} }),
}))

const PATCH = ["diff --git a/a.txt b/a.txt", "--- a/a.txt", "+++ b/a.txt", "@@ -1,2 +1,2 @@", " keep", "-old", "+new"].join("\n")

function selectCode(start: HTMLElement, end = start, startOffset = 0, endOffset = end.textContent!.length) {
  const range = document.createRange()
  range.setStart(start.firstChild!, startOffset)
  range.setEnd(end.firstChild!, endOffset)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
  fireEvent(document, new Event("selectionchange"))
  fireEvent.pointerUp(start.closest('[aria-label="Diff"]')!)
}

async function openComment(view: ReturnType<typeof render>, text = "new") {
  selectCode(await within(view.getByLabelText("Diff")).findByText(text))
  fireEvent.click(view.getByRole("button", { name: "Add comment to selection" }))
}

beforeEach(() => {
  neovim.enabled = false
  Object.defineProperty(Range.prototype, "getBoundingClientRect", { configurable: true, value: () => ({ right: 100, bottom: 100 }) })
  api.fetchRootDiffFiles.mockReset().mockResolvedValue({ is_git_repository: true, files: [{ path: "a.txt", status: "modified", stamp: "1", added: 1, removed: 1 }, { path: "dir/b.png", status: "added", stamp: "1", added: null, removed: null }] })
  api.fetchRootFileDiff.mockReset().mockImplementation(async (_root, path) => path === "a.txt" ? { supported: true, status: "modified", patch: PATCH } : { supported: false })
  api.fetchRootDiffContent.mockReset().mockReturnValue(new Promise(() => {}))
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => { stored.set(key, value) },
    removeItem: (key: string) => { stored.delete(key) },
  })
})
afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
  delete (Range.prototype as Partial<Range>).getBoundingClientRect
  vi.unstubAllGlobals()
})

describe("DiffViewer", () => {
  it("renders every file together, including files without a text diff", async () => {
    const view = render(<DiffViewer root="/repo" />)
    const diff = within(view.getByLabelText("Diff"))
    expect(await diff.findByText("old")).toBeVisible()
    expect(diff.getByText("new")).toBeVisible()
    expect(await diff.findByText("No text diff for this file")).toBeVisible()
    expect(diff.getByText("old")).toBeVisible()
    expect(view.getByLabelText("Diff for a.txt")).toBeVisible()
    expect(view.getByLabelText("Diff for dir/b.png")).toBeVisible()
    expect(view.getByLabelText("Change summary")).toHaveTextContent("2 files+1−1")
    expect(view.getByLabelText("Line changes in a.txt")).toHaveTextContent("+1−1")
    expect(view.queryByLabelText("Line changes in dir/b.png")).toBeNull()
    expect(within(view.getByLabelText("Changed files")).getByRole("button", { name: /^a\.txt/ })).toHaveTextContent("a.txtM+1−1")
    expect(within(view.getByLabelText("Changed files")).getByRole("button", { name: /^b\.png/ })).toHaveTextContent(/^b\.pngA$/)
  })

  it("jumps within the shared scroll container and tracks the file reached by scrolling", async () => {
    const view = render(<DiffViewer root="/repo" />)
    await view.findByText("old")
    const container = view.getByLabelText("Diff")
    const first = view.getByLabelText("Diff for a.txt")
    const second = view.getByLabelText("Diff for dir/b.png")
    vi.spyOn(container, "getBoundingClientRect").mockImplementation(() => ({ top: 40 }) as DOMRect)
    vi.spyOn(first, "getBoundingClientRect").mockImplementation(() => ({ top: 40 - container.scrollTop }) as DOMRect)
    vi.spyOn(second, "getBoundingClientRect").mockImplementation(() => ({ top: 640 - container.scrollTop }) as DOMRect)

    fireEvent.click(view.getByRole("button", { name: /^b\.png/ }))
    expect(container.scrollTop).toBe(600)
    expect(view.getByText("old")).toBeVisible()
    container.scrollTop = 0
    fireEvent.scroll(container)
    await waitFor(() => expect(view.getByRole("button", { name: /^a\.txt/ })).toHaveAttribute("aria-current", "page"))
  })

  it("edits the file whose header button was clicked, deleted files have none", async () => {
    neovim.enabled = true
    api.fetchRootDiffFiles.mockResolvedValue({ is_git_repository: true, files: [{ path: "a.txt", status: "modified", stamp: "1", added: null, removed: null }, { path: "dir/b.png", status: "added", stamp: "1", added: null, removed: null }, { path: "gone.txt", status: "deleted", stamp: "1", added: null, removed: null }] })
    const view = render(<DiffViewer root="/repo" />)
    const second = within(await view.findByLabelText("Diff for dir/b.png"))
    expect(within(view.getByLabelText("Diff for gone.txt")).queryByRole("button")).toBeNull()
    fireEvent.click(second.getByRole("button", { name: "Edit diff" }))
    expect(api.fetchRootDiffContent).toHaveBeenCalledWith("/repo", "dir/b.png")
    expect(second.getByRole("button", { name: "Opening…" })).toBeDisabled()
    expect(within(view.getByLabelText("Diff for a.txt")).getByRole("button", { name: "Edit diff" })).toBeDisabled()
  })

  it("merges folder chains without files into one sidebar row, branching folders stay separate", async () => {
    api.fetchRootDiffFiles.mockResolvedValue({ is_git_repository: true, files: [{ path: "src/ui/tree/a.ts", status: "modified", stamp: "1", added: null, removed: null }, { path: "src/ui/tree/b.ts", status: "modified", stamp: "1", added: null, removed: null }, { path: "src/lib/c.ts", status: "added", stamp: "1", added: null, removed: null }] })
    const sidebar = within(render(<DiffViewer root="/repo" />).getByLabelText("Changed files"))
    expect(await sidebar.findByRole("button", { name: "ui/tree" })).toHaveAttribute("aria-expanded", "true")
    expect(sidebar.getByRole("button", { name: "src" })).toBeVisible()
    expect(sidebar.getByRole("button", { name: /^a\.ts/ })).toBeVisible()
  })

  it("explains a mount that is not a git repository", async () => {
    api.fetchRootDiffFiles.mockResolvedValue({ is_git_repository: false, files: [] })
    const view = render(<DiffViewer root="/plain" />)
    expect(await view.findByText("This file mount is not a git repository")).toBeVisible()
  })

  it("reports a clean working tree", async () => {
    api.fetchRootDiffFiles.mockResolvedValue({ is_git_repository: true, files: [] })
    const view = render(<DiffViewer root="/clean" />)
    expect(await view.findByText("No changes")).toBeVisible()
  })

  it("refetches only the file whose stamp changed and updates its counts", async () => {
    const view = render(<DiffViewer root="/repo" />)
    await view.findByText("old")
    expect(api.fetchRootFileDiff).toHaveBeenCalledTimes(2)
    const edited = ["diff --git a/a.txt b/a.txt", "--- a/a.txt", "+++ b/a.txt", "@@ -1,1 +1,3 @@", " keep", "+x", "+y"].join("\n")
    api.fetchRootFileDiff.mockImplementation(async (_root, path) => path === "a.txt" ? { supported: true, status: "modified", patch: edited } : { supported: false })
    api.fetchRootDiffFiles.mockResolvedValue({ is_git_repository: true, files: [{ path: "a.txt", status: "modified", stamp: "2", added: 2, removed: 0 }, { path: "dir/b.png", status: "added", stamp: "1", added: null, removed: null }] })

    expect(await view.findByText("y", {}, { timeout: 4000 })).toBeVisible()
    expect(view.getByLabelText("Line changes in a.txt")).toHaveTextContent("+2−0")
    expect(view.queryByText("old")).toBeNull()
    expect(api.fetchRootFileDiff).toHaveBeenCalledTimes(3)
  })

  it("collects line notes and sends them as one review, keeping them when sending fails", async () => {
    const send = vi.fn<(text: string) => Promise<void>>().mockRejectedValueOnce(new Error("Agent is busy")).mockResolvedValue()
    const view = render(<DiffViewer root="/repo" review={{ send, canSend: true }} />)
    await openComment(view)
    fireEvent.change(view.getByRole("textbox", { name: "Note on a.txt line 2" }), { target: { value: "say why" } })
    fireEvent.keyDown(view.getByRole("textbox"), { key: "Enter" })
    expect(view.getByText("say why")).toBeVisible()
    expect(within(view.getByLabelText("Changed files")).getByLabelText("1 notes")).toBeVisible()

    fireEvent.click(view.getByRole("button", { name: "Send review" }))
    expect(await view.findByRole("alert")).toHaveTextContent("Agent is busy")
    expect(view.getByText("say why")).toBeVisible()

    fireEvent.click(view.getByRole("button", { name: "Send review" }))
    await waitFor(() => expect(view.queryByText("say why")).toBeNull())
    expect(send.mock.lastCall![0]).toContain("`a.txt:2`")
    expect(send.mock.lastCall![0]).toContain("Selected code:\n```\nnew\n```")
    expect(send.mock.lastCall![0]).toContain("say why")
    expect(view.queryByRole("button", { name: "Send review" })).toBeNull()
  })

  it("keeps unsent notes across remounts and shows a note whose line changed above the diff", async () => {
    const first = render(<DiffViewer root="/repo" review={{ send: vi.fn(), canSend: false }} />)
    await openComment(first)
    fireEvent.change(first.getByRole("textbox"), { target: { value: "tighten" } })
    fireEvent.keyDown(first.getByRole("textbox"), { key: "Enter" })
    expect(first.getByRole("button", { name: "Send review" })).toBeDisabled()
    first.unmount()

    api.fetchRootFileDiff.mockResolvedValue({ supported: true, status: "modified", patch: PATCH.replace("+new", "+renewed") })
    const second = render(<DiffViewer root="/repo" review={{ send: vi.fn(), canSend: true }} />)
    expect(await second.findByText("Line 2 changed since this note")).toBeVisible()
    expect(second.getByText("tighten")).toBeVisible()
  })

  it("sends refreshed line numbers rather than the stored anchor number", async () => {
    const send = vi.fn<(text: string) => Promise<void>>().mockResolvedValue()
    const view = render(<DiffViewer root="/repo" review={{ send, canSend: true }} />)
    await openComment(view)
    fireEvent.change(view.getByRole("textbox"), { target: { value: "explain this" } })
    fireEvent.keyDown(view.getByRole("textbox"), { key: "Enter" })

    api.fetchRootDiffFiles.mockResolvedValue({ is_git_repository: true, files: [{ path: "a.txt", status: "modified", stamp: "2", added: 2, removed: 1 }] })
    api.fetchRootFileDiff.mockResolvedValue({ supported: true, status: "modified", patch: "@@ -1,2 +1,3 @@\n+inserted\n keep\n-old\n+new" })
    fireEvent.focus(window)
    await within(view.getByLabelText("Diff")).findByText("inserted")
    fireEvent.click(view.getByRole("button", { name: "Send review" }))
    await waitFor(() => expect(send).toHaveBeenCalledWith(expect.stringContaining("`a.txt:3`")))
    expect(send.mock.calls[0]![0]).not.toContain("outdated")
  })

  it("comments on selected blocks in ordinary diffs and copies them without clearing drafts", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } })
    api.fetchRootFileDiff.mockImplementation(async (_root, path) => path === "a.txt" ? { supported: true, status: "modified", patch: PATCH.replace(" keep", " keep\n ") } : { supported: false })
    const view = render(<DiffViewer root="/repo" />)
    const diff = within(view.getByLabelText("Diff"))
    const start = await diff.findByText("keep")
    const end = diff.getByText("new")
    selectCode(start, end, 1, 2)
    fireEvent.click(view.getByRole("button", { name: "Add comment to selection" }))
    const editor = view.getByRole("textbox", { name: "Note on a.txt line 3" })
    expect(editor).toHaveFocus()
    expect(end.compareDocumentPosition(editor) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    fireEvent.change(view.getByRole("textbox"), { target: { value: "simplify this block" } })
    fireEvent.keyDown(view.getByRole("textbox"), { key: "Enter" })
    fireEvent.click(view.getByRole("button", { name: "Copy review" }))
    await view.findByRole("status")
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining("Selected code:\n```\neep\n\nold\nne\n```"))
    expect(writeText).toHaveBeenCalledWith(expect.stringContaining("simplify this block"))
    expect(view.getByText("simplify this block")).toBeVisible()
  })

  it("anchors to the last selected character, not an unselected next line", async () => {
    const view = render(<DiffViewer root="/repo" />)
    const diff = within(view.getByLabelText("Diff"))
    const start = await diff.findByText("keep")
    selectCode(start, diff.getByText("new"), 0, 0)
    fireEvent.click(view.getByRole("button", { name: "Add comment to selection" }))
    expect(view.getByRole("textbox", { name: "Note on a.txt old line 2" })).toHaveFocus()
  })

  it("keeps split-view selections on one side and dismisses the action when selection clears", async () => {
    const view = render(<DiffViewer root="/repo" />)
    await view.findByText("new")
    fireEvent.click(view.getByRole("button", { name: "split" }))
    const diff = within(view.getByLabelText("Diff"))
    selectCode(diff.getByText("old"), diff.getByText("new"))
    expect(view.queryByRole("button", { name: "Add comment to selection" })).toBeNull()
    selectCode(diff.getAllByText("keep")[1]!, diff.getByText("new"))
    expect(view.getByRole("button", { name: "Add comment to selection" })).toBeVisible()
    window.getSelection()?.removeAllRanges()
    fireEvent(document, new Event("selectionchange"))
    expect(view.queryByRole("button", { name: "Add comment to selection" })).toBeNull()
  })
})
