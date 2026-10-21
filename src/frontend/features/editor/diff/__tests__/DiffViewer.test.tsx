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

beforeEach(() => {
  neovim.enabled = false
  api.fetchRootDiffFiles.mockReset().mockResolvedValue({ is_git_repository: true, files: [{ path: "a.txt", status: "modified", stamp: "1", added: 1, removed: 1 }, { path: "dir/b.png", status: "added", stamp: "1", added: null, removed: null }] })
  api.fetchRootFileDiff.mockReset().mockImplementation(async (_root, path) => path === "a.txt" ? { supported: true, status: "modified", patch: PATCH } : { supported: false })
  api.fetchRootDiffContent.mockReset().mockReturnValue(new Promise(() => {}))
})
afterEach(cleanup)

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
})
