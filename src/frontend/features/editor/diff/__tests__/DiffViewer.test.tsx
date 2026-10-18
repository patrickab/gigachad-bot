import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, within } from "@testing-library/react"
import { DiffViewer } from "@/features/editor/diff/DiffViewer"

const api = vi.hoisted(() => ({
  fetchRootDiffFiles: vi.fn<typeof import("@/lib/backend/api").fetchRootDiffFiles>(),
  fetchRootFileDiff: vi.fn<typeof import("@/lib/backend/api").fetchRootFileDiff>(),
}))
vi.mock("@/lib/backend/api", () => api)

const PATCH = ["diff --git a/a.txt b/a.txt", "--- a/a.txt", "+++ b/a.txt", "@@ -1,2 +1,2 @@", " keep", "-old", "+new"].join("\n")

beforeEach(() => {
  api.fetchRootDiffFiles.mockReset().mockResolvedValue({ is_git_repository: true, files: [{ path: "a.txt", status: "modified" }, { path: "dir/b.png", status: "added" }] })
  api.fetchRootFileDiff.mockReset().mockImplementation(async (_root, path) => path === "a.txt" ? { supported: true, status: "modified", patch: PATCH } : { supported: false })
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
    expect(view.getByRole("button", { name: /^a\.txt/ })).toHaveAttribute("aria-current", "page")
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
})
