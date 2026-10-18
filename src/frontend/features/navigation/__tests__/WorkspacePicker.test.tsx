import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WorkspacePicker } from "@/features/navigation/WorkspacePicker"
import type { DirectoryListing } from "@/lib/domain/types"

const tree: Record<string, DirectoryListing> = {
  "~": { path: "/home/noob", parent: "/home", directories: [".config", "git"] },
  "/home/noob/git": { path: "/home/noob/git", parent: "/home/noob", directories: ["gigachad-bot"] },
  "/home/noob": { path: "/home/noob", parent: "/home", directories: [".config", "git"] },
}

const api = vi.hoisted(() => ({
  browseDirectories: vi.fn(async (path: string) => {
    const listing = tree[path]
    if (!listing) throw new Error(`Not a directory: ${path}`)
    return listing
  }),
}))
vi.mock("@/lib/backend/api", () => api)

describe("WorkspacePicker", () => {
  beforeEach(() => { api.browseDirectories.mockClear() })

  it("starts at home, opens a clicked directory, goes up, and mounts the path on Enter", async () => {
    const onConfirm = vi.fn(async () => {})
    const onClose = vi.fn()
    render(<WorkspacePicker open onClose={onClose} onConfirm={onConfirm} />)
    const input = screen.getByLabelText("Workspace path")
    await waitFor(() => expect(input).toHaveValue("/home/noob"))

    fireEvent.click(screen.getByRole("button", { name: "git" }))
    await waitFor(() => expect(input).toHaveValue("/home/noob/git"))
    expect(screen.getByRole("button", { name: "gigachad-bot" })).toBeVisible()
    expect(input).toHaveFocus()

    fireEvent.click(screen.getByRole("button", { name: "Open parent directory" }))
    await waitFor(() => expect(input).toHaveValue("/home/noob"))
    fireEvent.click(screen.getByRole("button", { name: "git" }))
    await waitFor(() => expect(input).toHaveValue("/home/noob/git"))

    fireEvent.submit(input)
    await waitFor(() => expect(onConfirm).toHaveBeenCalledWith("/home/noob/git"))
    expect(onClose).toHaveBeenCalled()
  })

  it("keeps the window open and shows the reason when mounting fails", async () => {
    const onClose = vi.fn()
    render(<WorkspacePicker open onClose={onClose} onConfirm={async () => { throw new Error("Not a directory: /nope") }} />)
    const input = screen.getByLabelText("Workspace path")
    fireEvent.change(input, { target: { value: "/nope" } })
    fireEvent.submit(input)
    expect(await screen.findByRole("alert")).toHaveTextContent("Not a directory: /nope")
    expect(onClose).not.toHaveBeenCalled()
  })
})
