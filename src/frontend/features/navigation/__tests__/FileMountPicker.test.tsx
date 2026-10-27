import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { FileMountPicker } from "@/features/navigation/FileMountPicker"
import type { DirectoryListing } from "@/lib/domain/files"

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

describe("FileMountPicker", () => {
  beforeEach(() => { api.browseDirectories.mockClear() })

  it("starts at home, opens a clicked directory, goes up, and mounts the path on Enter", async () => {
    const onConfirm = vi.fn(async () => {})
    const onClose = vi.fn()
    render(<FileMountPicker open onClose={onClose} onConfirm={onConfirm} onClone={vi.fn()} />)
    const input = screen.getByLabelText("File mount path")
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
    render(<FileMountPicker open onClose={onClose} onConfirm={async () => { throw new Error("Not a directory: /nope") }} onClone={vi.fn()} />)
    const input = screen.getByLabelText("File mount path")
    fireEvent.change(input, { target: { value: "/nope" } })
    fireEvent.submit(input)
    expect(await screen.findByRole("alert")).toHaveTextContent("Not a directory: /nope")
    expect(onClose).not.toHaveBeenCalled()
  })

  it("clones a pasted git link instead of mounting it, without browsing it as a directory", async () => {
    const onConfirm = vi.fn(async () => {})
    const onClone = vi.fn(async () => {})
    const onClose = vi.fn()
    render(<FileMountPicker open onClose={onClose} onConfirm={onConfirm} onClone={onClone} />)
    const input = screen.getByLabelText("File mount path")
    await waitFor(() => expect(input).toHaveValue("/home/noob"))
    api.browseDirectories.mockClear()

    fireEvent.change(input, { target: { value: " git@github.com:owner/repo.git " } })
    expect(screen.getByRole("button", { name: "Clone" })).toBeInTheDocument()
    fireEvent.submit(input)

    await waitFor(() => expect(onClone).toHaveBeenCalledWith("git@github.com:owner/repo.git"))
    expect(onConfirm).not.toHaveBeenCalled()
    expect(api.browseDirectories).not.toHaveBeenCalled()
    expect(screen.queryByRole("alert")).toBeNull()
    expect(onClose).toHaveBeenCalled()
  })

  it("keeps the window open and shows git's reason when cloning fails", async () => {
    const onClose = vi.fn()
    render(<FileMountPicker open onClose={onClose} onConfirm={vi.fn()} onClone={async () => { throw new Error("Directory already exists: /home/noob/git/repo") }} />)
    const input = screen.getByLabelText("File mount path")
    fireEvent.change(input, { target: { value: "https://github.com/owner/repo" } })
    fireEvent.submit(input)
    expect(await screen.findByRole("alert")).toHaveTextContent("Directory already exists")
    expect(onClose).not.toHaveBeenCalled()
  })
})
