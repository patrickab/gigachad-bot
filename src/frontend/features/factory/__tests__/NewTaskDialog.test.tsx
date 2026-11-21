import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor } from "@testing-library/react"
import type { FactorySettings, FactoryTaskCreate } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  createFactoryTask: vi.fn<(body: FactoryTaskCreate) => Promise<unknown>>(),
  getFactorySettings: vi.fn<(repo: string) => Promise<FactorySettings>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { NewTaskDialog } from "@/features/factory/NewTaskDialog"

const REPO = "/home/me/git/gigachad-bot"

const SETTINGS: FactorySettings = { merge: "review", max_rounds: 3, review: true, worktree: true, preview: false, models: {}, branch: null }

beforeEach(() => {
  api.createFactoryTask.mockReset().mockResolvedValue({})
  api.getFactorySettings.mockReset().mockResolvedValue(SETTINGS)
})

function fill(view: ReturnType<typeof render>) {
  fireEvent.change(view.getByLabelText("Title"), { target: { value: " Fix the header " } })
  fireEvent.change(view.getByLabelText("Request"), { target: { value: " Make it sticky " } })
}

describe("NewTaskDialog", () => {
  it("starts a task in the repo by path, in a worktree when the repo's settings say so", async () => {
    const onClose = vi.fn()
    const onCreated = vi.fn()
    const view = render(<NewTaskDialog repo={REPO} open onClose={onClose} onCreated={onCreated} />)
    await waitFor(() => expect(view.getByLabelText("Worktree")).toBeChecked())
    expect(api.getFactorySettings).toHaveBeenCalledWith(REPO)

    fill(view)
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(api.createFactoryTask).toHaveBeenCalledWith({ repo: REPO, title: "Fix the header", request: "Make it sticky", overrides: { worktree: true, branch: null } })
    expect(onCreated).toHaveBeenCalledTimes(1)
  })

  it("defaults to the current checkout for a repo without worktrees, and sends the user's choice and merge override", async () => {
    api.getFactorySettings.mockResolvedValue({ ...SETTINGS, worktree: false })
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const worktree = view.getByLabelText("Worktree")
    await waitFor(() => expect(worktree).toBeEnabled())
    expect(worktree).not.toBeChecked()

    fill(view)
    fireEvent.click(worktree)
    fireEvent.click(view.getByRole("button", { name: "Overrides" }))
    fireEvent.click(view.getByLabelText("Merge"))
    expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual(["Repo default (review)", "Review", "Auto"])
    fireEvent.click(view.getByRole("option", { name: "Auto" }))
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(api.createFactoryTask).toHaveBeenCalled())
    expect(api.createFactoryTask.mock.calls[0][0].overrides).toEqual({ worktree: true, branch: null, merge: "auto" })
  })

  it("offers no workflow, image, sandbox or backlog control", async () => {
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    await waitFor(() => expect(view.getByLabelText("Worktree")).toBeEnabled())
    fireEvent.click(view.getByRole("button", { name: "Overrides" }))

    expect(view.queryByLabelText("Workflow")).toBeNull()
    expect(view.queryByLabelText("Image files")).toBeNull()
    expect(view.queryByLabelText("Sandbox")).toBeNull()
    expect(view.getAllByRole("button").map((button) => button.textContent)).not.toContain("Save to backlog")
  })

  it("does not start a task before the title, the request and the repo's settings are there", async () => {
    let deliver: (settings: FactorySettings) => void = () => {}
    api.getFactorySettings.mockReturnValue(new Promise((resolve) => { deliver = resolve }))
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const start = view.getByRole("button", { name: "Start" })

    fill(view)
    expect(start).toBeDisabled()
    expect(view.getByLabelText("Worktree")).toBeDisabled()
    deliver(SETTINGS)
    await waitFor(() => expect(start).toBeEnabled())

    fireEvent.change(view.getByLabelText("Request"), { target: { value: "  " } })
    expect(start).toBeDisabled()
  })

  it("shows the backend's refusal inline and stays open", async () => {
    api.createFactoryTask.mockRejectedValueOnce(new Error("The checkout already has a task"))
    const onClose = vi.fn()
    const onCreated = vi.fn()
    const view = render(<NewTaskDialog repo={REPO} open onClose={onClose} onCreated={onCreated} />)
    await waitFor(() => expect(view.getByLabelText("Worktree")).toBeEnabled())

    fill(view)
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    expect(await view.findByRole("alert")).toHaveTextContent("The checkout already has a task")
    expect(onClose).not.toHaveBeenCalled()
    expect(onCreated).not.toHaveBeenCalled()
  })

  it("prefills the branch from the repo's setting, sends the user's branch, and null for an empty field", async () => {
    api.getFactorySettings.mockResolvedValue({ ...SETTINGS, branch: "develop" })
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const branch = view.getByLabelText("Branch")
    await waitFor(() => expect(branch).toHaveValue("develop"))
    fill(view)
    fireEvent.change(branch, { target: { value: "" } })
    fireEvent.click(view.getByRole("button", { name: "Start" }))
    await waitFor(() => expect(api.createFactoryTask).toHaveBeenCalled())
    expect(api.createFactoryTask.mock.calls[0][0].overrides).toEqual({ worktree: true, branch: null })
  })

  it("disables the branch for a task in the current checkout and does not send it", async () => {
    api.getFactorySettings.mockResolvedValue({ ...SETTINGS, branch: "develop" })
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const branch = view.getByLabelText("Branch")
    await waitFor(() => expect(branch).toBeEnabled())
    fireEvent.click(view.getByLabelText("Worktree"))
    expect(branch).toBeDisabled()
    fill(view)
    fireEvent.click(view.getByRole("button", { name: "Start" }))
    await waitFor(() => expect(api.createFactoryTask).toHaveBeenCalled())
    expect(api.createFactoryTask.mock.calls[0][0].overrides).toEqual({ worktree: false })
  })
})
