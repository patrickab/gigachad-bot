import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor, within, type RenderResult } from "@testing-library/react"
import type { FactoryResolvedSettings, FactoryTaskStart } from "@/lib/domain/factory"
import type { ModelsResponse } from "@/lib/domain/types"

const core = vi.hoisted(() => ({
  startTask: vi.fn<(body: FactoryTaskStart) => Promise<unknown>>(),
  listTasks: vi.fn<(repo: string) => Promise<unknown[]>>(),
  getSettings: vi.fn<(repo: string) => Promise<FactoryResolvedSettings>>(),
  listBranches: vi.fn<(repo: string) => Promise<string[]>>(),
}))
const api = vi.hoisted(() => ({
  callFactory: vi.fn(async (action: string, params: Record<string, unknown>) => {
    const repo = params.repo as string
    if (action === "tasks.start") return core.startTask(params as unknown as FactoryTaskStart)
    if (action === "tasks.list") return { tasks: await core.listTasks(repo) }
    if (action === "settings.get") {
      const resolved = await core.getSettings(repo)
      return { stored: {}, resolved, defaults: { ...resolved, branch: "main" } }
    }
    if (action === "branches.list") return { branches: (await core.listBranches(repo)).map((name) => ({ name, default: name === "main", hasConfig: true })) }
    throw new Error(`unexpected ${action}`)
  }),
  fetchModels: vi.fn<() => Promise<ModelsResponse>>(),
}))

const CATALOG: ModelsResponse = {
  ollama: [],
  providers: [{ label: "Pi", provider_id: "litellm_proxy", source: "pi", models: ["anthropic/claude-sonnet-5-5", "anthropic/claude-opus-5-5"] }],
  defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", sandbox_model: "" },
  tab_order: [],
} as unknown as ModelsResponse

vi.mock("@/lib/backend/api", () => api)

import { NewTaskDialog } from "@/features/factory/NewTaskDialog"

const REPO = "/home/me/git/gigachad-bot"

const SETTINGS: FactoryResolvedSettings = { merge: "review", maxRounds: 3, review: true, preview: false, models: {}, branch: "main" }

beforeEach(() => {
  core.startTask.mockReset().mockResolvedValue({})
  core.getSettings.mockReset().mockResolvedValue(SETTINGS)
  core.listTasks.mockReset().mockResolvedValue([])
  core.listBranches.mockReset().mockResolvedValue(["develop", "feature/x", "main", "release"])
  api.fetchModels.mockReset().mockResolvedValue(CATALOG)
})

function fill(view: ReturnType<typeof render>) {
  fireEvent.change(view.getByLabelText("Title"), { target: { value: " Fix the header " } })
  fireEvent.change(view.getByLabelText("Request"), { target: { value: " Make it sticky " } })
}

/** The pill `name` of the dialog's pill row `group`. */
function pill(view: RenderResult, group: string, name: string): HTMLElement {
  return within(view.getByRole("group", { name: group })).getByRole("button", { name })
}

describe("NewTaskDialog", () => {
  it("starts a task in the repo by path with the repo's settings", async () => {
    const onClose = vi.fn()
    const view = render(<NewTaskDialog repo={REPO} open onClose={onClose} />)
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeChecked())
    expect(core.getSettings).toHaveBeenCalledWith(REPO)

    fill(view)
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    expect(onClose).toHaveBeenCalled()
    expect(core.startTask).toHaveBeenCalledWith({ repo: REPO, title: "Fix the header", request: "Make it sticky" })
  })

  it("sends no choice that equals the repo's setting", async () => {
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const review = view.getByRole("switch", { name: "Review" })
    await waitFor(() => expect(review).toBeEnabled())

    fill(view)
    fireEvent.click(review)
    fireEvent.click(review)
    fireEvent.click(pill(view, "Merge", "Auto"))
    fireEvent.click(pill(view, "Merge", "Approve"))
    fireEvent.click(view.getByLabelText("Branch"))
    fireEvent.click(view.getByRole("option", { name: "main" }))
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(core.startTask).toHaveBeenCalled())
    expect(core.startTask.mock.calls[0][0]).not.toHaveProperty("overrides")
  })

  it("sends the user's review choice and merge override", async () => {
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const review = view.getByRole("switch", { name: "Review" })
    await waitFor(() => expect(review).toBeEnabled())

    fill(view)
    fireEvent.click(review)
    expect(pill(view, "Merge", "Approve")).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(pill(view, "Merge", "Auto"))
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(core.startTask).toHaveBeenCalled())
    expect(core.startTask.mock.calls[0][0].overrides).toEqual({ review: false, merge: "auto" })
  })

  it("sends the model picked per role, the others stay the repo's; the reviewer's only matters with review on", async () => {
    core.getSettings.mockResolvedValue({ ...SETTINGS, review: false, models: { coder: "anthropic/claude-sonnet-5-5" } })
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())
    expect(view.getByLabelText("Model of reviewer")).toBeDisabled()
    fireEvent.click(view.getByRole("switch", { name: "Review" }))
    expect(view.getByLabelText("Model of reviewer")).toBeEnabled()

    fill(view)
    fireEvent.click(view.getByLabelText("Model of coder"))
    expect(view.getByRole("option", { name: "Repo default (anthropic/claude-sonnet-5-5)" })).toBeTruthy()
    fireEvent.click(await view.findByRole("option", { name: "anthropic/claude-opus-5-5" }))
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(core.startTask).toHaveBeenCalled())
    expect(core.startTask.mock.calls[0][0].overrides?.models).toEqual({ coder: "anthropic/claude-opus-5-5" })
  })

  it("offers no workflow, image, sandbox or backlog control", async () => {
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())

    expect(view.queryByLabelText("Workflow")).toBeNull()
    expect(view.queryByLabelText("Image files")).toBeNull()
    expect(view.queryByLabelText("Sandbox")).toBeNull()
    expect(view.getAllByRole("button").map((button) => button.textContent)).not.toContain("Save to backlog")
  })

  it("does not start a task before the title, the request and the repo's settings are there", async () => {
    let deliver: (settings: FactoryResolvedSettings) => void = () => {}
    core.getSettings.mockReturnValue(new Promise((resolve) => { deliver = resolve }))
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    const start = view.getByRole("button", { name: "Start" })

    fill(view)
    expect(start).toBeDisabled()
    expect(pill(view, "Merge", "Auto")).toBeDisabled()
    expect(view.getByRole("switch", { name: "Review" })).toBeDisabled()
    deliver(SETTINGS)
    await waitFor(() => expect(start).toBeEnabled())

    fireEvent.change(view.getByLabelText("Request"), { target: { value: "  " } })
    expect(start).toBeDisabled()
  })

  it("closes at once, before the backend answered the start", async () => {
    core.startTask.mockReturnValue(new Promise(() => {}))
    const onClose = vi.fn()
    const view = render(<NewTaskDialog repo={REPO} open onClose={onClose} />)
    await waitFor(() => expect(view.getByRole("switch", { name: "Review" })).toBeEnabled())

    fill(view)
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(core.startTask).toHaveBeenCalledTimes(1)
  })

  it("offers the repo's branches alphabetically after its default, and sends the branch only when picked", async () => {
    core.getSettings.mockResolvedValue({ ...SETTINGS, branch: "develop" })
    const view = render(<NewTaskDialog repo={REPO} open onClose={vi.fn()} />)
    await waitFor(() => expect(view.getByLabelText("Branch")).toHaveTextContent("Repo default (develop)"))
    expect(core.listBranches).toHaveBeenCalledWith(REPO)
    fill(view)

    fireEvent.click(view.getByLabelText("Branch"))
    expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual(["Repo default (develop)", "develop", "feature/x", "main", "release"])
    fireEvent.click(view.getByRole("option", { name: "feature/x" }))
    fireEvent.click(view.getByRole("button", { name: "Start" }))

    await waitFor(() => expect(core.startTask).toHaveBeenCalled())
    expect(core.startTask.mock.calls[0][0].overrides).toEqual({ branch: "feature/x" })
  })
})
