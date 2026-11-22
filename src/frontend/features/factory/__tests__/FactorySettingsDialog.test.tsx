import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor, within } from "@testing-library/react"
import type { ModelsResponse } from "@/lib/domain/types"
import type { FactorySettings } from "@/lib/domain/factory"

const api = vi.hoisted(() => ({
  getFactorySettings: vi.fn<(repo: string) => Promise<FactorySettings>>(),
  fetchModels: vi.fn<() => Promise<ModelsResponse>>(),
  putFactorySettings: vi.fn<(repo: string, settings: FactorySettings) => Promise<FactorySettings>>(),
  listFactoryBranches: vi.fn<(repo: string) => Promise<string[]>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { FactorySettingsDialog } from "@/features/factory/FactorySettingsDialog"

const REPO = "/home/me/git/gigachad-bot"

const SETTINGS: FactorySettings = { merge: "review", max_rounds: 3, review: true, worktree: true, preview: true, models: { coder: "old-model" }, source_branch: null, target_branch: null }

const CATALOG: ModelsResponse = {
  ollama: ["local-model"],
  providers: [
    { label: "Pi", provider_id: "litellm_proxy", source: "pi", models: ["openai-codex/fast", "openrouter/vendor/model"] },
    { label: "OpenRouter", provider_id: "openrouter", models: ["hand-added", "gone"] },
    { label: "Gemini", provider_id: "gemini", models: ["gemini-pro"] },
  ],
  defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", sandbox_model: "" },
  tab_order: [],
  unavailable: ["openrouter/gone"],
}

beforeEach(() => {
  api.listFactoryBranches.mockReset().mockResolvedValue(["develop", "feature/x", "main", "release"])
  api.getFactorySettings.mockReset().mockResolvedValue(SETTINGS)
  api.fetchModels.mockReset().mockResolvedValue(CATALOG)
  api.putFactorySettings.mockReset().mockImplementation(async (_repo, saved) => saved)
})

describe("FactorySettingsDialog", () => {
  it("saves the repo's edited settings and models by path, dropping empty model selectors", async () => {
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={onClose} />)
    const rounds = await view.findByLabelText("Max rounds")
    expect(api.getFactorySettings).toHaveBeenCalledWith(REPO)

    const pill = (group: string, name: string) => within(view.getByRole("group", { name: group })).getByRole("button", { name })
    fireEvent.change(rounds, { target: { value: "5" } })
    fireEvent.click(pill("Merge", "Auto"))
    // The reviewer's model is picked while review is on: off, its select is disabled.
    fireEvent.click(view.getByLabelText("Model of reviewer"))
    fireEvent.click(await view.findByRole("option", { name: "openai-codex/fast" }))
    fireEvent.click(view.getByRole("switch", { name: "Review" }))
    expect(view.getByLabelText("Model of reviewer")).toBeDisabled()
    fireEvent.click(pill("Workspace", "Checkout"))
    fireEvent.click(view.getByRole("switch", { name: "Preview" }))
    fireEvent.click(view.getByLabelText("Model of coder"))
    fireEvent.click(view.getByRole("option", { name: "Pi default" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(api.putFactorySettings).toHaveBeenCalledWith(REPO, {
      merge: "auto", max_rounds: 5, review: false, worktree: false, preview: false, models: { reviewer: "openai-codex/fast" }, source_branch: null, target_branch: null,
    })
  })

  it("edits nothing of the old engine: no sandbox, timeouts, scout, planner or agent markdown", async () => {
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    await view.findByLabelText("Max rounds")

    for (const label of ["Sandbox", "Model of scout", "Model of planner", "Markdown of coder"]) expect(view.queryByLabelText(label)).toBeNull()
    expect(view.queryByRole("tab")).toBeNull()
    expect(view.getAllByRole("spinbutton")).toHaveLength(1)
  })

  it("does not save a round count outside 1 to 20", async () => {
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    const rounds = await view.findByLabelText("Max rounds")
    const save = view.getByRole("button", { name: "Save" })

    for (const value of ["", "0", "21", "2.5"]) {
      fireEvent.change(rounds, { target: { value } })
      expect(save).toBeDisabled()
    }
    fireEvent.change(rounds, { target: { value: "20" } })
    expect(save).toBeEnabled()
  })

  it("shows the backend's refusal inline and stays open", async () => {
    api.putFactorySettings.mockRejectedValueOnce(new Error("Unknown repo"))
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={onClose} />)
    await view.findByLabelText("Max rounds")

    fireEvent.click(view.getByRole("button", { name: "Save" }))

    expect(await view.findByRole("alert")).toHaveTextContent("Unknown repo")
    expect(onClose).not.toHaveBeenCalled()
  })

  it("offers every served model of the selector as a gateway id and preserves native selectors with slashes", async () => {
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    fireEvent.click(await view.findByLabelText("Model of coder"))
    expect(await view.findByRole("option", { name: "openai-codex/fast" })).toBeVisible()
    expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Pi default", "google/gemini-pro", "local-model", "openai-codex/fast", "openrouter/hand-added", "openrouter/vendor/model",
      "old-model (saved, not added)",
    ])
    expect(view.getByRole("option", { name: "old-model (saved, not added)" })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(view.getByRole("option", { name: "openrouter/vendor/model" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.putFactorySettings).toHaveBeenCalledWith(REPO, { ...SETTINGS, models: { coder: "openrouter/vendor/model" } }))
  })

  it("keeps saved selections when loading models fails and permits clearing them", async () => {
    api.fetchModels.mockRejectedValue(new Error("Could not load Pi models"))
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    expect(await view.findByRole("alert")).toHaveTextContent("Could not load Pi models")
    fireEvent.click(view.getByLabelText("Model of coder"))
    expect(view.getByRole("option", { name: "old-model (saved, not added)" })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(view.getByRole("option", { name: "Pi default" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.putFactorySettings).toHaveBeenCalledWith(REPO, { ...SETTINGS, models: {} }))
  })

  it("picks the source and target branch from the repo's branches, the default saved as null", async () => {
    api.getFactorySettings.mockResolvedValue({ ...SETTINGS, source_branch: "develop" })
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    const source = await view.findByLabelText("Source branch")
    const target = view.getByLabelText("Target branch")
    expect(source).toHaveTextContent("develop")
    expect(target).toHaveTextContent("Default (same as source)")
    expect(api.listFactoryBranches).toHaveBeenCalledWith(REPO)

    fireEvent.click(target)
    await waitFor(() => expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual(["Default (same as source)", "develop", "feature/x", "main", "release"]))
    fireEvent.click(view.getByRole("option", { name: "release" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.putFactorySettings).toHaveBeenCalledTimes(1))
    expect(api.putFactorySettings.mock.calls[0][1]).toMatchObject({ source_branch: "develop", target_branch: "release" })

    fireEvent.click(source)
    fireEvent.click(view.getByRole("option", { name: "Default (origin HEAD)" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.putFactorySettings).toHaveBeenCalledTimes(2))
    expect(api.putFactorySettings.mock.calls[1][1]).toMatchObject({ source_branch: null, target_branch: "release" })
  })

  it("keeps a saved branch that no longer exists selectable, marked missing", async () => {
    api.getFactorySettings.mockResolvedValue({ ...SETTINGS, target_branch: "gone" })
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    const target = await view.findByLabelText("Target branch")
    await waitFor(() => expect(target).toHaveTextContent("gone (missing)"))
  })
})
