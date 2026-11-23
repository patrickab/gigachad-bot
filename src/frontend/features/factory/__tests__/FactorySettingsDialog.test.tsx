import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor, within } from "@testing-library/react"
import type { ModelsResponse } from "@/lib/domain/types"
import type { FactoryActions, FactoryRepoSettings, FactoryStoredSettings } from "@/lib/domain/factory"

const core = vi.hoisted(() => ({
  getSettings: vi.fn<(repo: string) => Promise<FactoryRepoSettings>>(),
  setSettings: vi.fn<(repo: string, settings: FactoryStoredSettings) => Promise<unknown>>(),
  listBranches: vi.fn<(repo: string) => Promise<string[]>>(),
  getModels: vi.fn<() => Promise<string[]>>(),
  setModels: vi.fn<(models: string[]) => Promise<string[]>>(),
  calls: [] as string[],
}))
const api = vi.hoisted(() => ({
  callFactory: vi.fn(async (action: string, params: Record<string, unknown>) => {
    const repo = params.repo as string
    core.calls.push(action)
    if (action === "models.get") return { models: await core.getModels() }
    if (action === "models.set") return { models: await core.setModels(params.models as string[]) }
    if (action === "settings.get") return core.getSettings(repo)
    if (action === "settings.set") return core.setSettings(repo, params.settings as FactoryStoredSettings)
    if (action === "branches.list") return { branches: (await core.listBranches(repo)).map((name) => ({ name, default: name === "main", hasConfig: true })) }
    throw new Error(`unexpected ${action}`)
  }),
  fetchModels: vi.fn<() => Promise<ModelsResponse>>(),
  getFactoryActions: vi.fn<() => Promise<FactoryActions>>(),
}))

/** The part of the core's `ACTIONS` the dialog reads: the bounds of `maxRounds`. */
const ACTIONS: FactoryActions = {
  actions: {
    "settings.set": {
      summary: "Replace the repository's stored settings",
      params: { type: "object", required: ["repo", "settings"], properties: { settings: { type: "object", properties: { maxRounds: { type: "integer", minimum: 1, maximum: 12 } } } } },
      mutates: true,
      hostOnly: false,
    },
  },
}

vi.mock("@/lib/backend/api", () => api)

import { FactorySettingsDialog } from "@/features/factory/FactorySettingsDialog"

const REPO = "/home/me/git/gigachad-bot"

const STORED: FactoryStoredSettings = { merge: "review", maxRounds: 3, review: true, preview: true, models: { coder: "old-model" } }

function settings(stored: FactoryStoredSettings): FactoryRepoSettings {
  return {
    stored,
    resolved: { merge: "auto", maxRounds: 2, review: false, preview: false, branch: "main", ...stored, models: stored.models ?? {} },
    defaults: { merge: "auto", maxRounds: 2, review: false, preview: false, models: {}, branch: "main" },
  }
}

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

/** The factory models list as the core holds it. */
const MODELS = ["openai-codex/fast", "openrouter/vendor/model"]

beforeEach(() => {
  core.calls = []
  core.getModels.mockReset().mockResolvedValue(MODELS)
  core.setModels.mockReset().mockImplementation(async (models) => models)
  core.listBranches.mockReset().mockResolvedValue(["develop", "feature/x", "main", "release"])
  core.getSettings.mockReset().mockResolvedValue(settings(STORED))
  api.fetchModels.mockReset().mockResolvedValue(CATALOG)
  api.getFactoryActions.mockReset().mockResolvedValue(ACTIONS)
  core.setSettings.mockReset().mockImplementation(async (_repo, saved) => settings(saved))
})

describe("FactorySettingsDialog", () => {
  it("saves the repo's edited settings and models by path, dropping empty model selectors", async () => {
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={onClose} />)
    const rounds = await view.findByLabelText("Max rounds")
    expect(core.getSettings).toHaveBeenCalledWith(REPO)

    const pill = (group: string, name: string) => within(view.getByRole("group", { name: group })).getByRole("button", { name })
    fireEvent.change(rounds, { target: { value: "5" } })
    fireEvent.click(pill("Merge", "Auto"))
    // The reviewer's model is picked while review is on: off, its select is disabled.
    fireEvent.click(view.getByLabelText("Model of reviewer"))
    fireEvent.click(await view.findByRole("option", { name: "openai-codex/fast" }))
    fireEvent.click(view.getByRole("switch", { name: "Review" }))
    expect(view.getByLabelText("Model of reviewer")).toBeDisabled()
    fireEvent.click(view.getByRole("switch", { name: "Preview" }))
    fireEvent.click(view.getByLabelText("Model of coder"))
    fireEvent.click(view.getByRole("option", { name: "Pi default" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(core.setSettings).toHaveBeenCalledWith(REPO, { merge: "auto", maxRounds: 5, review: false, preview: false, models: { reviewer: "openai-codex/fast" } })
  })

  it("edits nothing of the old engine: no sandbox, timeouts, scout, planner or agent markdown", async () => {
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    await view.findByLabelText("Max rounds")

    for (const label of ["Sandbox", "Model of scout", "Model of planner", "Markdown of coder"]) expect(view.queryByLabelText(label)).toBeNull()
    expect(view.queryByRole("tab")).toBeNull()
    expect(view.getAllByRole("spinbutton")).toHaveLength(1)
  })

  it("does not save a round count outside the bounds the core declares", async () => {
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    const rounds = await view.findByLabelText("Max rounds")
    await waitFor(() => expect(rounds).toHaveAttribute("max", "12"))
    expect(rounds).toHaveAttribute("min", "1")
    const save = view.getByRole("button", { name: "Save" })

    for (const value of ["", "0", "13", "2.5"]) {
      fireEvent.change(rounds, { target: { value } })
      expect(save).toBeDisabled()
    }
    fireEvent.change(rounds, { target: { value: "12" } })
    expect(save).toBeEnabled()
  })

  it("leaves a setting the file does not hold unset unless the user changes it, so it follows the defaults", async () => {
    core.getSettings.mockResolvedValue(settings({}))
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={onClose} />)
    await view.findByLabelText("Max rounds")

    fireEvent.click(view.getByRole("switch", { name: "Preview" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(core.setSettings).toHaveBeenCalledWith(REPO, { preview: true, models: {} })
  })

  it("shows the core's refusal inline and stays open", async () => {
    core.setSettings.mockRejectedValueOnce(new Error("settings.maxRounds must be at most 20"))
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={onClose} />)
    await view.findByLabelText("Max rounds")

    fireEvent.click(view.getByRole("button", { name: "Save" }))

    expect(await view.findByRole("alert")).toHaveTextContent("settings.maxRounds must be at most 20")
    expect(onClose).not.toHaveBeenCalled()
  })

  it("offers exactly the factory models list per agent, a saved model not on it marked", async () => {
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    fireEvent.click(await view.findByLabelText("Model of coder"))
    await waitFor(() => expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Pi default", "openai-codex/fast", "openrouter/vendor/model", "old-model (not in the list)",
    ]))
    expect(view.getByRole("option", { name: "old-model (not in the list)" })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(view.getByRole("option", { name: "openrouter/vendor/model" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(core.setSettings).toHaveBeenCalledWith(REPO, { ...STORED, models: { coder: "openrouter/vendor/model" } }))
    expect(core.setModels).not.toHaveBeenCalled()
  })

  it("edits the factory models list: removes one, adds one gigachad knows, and saves it before the settings", async () => {
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={onClose} />)
    const list = await view.findByRole("list", { name: "Factory models" })
    expect(within(list).getAllByRole("listitem").map((item) => item.textContent)).toEqual(MODELS)

    fireEvent.click(view.getByRole("button", { name: "Remove openai-codex/fast" }))
    fireEvent.click(view.getByLabelText("Add model"))
    // The served models of the chat selector as gateway ids, without those already listed or unavailable.
    await waitFor(() => expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Add model", "google/gemini-pro", "local-model", "openai-codex/fast", "openrouter/hand-added",
    ]))
    fireEvent.click(view.getByRole("option", { name: "google/gemini-pro" }))
    expect(within(list).getAllByRole("listitem").map((item) => item.textContent)).toEqual(["openrouter/vendor/model", "google/gemini-pro"])

    fireEvent.click(view.getByLabelText("Model of reviewer"))
    expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual(["Pi default", "openrouter/vendor/model", "google/gemini-pro"])
    fireEvent.click(view.getByRole("option", { name: "google/gemini-pro" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(core.setModels).toHaveBeenCalledWith(["openrouter/vendor/model", "google/gemini-pro"])
    expect(core.calls.indexOf("models.set")).toBeLessThan(core.calls.indexOf("settings.set"))
    expect(core.setSettings).toHaveBeenCalledWith(REPO, { ...STORED, models: { coder: "old-model", reviewer: "google/gemini-pro" } })
  })

  it("keeps saved selections when loading the factory models fails and permits clearing them", async () => {
    core.getModels.mockRejectedValue(new Error("Could not load the factory models"))
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    expect(await view.findByRole("alert")).toHaveTextContent("Could not load the factory models")
    expect(view.queryByLabelText("Add model")).toBeNull()
    fireEvent.click(view.getByLabelText("Model of coder"))
    expect(view.getByRole("option", { name: "old-model (not in the list)" })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(view.getByRole("option", { name: "Pi default" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(core.setSettings).toHaveBeenCalledWith(REPO, { ...STORED, models: {} }))
    expect(core.setModels).not.toHaveBeenCalled()
  })

  it("picks the branch from the repo's branches, the default saved as no branch", async () => {
    core.getSettings.mockResolvedValue(settings({ ...STORED, branch: "develop" }))
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    const branch = await view.findByLabelText("Branch")
    expect(branch).toHaveTextContent("develop")
    expect(core.listBranches).toHaveBeenCalledWith(REPO)

    fireEvent.click(branch)
    await waitFor(() => expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual(["Default (main)", "develop", "feature/x", "main", "release"]))
    fireEvent.click(view.getByRole("option", { name: "release" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(core.setSettings).toHaveBeenCalledTimes(1))
    expect(core.setSettings.mock.calls[0][1]).toMatchObject({ branch: "release" })

    fireEvent.click(branch)
    fireEvent.click(view.getByRole("option", { name: "Default (main)" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(core.setSettings).toHaveBeenCalledTimes(2))
    expect(core.setSettings.mock.calls[1][1]).not.toHaveProperty("branch")
  })

  it("keeps a saved branch that no longer exists selectable, marked missing", async () => {
    core.getSettings.mockResolvedValue(settings({ ...STORED, branch: "gone" }))
    const view = render(<FactorySettingsDialog repo={REPO} open onClose={vi.fn()} />)
    const branch = await view.findByLabelText("Branch")
    await waitFor(() => expect(branch).toHaveTextContent("gone (missing)"))
  })
})
