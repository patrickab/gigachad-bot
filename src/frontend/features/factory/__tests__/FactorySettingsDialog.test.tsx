import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor } from "@testing-library/react"
import type { FactoryDefaults, ModelsResponse } from "@/lib/domain/types"

const api = vi.hoisted(() => ({
  fetchFactoryDefaults: vi.fn<() => Promise<FactoryDefaults>>(),
  fetchModels: vi.fn<() => Promise<ModelsResponse>>(),
  saveFactoryDefaults: vi.fn<(defaults: FactoryDefaults) => Promise<FactoryDefaults>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { FactorySettingsDialog } from "@/features/factory/FactorySettingsDialog"

const DEFAULTS: FactoryDefaults = {
  settings: { merge: "review", max_rounds: 3, sandbox: true, preview: true, models: { coder: "old-model" }, agent_timeout_s: 3600, command_timeout_s: 1800 },
  agents: { scout: "scout text", planner: "planner text", coder: "coder text", reviewer: "reviewer text" },
}

const CATALOG: ModelsResponse = {
  ollama: ["local-model"],
  providers: [
    { label: "OMP", litellm_id: "litellm_proxy", source: "omp", models: ["openai-codex/fast", "openrouter/vendor/model"] },
    { label: "OpenRouter", litellm_id: "openrouter", models: ["not-added-from-omp"] },
  ],
  defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", omp_model: "" },
  tab_order: [],
}

beforeEach(() => {
  api.fetchFactoryDefaults.mockReset().mockResolvedValue(DEFAULTS)
  api.fetchModels.mockReset().mockResolvedValue(CATALOG)
  api.saveFactoryDefaults.mockReset().mockImplementation(async (saved) => saved)
})

describe("FactorySettingsDialog", () => {
  it("saves the edited settings, models and the agent markdown, dropping empty model selectors", async () => {
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog open onClose={onClose} />)
    const rounds = await view.findByLabelText("Max rounds")

    fireEvent.change(rounds, { target: { value: "5" } })
    fireEvent.click(view.getByLabelText("Preview"))
    fireEvent.click(view.getByLabelText("Model of scout"))
    fireEvent.click(await view.findByRole("option", { name: "openai-codex/fast" }))
    fireEvent.click(view.getByLabelText("Model of coder"))
    fireEvent.click(view.getByRole("option", { name: "Agent's own model" }))
    expect(view.getByLabelText("Markdown of scout")).toHaveValue("scout text")
    fireEvent.click(view.getByRole("tab", { name: "reviewer" }))
    fireEvent.change(view.getByLabelText("Markdown of reviewer"), { target: { value: "new reviewer" } })
    fireEvent.click(view.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(api.saveFactoryDefaults).toHaveBeenCalledWith({
      settings: { ...DEFAULTS.settings, max_rounds: 5, preview: false, models: { scout: "openai-codex/fast" } },
      agents: { ...DEFAULTS.agents, reviewer: "new reviewer" },
    })
  })

  it("shows the engine's refusal inline and stays open", async () => {
    api.saveFactoryDefaults.mockRejectedValueOnce(new Error("agent coder: invalid front matter"))
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog open onClose={onClose} />)
    await view.findByLabelText("Max rounds")

    fireEvent.click(view.getByRole("button", { name: "Save" }))

    expect(await view.findByRole("alert")).toHaveTextContent("agent coder: invalid front matter")
    expect(onClose).not.toHaveBeenCalled()
  })

  it("offers only added OMP models and preserves native selectors with slashes", async () => {
    const view = render(<FactorySettingsDialog open onClose={vi.fn()} />)
    fireEvent.click(await view.findByLabelText("Model of coder"))
    expect(await view.findByRole("option", { name: "openai-codex/fast" })).toBeVisible()
    expect(view.getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Agent's own model", "openai-codex/fast", "openrouter/vendor/model", "old-model (saved, not added)",
    ])
    expect(view.getByRole("option", { name: "old-model (saved, not added)" })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(view.getByRole("option", { name: "openrouter/vendor/model" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.saveFactoryDefaults).toHaveBeenCalledWith({
      ...DEFAULTS, settings: { ...DEFAULTS.settings, models: { coder: "openrouter/vendor/model" } },
    }))
  })

  it("keeps saved selections when loading models fails and permits clearing them", async () => {
    api.fetchModels.mockRejectedValue(new Error("Could not load OMP models"))
    const view = render(<FactorySettingsDialog open onClose={vi.fn()} />)
    expect(await view.findByRole("alert")).toHaveTextContent("Could not load OMP models")
    fireEvent.click(view.getByLabelText("Model of coder"))
    expect(view.getByRole("option", { name: "old-model (saved, not added)" })).toHaveAttribute("aria-selected", "true")
    fireEvent.click(view.getByRole("option", { name: "Agent's own model" }))
    fireEvent.click(view.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(api.saveFactoryDefaults).toHaveBeenCalledWith({
      ...DEFAULTS, settings: { ...DEFAULTS.settings, models: {} },
    }))
  })
})
