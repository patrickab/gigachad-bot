import { describe, it, expect, vi, beforeEach } from "vitest"
import { fireEvent, render, waitFor } from "@testing-library/react"
import type { FactoryDefaults } from "@/lib/domain/types"

const api = vi.hoisted(() => ({
  fetchFactoryDefaults: vi.fn<() => Promise<FactoryDefaults>>(),
  saveFactoryDefaults: vi.fn<(defaults: FactoryDefaults) => Promise<FactoryDefaults>>(),
}))

vi.mock("@/lib/backend/api", () => api)

import { FactorySettingsDialog } from "@/features/factory/FactorySettingsDialog"

const DEFAULTS: FactoryDefaults = {
  settings: { merge: "review", max_rounds: 3, sandbox: true, preview: true, models: { coder: "old-model" }, agent_timeout_s: 3600, command_timeout_s: 1800 },
  agents: { scout: "scout text", planner: "planner text", coder: "coder text", reviewer: "reviewer text" },
}

beforeEach(() => {
  api.fetchFactoryDefaults.mockReset().mockResolvedValue(DEFAULTS)
  api.saveFactoryDefaults.mockReset().mockImplementation(async (saved) => saved)
})

describe("FactorySettingsDialog", () => {
  it("saves the edited settings, models and the agent markdown, dropping empty model selectors", async () => {
    const onClose = vi.fn()
    const view = render(<FactorySettingsDialog open onClose={onClose} />)
    const rounds = await view.findByLabelText("Max rounds")

    fireEvent.change(rounds, { target: { value: "5" } })
    fireEvent.click(view.getByLabelText("Preview"))
    fireEvent.change(view.getByLabelText("Model of scout"), { target: { value: "fast" } })
    fireEvent.change(view.getByLabelText("Model of coder"), { target: { value: " " } })
    expect(view.getByLabelText("Markdown of scout")).toHaveValue("scout text")
    fireEvent.click(view.getByRole("tab", { name: "reviewer" }))
    fireEvent.change(view.getByLabelText("Markdown of reviewer"), { target: { value: "new reviewer" } })
    fireEvent.click(view.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(onClose).toHaveBeenCalled())
    expect(api.saveFactoryDefaults).toHaveBeenCalledWith({
      settings: { ...DEFAULTS.settings, max_rounds: 5, preview: false, models: { scout: "fast" } },
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
})
