import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import type { ModelsResponse, OmpCatalog } from "@/lib/domain/types"

const fetchOmpCatalog = vi.hoisted(() => vi.fn())
vi.mock("@/lib/backend/api", () => ({ fetchOmpCatalog }))

const models: ModelsResponse = {
  ollama: ["ollama/gemma4:31b-cloud"],
  providers: [{ label: "Gemini", litellm_id: "gemini", models: ["gemini-3.1-pro"] }],
  defaults: { default_model: "ollama/gemma4:31b-cloud", small_model: "ollama/gemma4:31b-cloud", vision_model: "ollama/gemma4:31b-cloud", memory_model: "ollama/gemma4:31b-cloud", omp_model: "ollama/gemma4:31b-cloud" },
  tab_order: [],
}

const onlineCatalog: OmpCatalog = {
  installed: true,
  online: true,
  gateway: "http://127.0.0.1:4000/v1",
  litellm_id: "litellm_proxy",
  providers: [{ id: "anthropic", label: "Anthropic", models: [{ id: "anthropic/claude-opus-5", name: "Claude Opus 5", vision: true, context_window: 200000 }] }],
  error: null,
}

function openSettings() {
  fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
  fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
}

describe("ModelDropdown OMP source", () => {
  beforeEach(() => {
    fetchOmpCatalog.mockReset()
  })

  it("adds a picked OMP model as a proxy-prefixed provider entry", async () => {
    fetchOmpCatalog.mockResolvedValue(onlineCatalog)
    const onProvidersChange = vi.fn().mockResolvedValue(undefined)
    render(<ModelDropdown models={models} selectedModel="" onSelect={vi.fn()} onProvidersChange={onProvidersChange} onDefaultsChange={vi.fn()} />)

    openSettings()
    fireEvent.click(await screen.findByText("Anthropic"))
    fireEvent.click(await screen.findByText("claude-opus-5"))

    await waitFor(() => expect(onProvidersChange).toHaveBeenCalled())
    expect(onProvidersChange.mock.calls[0][0]).toEqual([
      { label: "Gemini", litellm_id: "gemini", models: ["gemini-3.1-pro"] },
      { label: "OMP", litellm_id: "litellm_proxy", source: "omp", models: ["anthropic/claude-opus-5"] },
    ])
  })

  it("removes an already-picked OMP model instead of duplicating it", async () => {
    fetchOmpCatalog.mockResolvedValue(onlineCatalog)
    const onProvidersChange = vi.fn().mockResolvedValue(undefined)
    const withPick: ModelsResponse = {
      ...models,
      providers: [...models.providers, { label: "OMP", litellm_id: "litellm_proxy", source: "omp", models: ["anthropic/claude-opus-5"] }],
    }
    render(<ModelDropdown models={withPick} selectedModel="" onSelect={vi.fn()} onProvidersChange={onProvidersChange} onDefaultsChange={vi.fn()} />)

    openSettings()
    fireEvent.click(await screen.findByText("1 of 1 added"))
    fireEvent.click(await screen.findByText("claude-opus-5"))

    await waitFor(() => expect(onProvidersChange).toHaveBeenCalled())
    expect(onProvidersChange.mock.calls[0][0][1].models).toEqual([])
  })

  it("only probes the gateway once the settings panel is opened", async () => {
    fetchOmpCatalog.mockResolvedValue(onlineCatalog)
    render(<ModelDropdown models={models} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    expect(fetchOmpCatalog).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
    await waitFor(() => expect(fetchOmpCatalog).toHaveBeenCalledTimes(1))
  })

  it("lists providers in the persisted order, unlisted providers last", async () => {
    fetchOmpCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    const withThree: ModelsResponse = {
      ...models,
      providers: [
        { label: "Gemini", litellm_id: "gemini", models: ["gemini-3.1-pro"] },
        { label: "DeepSeek", litellm_id: "deepseek", models: ["deepseek-v4"] },
      ],
      tab_order: ["DeepSeek", "Ollama"],
    }
    render(<ModelDropdown models={withThree} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onTabOrderChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))

    const tabBar = (await screen.findByRole("button", { name: "DeepSeek" })).closest("div")?.parentElement
    const labels = Array.from(tabBar?.querySelectorAll("button") ?? []).map((button) => button.textContent)
    expect(labels).toEqual(["DeepSeek", "Ollama", "Gemini"])
  })

  it("sorts models alphabetically within each provider", async () => {
    fetchOmpCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    const unsorted: ModelsResponse = {
      ...models,
      ollama: ["ollama/zulu", "ollama/alpha"],
      providers: [{ label: "Gemini", litellm_id: "gemini", models: ["zeta", "alpha"] }],
    }
    render(<ModelDropdown models={unsorted} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Ollama" }))
    expect(screen.getAllByRole("menuitemradio").map((row) => row.textContent)).toEqual(["alpha", "zulu"])

    fireEvent.click(screen.getByRole("button", { name: "Gemini" }))
    expect(screen.getAllByRole("menuitemradio").map((row) => row.textContent)).toEqual(["alpha", "zeta"])
  })

  it("opens on the provider tab holding the currently selected model, not Ollama", async () => {
    fetchOmpCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    render(<ModelDropdown models={models} selectedModel="gemini/gemini-3.1-pro" onSelect={vi.fn()} onProvidersChange={vi.fn()} onTabOrderChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))

    expect(screen.getByRole("menuitemradio", { name: "gemini-3.1-pro" })).toBeInTheDocument()
    expect(screen.queryByText("gemma4:31b-cloud")).not.toBeInTheDocument()
  })
})
