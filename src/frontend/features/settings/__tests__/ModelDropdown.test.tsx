import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { ModelDropdown } from "@/features/settings/ModelDropdown"
import type { ModelsResponse, PiCatalog } from "@/lib/domain/types"

const fetchPiCatalog = vi.hoisted(() => vi.fn())
vi.mock("@/lib/backend/api", () => ({ fetchPiCatalog }))

const models: ModelsResponse = {
  ollama: ["ollama/gemma4:31b-cloud"],
  providers: [{ label: "Gemini", provider_id: "gemini", models: ["gemini-3.1-pro"] }],
  defaults: { default_model: "ollama/gemma4:31b-cloud", small_model: "ollama/gemma4:31b-cloud", vision_model: "ollama/gemma4:31b-cloud", sandbox_model: "ollama/gemma4:31b-cloud" },
  tab_order: [],
  unavailable: [],
}

const onlineCatalog: PiCatalog = {
  installed: true,
  online: true,
  gateway: "http://127.0.0.1:4000/v1",
  provider_id: "litellm_proxy",
  providers: [{ id: "anthropic", label: "Anthropic", models: [{ id: "anthropic/claude-opus-5", name: "Claude Opus 5", vision: true, context_window: 200000, thinking_levels: ["off", "low", "medium", "high"] }] }],
  error: null,
}

function openSettings() {
  fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
  fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
}

describe("ModelDropdown Pi source", () => {
  beforeEach(() => {
    fetchPiCatalog.mockReset()
  })

  it("adds a picked Pi model as a proxy-prefixed provider entry", async () => {
    fetchPiCatalog.mockResolvedValue(onlineCatalog)
    const onProvidersChange = vi.fn().mockResolvedValue(undefined)
    render(<ModelDropdown models={models} selectedModel="" onSelect={vi.fn()} onProvidersChange={onProvidersChange} onDefaultsChange={vi.fn()} />)

    openSettings()
    fireEvent.click(await screen.findByText("Anthropic"))
    fireEvent.click(await screen.findByText("claude-opus-5"))

    await waitFor(() => expect(onProvidersChange).toHaveBeenCalled())
    expect(onProvidersChange.mock.calls[0][0]).toEqual([
      { label: "Gemini", provider_id: "gemini", models: ["gemini-3.1-pro"] },
      { label: "Pi", provider_id: "litellm_proxy", source: "pi", models: ["anthropic/claude-opus-5"] },
    ])
  })

  it("removes an already-picked Pi model instead of duplicating it", async () => {
    fetchPiCatalog.mockResolvedValue(onlineCatalog)
    const onProvidersChange = vi.fn().mockResolvedValue(undefined)
    const withPick: ModelsResponse = {
      ...models,
      providers: [...models.providers, { label: "Pi", provider_id: "litellm_proxy", source: "pi", models: ["anthropic/claude-opus-5"] }],
    }
    render(<ModelDropdown models={withPick} selectedModel="" onSelect={vi.fn()} onProvidersChange={onProvidersChange} onDefaultsChange={vi.fn()} />)

    openSettings()
    fireEvent.click(await screen.findByText("1 of 1 added"))
    fireEvent.click(await screen.findByText("claude-opus-5"))

    await waitFor(() => expect(onProvidersChange).toHaveBeenCalled())
    expect(onProvidersChange.mock.calls[0][0][1].models).toEqual([])
  })

  it("only probes the gateway once the settings panel is opened", async () => {
    fetchPiCatalog.mockResolvedValue(onlineCatalog)
    render(<ModelDropdown models={models} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    expect(fetchPiCatalog).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
    await waitFor(() => expect(fetchPiCatalog).toHaveBeenCalledTimes(1))
  })

  it("points at run-pi-host.sh while the gateway is offline", async () => {
    fetchPiCatalog.mockResolvedValue({ ...onlineCatalog, online: false })
    render(<ModelDropdown models={models} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    openSettings()

    expect(await screen.findByText(/not answering on http:\/\/127\.0\.0\.1:4000\/v1\. Start it with run-pi-host\.sh/)).toBeInTheDocument()
  })

  it("lists providers in the persisted order, unlisted providers last", async () => {
    fetchPiCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    const withThree: ModelsResponse = {
      ...models,
      providers: [
        { label: "Gemini", provider_id: "gemini", models: ["gemini-3.1-pro"] },
        { label: "DeepSeek", provider_id: "deepseek", models: ["deepseek-v4"] },
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
    fetchPiCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    const unsorted: ModelsResponse = {
      ...models,
      ollama: ["ollama/zulu", "ollama/alpha"],
      providers: [{ label: "Gemini", provider_id: "gemini", models: ["zeta", "alpha"] }],
    }
    render(<ModelDropdown models={unsorted} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Ollama" }))
    expect(screen.getAllByRole("menuitemradio").map((row) => row.textContent)).toEqual(["alpha", "zulu"])

    fireEvent.click(screen.getByRole("button", { name: "Gemini" }))
    expect(screen.getAllByRole("menuitemradio").map((row) => row.textContent)).toEqual(["alpha", "zeta"])
  })

  it("opens on the provider tab holding the currently selected model, not Ollama", async () => {
    fetchPiCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    render(<ModelDropdown models={models} selectedModel="gemini/gemini-3.1-pro" onSelect={vi.fn()} onProvidersChange={vi.fn()} onTabOrderChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))

    expect(screen.getByRole("menuitemradio", { name: "gemini-3.1-pro" })).toBeInTheDocument()
    expect(screen.queryByText("gemma4:31b-cloud")).not.toBeInTheDocument()
  })

  it("narrows only the pick list with pickModels: Manage models still saves the full catalog", async () => {
    fetchPiCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    const onProvidersChange = vi.fn().mockResolvedValue(undefined)
    const ollamaOnly = (catalog: ModelsResponse): ModelsResponse => ({ ...catalog, providers: [] })
    render(<ModelDropdown models={models} pickModels={ollamaOnly} selectedModel="" onSelect={vi.fn()} onProvidersChange={onProvidersChange} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    expect(screen.queryByRole("button", { name: "Gemini" })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
    fireEvent.click(screen.getByRole("button", { name: "Expand Gemini" }))
    fireEvent.change(screen.getByRole("textbox", { name: "Model name" }), { target: { value: "gemini-4" } })
    fireEvent.click(screen.getByRole("button", { name: "Add model" }))

    await waitFor(() => expect(onProvidersChange).toHaveBeenCalledTimes(1))
    expect(onProvidersChange.mock.calls[0][0]).toEqual([{ label: "Gemini", provider_id: "gemini", models: ["gemini-3.1-pro", "gemini-4"] }])
  })

  it("leaves models the gateway does not list out of the pick list, Manage models still shows them", async () => {
    fetchPiCatalog.mockResolvedValue({ ...onlineCatalog, installed: false })
    const withUnlisted: ModelsResponse = {
      ...models,
      providers: [{ label: "Gemini", provider_id: "gemini", models: ["gemini-3.1-pro", "gemini-retired"] }],
      unavailable: ["gemini/gemini-retired"],
    }
    render(<ModelDropdown models={withUnlisted} selectedModel="" onSelect={vi.fn()} onProvidersChange={vi.fn()} onDefaultsChange={vi.fn()} />)

    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Gemini" }))
    expect(screen.getAllByRole("menuitemradio").map((row) => row.textContent)).toEqual(["gemini-3.1-pro"])
    fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
    fireEvent.click(screen.getByRole("button", { name: "Expand Gemini" }))
    expect(screen.getByText("gemini-retired")).toBeInTheDocument()
  })
})
