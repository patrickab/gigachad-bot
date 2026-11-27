import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { ModelsResponse } from "@/lib/domain/types"
import { PiTopBar, piModelFor, type PiCatalogProps, type PiSessionControls, type PiSessionHeader, type PiTopBarSlots } from "../PiTopBar"
import type { PiModel } from "../types"

vi.mock("@/lib/backend/api", () => ({ fetchPiCatalog: vi.fn(() => new Promise(() => {})) }))

const MODELS: ModelsResponse = {
  ollama: [],
  providers: [
    { label: "Pi", provider_id: "litellm_proxy", models: ["anthropic/claude-haiku-4-5", "anthropic/claude-opus-4-5"], source: "pi" },
    { label: "OpenRouter", provider_id: "openrouter", models: ["acme/model-x", "acme/gone"] },
  ],
  defaults: { default_model: "", small_model: "", vision_model: "", sandbox_model: "" },
  tab_order: ["Pi", "OpenRouter"],
  unavailable: ["openrouter/acme/gone"],
}
const MODEL_PROPS: PiCatalogProps = { models: MODELS, onProvidersChange: async () => {}, onDefaultsChange: async () => {}, placement: "below" }

const HAIKU: PiModel = { provider: "anthropic", id: "claude-haiku-4-5", name: "Claude Haiku 4.5", reasoning: true }
const OPUS: PiModel = { provider: "anthropic", id: "claude-opus-4-5", name: "Claude Opus 4.5", reasoning: true }
const GPT: PiModel = { provider: "openai", id: "gpt-5", reasoning: true }

function controls(overrides: Partial<PiSessionControls> = {}): PiSessionControls {
  return {
    model: OPUS,
    models: [HAIKU, OPUS, GPT],
    thinkingLevel: "medium",
    thinkingLevels: ["off", "minimal", "low", "medium", "high"],
    stats: null,
    onModel: vi.fn(),
    onThinking: vi.fn(),
    ...overrides,
  }
}

function header(overrides: Partial<PiSessionHeader> = {}): PiSessionHeader {
  return { tone: "ready", sessionName: undefined, controls: controls(), column: null, ...overrides }
}

/** The tab top bar's three places, in the chat's order. */
function topBarSlots(): PiTopBarSlots & { bar: HTMLElement } {
  const bar = document.createElement("div")
  const [thinking, model, usage] = [0, 1, 2].map(() => bar.appendChild(document.createElement("div")))
  document.body.append(bar)
  return { bar, thinking, model, usage }
}

describe("PiTopBar", () => {
  afterEach(() => {
    cleanup()
    document.body.innerHTML = ""
  })

  it("shows the session's model by its catalog entry and offers only the catalog models Pi offers and the gateway serves", () => {
    const offered = [OPUS, GPT, { provider: "openrouter", id: "acme/model-x" }, { provider: "openrouter", id: "acme/gone" }]
    const props = controls({ models: offered })
    render(<PiTopBar slots={topBarSlots()} header={header({ controls: props })} modelProps={MODEL_PROPS} />)
    fireEvent.click(screen.getByRole("button", { name: "Model: anthropic/claude-opus-4-5" }))
    expect(screen.getByRole("menuitemradio", { name: "anthropic/claude-opus-4-5" })).toHaveAttribute("aria-checked", "true")
    expect(screen.queryByRole("menuitemradio", { name: "anthropic/claude-haiku-4-5" })).toBeNull()
    expect(screen.queryByRole("menuitemradio", { name: "gpt-5" })).toBeNull()

    fireEvent.click(screen.getByRole("button", { name: "OpenRouter" }))
    expect(screen.queryByRole("menuitemradio", { name: "acme/gone" })).toBeNull()
    fireEvent.click(screen.getByRole("menuitemradio", { name: "acme/model-x" }))
    expect(props.onModel).toHaveBeenCalledWith({ provider: "openrouter", id: "acme/model-x" })
  })
})

describe("piModelFor", () => {
  const models: PiModel[] = [
    { provider: "anthropic", id: "claude-opus-5" },
    { provider: "gateway", id: "anthropic/claude-opus-5" },
    { provider: "gateway", id: "openrouter/acme/model-x" },
    { provider: "gateway", id: "google/gemini-3.5-flash" },
  ]
  it("prefers the session's own provider, else Pi's gateway provider", () => {
    expect(piModelFor(models, "litellm_proxy/anthropic/claude-opus-5")).toEqual({ provider: "anthropic", id: "claude-opus-5" })
    expect(piModelFor(models, "openrouter/acme/model-x")).toEqual({ provider: "gateway", id: "openrouter/acme/model-x" })
    expect(piModelFor(models, "gemini/gemini-3.5-flash")).toEqual({ provider: "gateway", id: "google/gemini-3.5-flash" })
    expect(piModelFor(models, "openrouter/acme/gone")).toBeNull()
  })
})
