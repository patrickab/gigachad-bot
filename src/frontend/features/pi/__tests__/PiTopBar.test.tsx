import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import type { ModelsResponse } from "@/lib/domain/types"
import { PiPanelBar, PiTopBar, PiUsage, contextTone, piGatewayId, piModelFor, type PiCatalogProps, type PiSessionControls, type PiSessionHeader, type PiTopBarSlots } from "../PiTopBar"
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

  it("puts thinking, model and usage where a chat's bar has its reasoning, model and context readout", () => {
    const slots = topBarSlots()
    const stats = { cost: 0.16, contextUsage: { tokens: 12_340, contextWindow: 1_000_000, percent: 1.2 } }
    render(<PiTopBar slots={slots} header={header({ sessionName: "Fix", controls: controls({ stats }) })} modelProps={MODEL_PROPS} />)
    expect(within(slots.thinking!).getByRole("button", { name: "Reasoning: medium" })).toBeInTheDocument()
    expect(within(slots.model!).getByRole("button", { name: "Model: anthropic/claude-opus-4-5" })).toBeInTheDocument()
    expect(slots.model).toHaveTextContent("Fix")
    expect(within(slots.usage!).getByLabelText("Cost 0.16$ (0.00$), context 12.3k tokens, 1% of 1000k")).toHaveTextContent("0.16$ (0.00$)·12.3k (1%)")
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

  it("waits for Pi to name its models before showing the selector", () => {
    render(<PiTopBar slots={topBarSlots()} header={header({ controls: controls({ models: [] }) })} modelProps={MODEL_PROPS} />)
    expect(screen.queryByRole("button", { name: /^Model:/ })).toBeNull()
    expect(screen.getByRole("button", { name: "Reasoning: medium" })).toBeInTheDocument()
  })

  it("opens Manage models from the selector", () => {
    render(<PiTopBar slots={topBarSlots()} header={header()} modelProps={MODEL_PROPS} />)
    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
    expect(screen.getByRole("button", { name: "Providers / Models" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Default Models" })).toBeInTheDocument()
  })

  it("sets a picked thinking level and hides the selector for a model without reasoning", () => {
    const props = controls()
    const { rerender } = render(<PiTopBar slots={topBarSlots()} header={header({ controls: props })} modelProps={MODEL_PROPS} />)
    fireEvent.click(screen.getByRole("button", { name: "Reasoning: medium" }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: "high" }))
    expect(props.onThinking).toHaveBeenCalledWith("high")
    rerender(<PiTopBar slots={topBarSlots()} header={header({ controls: controls({ thinkingLevel: "off", thinkingLevels: ["off"] }) })} modelProps={MODEL_PROPS} />)
    expect(screen.queryByRole("button", { name: /^Reasoning/ })).toBeNull()
  })

  it("shows what a new session starts on before it runs, without usage", () => {
    const slots = topBarSlots()
    render(<PiTopBar slots={slots} header={null} controls={controls()} modelProps={MODEL_PROPS} />)
    expect(within(slots.thinking!).getByRole("button", { name: "Reasoning: medium" })).toBeInTheDocument()
    expect(within(slots.model!).getByRole("button", { name: /^Model:/ })).toBeInTheDocument()
    expect(slots.usage).toBeEmptyDOMElement()
  })
})

describe("PiUsage", () => {
  afterEach(cleanup)

  it("marks both unknown right after a compaction", () => {
    render(<PiUsage stats={{ cost: 0, contextUsage: { tokens: null, contextWindow: 200_000, percent: null } }} />)
    expect(screen.getByText("? (?)", { exact: false })).toBeTruthy()
  })
})

describe("PiPanelBar", () => {
  afterEach(cleanup)

  it("orders a side panel's header like a chat's top bar: thinking, the chats' model dropdown, then the context readout and actions", () => {
    const stats = { cost: 0.01, contextUsage: { tokens: 60_000, contextWindow: 200_000, percent: 30 } }
    const { container } = render(<PiPanelBar header={header({ sessionName: "Notes", controls: controls({ stats }) })} modelProps={MODEL_PROPS} actions={<button type="button">New session</button>} />)
    const labels = [...container.querySelectorAll("button, [aria-label^='Cost']")].map((element) => element.getAttribute("aria-label") ?? element.textContent)
    expect(labels).toEqual(["Reasoning: medium", "Model: anthropic/claude-opus-4-5", "Cost 0.01$ (0.00$), context 60k tokens, 30% of 200k", "New session"])
    expect(container).toHaveTextContent("Notes")
    expect(container).toHaveTextContent("0.01$ (0.00$)")
  })

  it("offers the catalog models Pi offers, like a Pi tab", () => {
    const props = controls()
    render(<PiPanelBar header={header({ controls: props })} modelProps={MODEL_PROPS} />)
    fireEvent.click(screen.getByRole("button", { name: "Model: anthropic/claude-opus-4-5" }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: "anthropic/claude-haiku-4-5" }))
    expect(props.onModel).toHaveBeenCalledWith(HAIKU)
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
  it("names a gateway model by its upstream id", () => {
    expect(piGatewayId({ provider: "gateway", id: "openrouter/acme/model-x" })).toBe("openrouter/acme/model-x")
    expect(piGatewayId({ provider: "anthropic", id: "claude-opus-5" })).toBe("anthropic/claude-opus-5")
  })
})

describe("PiUsage telemetry", () => {
  afterEach(cleanup)

  it("shows working subsessions, cost, context and cache hit rate, the context coloured by its use", () => {
    const stats = { cost: 0.48, subCost: 1.2, subActive: 2, tokens: { input: 10, output: 5, cacheRead: 80, cacheWrite: 10 }, contextUsage: { tokens: 90_000, contextWindow: 200_000, percent: 45 } }
    render(<PiUsage stats={stats} />)
    const readout = screen.getByLabelText("2 subsessions working, Cost 0.48$ (1.20$), context 90k tokens, 45% of 200k, 80% cached")
    expect(readout).toHaveTextContent("2·0.48$ (1.20$)·90k (45%)·80% cached")
    expect(screen.getByText("90k (45%)")).toHaveClass("text-warning")
  })

  it("hides the working count when nothing works and grades context use", () => {
    render(<PiUsage stats={{ cost: 0, subActive: 0 }} />)
    expect(screen.getByLabelText("Cost 0.00$ (0.00$)")).toHaveTextContent(/^0.00\$ \(0.00\$\)$/)
    expect([10, 25, 45, 70, null].map(contextTone)).toEqual(["text-ink-faint", "text-ink", "text-warning", "text-danger", "text-ink-faint"])
  })
})
