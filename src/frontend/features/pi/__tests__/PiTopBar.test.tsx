import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { ModelsResponse } from "@/lib/domain/types"
import { PiPanelBar, PiSessionControlsBar, PiTopBar, PiUsageBar, piModelOfSelector, statsText, type PiCatalogProps, type PiSessionControls } from "../PiTopBar"
import type { PiModel } from "../types"

vi.mock("@/lib/backend/api", () => ({ fetchPiCatalog: vi.fn(() => new Promise(() => {})) }))

const MODELS: ModelsResponse = {
  ollama: [],
  providers: [
    { label: "Pi", provider_id: "litellm_proxy", models: ["anthropic/claude-haiku-4-5", "anthropic/claude-opus-4-5"], source: "pi" },
    { label: "OpenRouter", provider_id: "openrouter", models: ["acme/model-x", "acme/gone"] },
  ],
  defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", sandbox_model: "" },
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

describe("PiSessionControlsBar", () => {
  afterEach(cleanup)

  it("preselects the session's model (Pi's default for a new session) and switches on pick", () => {
    const props = controls()
    render(<PiSessionControlsBar controls={props} />)
    fireEvent.click(screen.getByRole("button", { name: "Model: Claude Opus 4.5" }))
    expect(screen.getByRole("menuitemradio", { name: "Claude Opus 4.5" })).toHaveAttribute("aria-checked", "true")
    expect(screen.getByRole("menuitemradio", { name: "Claude Haiku 4.5" })).toHaveAttribute("aria-checked", "false")
    fireEvent.click(screen.getByRole("menuitemradio", { name: "Claude Haiku 4.5" }))
    expect(props.onModel).toHaveBeenCalledWith(HAIKU)
  })

  it("shows the session's thinking level (medium for a new session) and sets a picked one", () => {
    const props = controls()
    render(<PiSessionControlsBar controls={props} />)
    fireEvent.click(screen.getByRole("button", { name: "Thinking: medium" }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: "high" }))
    expect(props.onThinking).toHaveBeenCalledWith("high")
  })

  it("hides the thinking selector for a model without reasoning", () => {
    render(<PiSessionControlsBar controls={controls({ thinkingLevel: "off", thinkingLevels: ["off"] })} />)
    expect(screen.queryByRole("button", { name: /Thinking/ })).toBeNull()
  })

  it("shows context use and cost", () => {
    render(<PiSessionControlsBar controls={controls({ stats: { cost: 0.4512, contextUsage: { tokens: 60_000, contextWindow: 200_000, percent: 30 } } })} />)
    expect(screen.getByLabelText("Context 60k of 200k tokens, cost $0.45")).toHaveTextContent("30% · $0.45")
  })

  it("lets only the model name give way in a narrow side panel, thinking, usage and actions keep their width", () => {
    render(<PiPanelBar header={{ tone: "ready", sessionName: "Notes", controls: controls({ stats: { cost: 0.01 } }) }} actions={<button type="button">New session</button>} />)
    const model = screen.getByRole("button", { name: "Model: Claude Opus 4.5" })
    expect(model).toHaveClass("min-w-0", "max-w-full", "overflow-hidden")
    expect(model.parentElement).toHaveClass("min-w-0")
    expect(model.querySelector("span")).toHaveClass("min-w-0", "truncate")
    expect(screen.getByRole("button", { name: "Thinking: medium" }).parentElement?.parentElement).toHaveClass("shrink-0")
    expect(screen.getByLabelText("Cost $0.01")).toHaveClass("shrink-0")
    expect(screen.getByRole("button", { name: "New session" }).parentElement).toHaveClass("shrink-0")
  })
})

describe("PiUsageBar", () => {
  afterEach(cleanup)

  it("shows context tokens and use on one line without cost, the top bar keeps only model and thinking", () => {
    const stats = { cost: 0.16, contextUsage: { tokens: 12_340, contextWindow: 1_000_000, percent: 1.2 } }
    const { container } = render(<><PiTopBar header={{ tone: "ready", sessionName: undefined, controls: controls({ stats }) }} modelProps={MODEL_PROPS} /><PiUsageBar stats={stats} /></>)
    expect(screen.getByLabelText("Context 12.3k tokens, 1% of 1000k")).toHaveTextContent("12.3k | 1%")
    expect(container.textContent).not.toContain("$")
  })

  it("marks both unknown right after a compaction", () => {
    render(<PiUsageBar stats={{ cost: 0, contextUsage: { tokens: null, contextWindow: 200_000, percent: null } }} />)
    expect(screen.getByText("? | ?")).toBeTruthy()
  })
})

describe("PiTopBar", () => {
  afterEach(cleanup)

  it("shows the session's model by its catalog entry and offers only the catalog models Pi offers and the gateway serves", () => {
    const offered = [OPUS, GPT, { provider: "openrouter", id: "acme/model-x" }, { provider: "openrouter", id: "acme/gone" }]
    const props = controls({ models: offered })
    render(<PiTopBar header={{ tone: "ready", sessionName: "Fix", controls: props }} modelProps={MODEL_PROPS} />)
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
    render(<PiTopBar header={{ tone: "ready", sessionName: undefined, controls: controls({ models: [] }) }} modelProps={MODEL_PROPS} />)
    expect(screen.queryByRole("button", { name: /^Model:/ })).toBeNull()
    expect(screen.getByRole("button", { name: "Thinking: medium" })).toBeInTheDocument()
  })

  it("opens Manage models from the selector", () => {
    render(<PiTopBar header={{ tone: "ready", sessionName: undefined, controls: controls() }} modelProps={MODEL_PROPS} />)
    fireEvent.click(screen.getByRole("button", { name: /^Model:/ }))
    fireEvent.click(screen.getByRole("button", { name: "Manage models" }))
    expect(screen.getByRole("button", { name: "Providers / Models" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Default Models" })).toBeInTheDocument()
  })

  it("keeps the thinking selector for the session's model", () => {
    const props = controls()
    render(<PiTopBar header={{ tone: "ready", sessionName: undefined, controls: props }} modelProps={MODEL_PROPS} />)
    fireEvent.click(screen.getByRole("button", { name: "Thinking: medium" }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: "high" }))
    expect(props.onThinking).toHaveBeenCalledWith("high")
  })
})

describe("piModelOfSelector", () => {
  it("splits a selector's gateway id into Pi's provider and model id", () => {
    expect(piModelOfSelector("litellm_proxy/anthropic/claude-opus-5")).toEqual({ provider: "anthropic", id: "claude-opus-5" })
    expect(piModelOfSelector("openrouter/acme/model-x")).toEqual({ provider: "openrouter", id: "acme/model-x" })
    expect(piModelOfSelector("gemini/gemini-3.5-flash")).toEqual({ provider: "google", id: "gemini-3.5-flash" })
    expect(piModelOfSelector("bare")).toBeNull()
  })
})

describe("statsText", () => {
  it("marks an unknown context use after a compaction and omits a missing context window", () => {
    expect(statsText({ cost: 1, contextUsage: { tokens: null, contextWindow: 200_000, percent: null } })?.text).toBe("? · $1.00")
    expect(statsText({ cost: 0 })?.text).toBe("$0.00")
    expect(statsText(null)).toBeNull()
  })
})
