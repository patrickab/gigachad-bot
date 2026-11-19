import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { PiSessionControlsBar, statsText, type PiSessionControls } from "../PiTopBar"
import type { PiModel } from "../types"

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
})

describe("statsText", () => {
  it("marks an unknown context use after a compaction and omits a missing context window", () => {
    expect(statsText({ cost: 1, contextUsage: { tokens: null, contextWindow: 200_000, percent: null } })?.text).toBe("? · $1.00")
    expect(statsText({ cost: 0 })?.text).toBe("$0.00")
    expect(statsText(null)).toBeNull()
  })
})
