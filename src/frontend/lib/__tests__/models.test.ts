import { describe, expect, it } from "vitest"
import { displayName, gatewayModelId, PI_LITELLM_ID } from "@/lib/models"

describe("displayName", () => {
  it("strips a single provider prefix", () => {
    expect(displayName("gemini/gemini-3.1-pro")).toBe("gemini-3.1-pro")
    expect(displayName("ollama/gemma4:31b-cloud")).toBe("gemma4:31b-cloud")
  })

  it("keeps the Pi provider segment and drops only the proxy hop", () => {
    expect(displayName(`${PI_LITELLM_ID}/anthropic/claude-opus-5`)).toBe("anthropic/claude-opus-5")
    expect(displayName(`${PI_LITELLM_ID}/openai-codex/gpt-5.2`)).toBe("openai-codex/gpt-5.2")
  })

  it("leaves an unprefixed name alone", () => {
    expect(displayName("claude-opus-5")).toBe("claude-opus-5")
  })
})

describe("gatewayModelId", () => {
  it("drops the proxy hop of a Pi selector", () => {
    expect(gatewayModelId(`${PI_LITELLM_ID}/anthropic/claude-opus-5`)).toBe("anthropic/claude-opus-5")
  })

  it("renames gemini to the gateway's google", () => {
    expect(gatewayModelId("gemini/gemini-3.5-flash")).toBe("google/gemini-3.5-flash")
  })

  it("passes every other provider through, and a name without a provider", () => {
    expect(gatewayModelId("ollama/gemma4:31b-cloud")).toBe("ollama/gemma4:31b-cloud")
    expect(gatewayModelId("openrouter/vendor/model")).toBe("openrouter/vendor/model")
    expect(gatewayModelId("constructor/model")).toBe("constructor/model")
    expect(gatewayModelId("claude-opus-5")).toBe("claude-opus-5")
  })
})
