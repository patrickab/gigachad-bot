import { describe, expect, it } from "vitest"
import type { ModelsResponse } from "@/lib/domain/types"
import { displayName, gatewayModelId, PI_LITELLM_ID, selectorIdOf } from "@/lib/models"

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

describe("selectorIdOf", () => {
  const models: ModelsResponse = {
    ollama: ["ollama/gemma4"],
    providers: [
      { label: "Gemini", provider_id: "gemini", models: ["gemini-3.5-flash"] },
      { label: "Pi", provider_id: PI_LITELLM_ID, models: ["anthropic/claude-sonnet-5"], source: "pi" },
    ],
    defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", sandbox_model: "" },
    tab_order: [],
    unavailable: [],
  }

  it("names a gateway model by the catalog's selector id, else by its own id", () => {
    expect(selectorIdOf(models, "google/gemini-3.5-flash")).toBe("gemini/gemini-3.5-flash")
    expect(selectorIdOf(models, "anthropic/claude-sonnet-5")).toBe(`${PI_LITELLM_ID}/anthropic/claude-sonnet-5`)
    expect(selectorIdOf(models, "ollama/gemma4")).toBe("ollama/gemma4")
    expect(selectorIdOf(models, "openai/gone")).toBe("openai/gone")
  })
})
