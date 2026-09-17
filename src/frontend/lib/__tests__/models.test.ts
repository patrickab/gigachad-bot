import { describe, expect, it } from "vitest"
import { displayName, OMP_LITELLM_ID } from "@/lib/models"

describe("displayName", () => {
  it("strips a single provider prefix", () => {
    expect(displayName("gemini/gemini-3.1-pro")).toBe("gemini-3.1-pro")
    expect(displayName("ollama/gemma4:31b-cloud")).toBe("gemma4:31b-cloud")
  })

  it("keeps the OMP provider segment and drops only the proxy hop", () => {
    expect(displayName(`${OMP_LITELLM_ID}/anthropic/claude-opus-5`)).toBe("anthropic/claude-opus-5")
    expect(displayName(`${OMP_LITELLM_ID}/openai-codex/gpt-5.2`)).toBe("openai-codex/gpt-5.2")
  })

  it("leaves an unprefixed name alone", () => {
    expect(displayName("claude-opus-5")).toBe("claude-opus-5")
  })
})
