import { describe, expect, it } from "vitest"
import type { ModelsResponse } from "@/lib/domain/types"
import { isFrameStateOf, learnSelectorModels, servedModelOf } from "../model"

describe("isFrameStateOf", () => {
  it("matches a session by id, home and the new-session draft by null", () => {
    expect(isFrameStateOf({ kind: "session", sessionId: "s1" }, "s1")).toBe(true)
    expect(isFrameStateOf({ kind: "session", sessionId: "s1" }, null)).toBe(false)
    expect(isFrameStateOf({ kind: "new" }, null)).toBe(true)
    expect(isFrameStateOf({ kind: "home" }, "s1")).toBe(false)
  })
})

describe("learnSelectorModels", () => {
  const models: ModelsResponse = {
    ollama: ["ollama/gemma4", "ollama/qwen3"],
    providers: [
      { label: "Gemini", provider_id: "gemini", models: ["gemini-3.5-flash", "gemini-3.1-pro"] },
      { label: "Pi", provider_id: "litellm_proxy", models: ["anthropic/claude-sonnet-5"], source: "pi" },
    ],
    defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", sandbox_model: "" },
    tab_order: ["Pi"],
    unavailable: [],
  }

  it("offers only the models LearnHarness serves, matched by gateway id", () => {
    const served = [
      { provider: "custom", modelId: "google/gemini-3.5-flash" },
      { provider: "custom", modelId: "anthropic/claude-sonnet-5" },
      { provider: "custom", modelId: "ollama/qwen3" },
    ]

    expect(learnSelectorModels(models, served)).toEqual({
      ...models,
      ollama: ["ollama/qwen3"],
      providers: [
        { label: "Gemini", provider_id: "gemini", models: ["gemini-3.5-flash"] },
        { label: "Pi", provider_id: "litellm_proxy", models: ["anthropic/claude-sonnet-5"], source: "pi" },
      ],
    })
    expect(servedModelOf(served, "ollama/qwen3")).toEqual({ provider: "custom", modelId: "ollama/qwen3" })
    expect(servedModelOf(served, "litellm_proxy/anthropic/claude-sonnet-5")).toEqual({ provider: "custom", modelId: "anthropic/claude-sonnet-5" })
    expect(servedModelOf(served, "gemini/gemini-3.1-pro")).toBeNull()
  })
})
