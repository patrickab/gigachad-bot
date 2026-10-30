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
      { label: "Gemini", litellm_id: "gemini", models: ["gemini-3.5-flash", "gemini-3.1-pro"] },
      { label: "OMP", litellm_id: "litellm_proxy", models: ["anthropic/claude-sonnet-5"], source: "omp" },
    ],
    defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", omp_model: "" },
    tab_order: ["OMP"],
  }

  it("offers only the models the proxy serves, by selector string", () => {
    const served = [
      { provider: "custom", modelId: "gemini/gemini-3.5-flash" },
      { provider: "custom", modelId: "litellm_proxy/anthropic/claude-sonnet-5" },
      { provider: "custom", modelId: "ollama/qwen3" },
    ]

    expect(learnSelectorModels(models, served)).toEqual({
      ...models,
      ollama: ["ollama/qwen3"],
      providers: [
        { label: "Gemini", litellm_id: "gemini", models: ["gemini-3.5-flash"] },
        { label: "OMP", litellm_id: "litellm_proxy", models: ["anthropic/claude-sonnet-5"], source: "omp" },
      ],
    })
    expect(servedModelOf(served, "ollama/qwen3")).toEqual({ provider: "custom", modelId: "ollama/qwen3" })
    expect(servedModelOf(served, "gemini/gemini-3.1-pro")).toBeNull()
  })
})
