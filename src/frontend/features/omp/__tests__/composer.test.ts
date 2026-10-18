import { afterEach, describe, expect, it, vi } from "vitest"
import type { Attachment, ModelsResponse } from "@/lib/domain/types"
import { composerFor, ompModelOf, ompPrompt, ompSelectorModels, selectorModelId, type ComposerState } from "../composer"

const state = (sessionId: string | null): ComposerState => ({ type: "composer", sessionId, streaming: false, model: null, thinkingLevel: "auto", thinkingLevels: [] })

describe("composerFor", () => {
  it("matches the session a view shows", () => {
    expect(composerFor({ kind: "session", sessionId: "a" }, state("a"))).toEqual(state("a"))
    expect(composerFor({ kind: "session", sessionId: "a" }, state("b"))).toBeNull()
  })

  it("matches the new-session screen only to a composer without session", () => {
    expect(composerFor({ kind: "new", cwd: "/repo" }, state(null))).toEqual(state(null))
    expect(composerFor({ kind: "new", cwd: "/repo" }, state("a"))).toBeNull()
  })

  it("never offers a composer on the settings view", () => {
    expect(composerFor({ kind: "settings" }, state(null))).toBeNull()
  })
})

describe("omp selector ids", () => {
  it("round-trips models whose id carries slashes", () => {
    const model = { provider: "openrouter", modelId: "nvidia/nemotron-3:free" }
    expect(selectorModelId(model)).toBe("litellm_proxy/openrouter/nvidia/nemotron-3:free")
    expect(ompModelOf(selectorModelId(model))).toEqual(model)
  })

  it("rejects ids outside the omp rows", () => {
    expect(ompModelOf("openai/gpt-5.6")).toBeNull()
    expect(ompModelOf("litellm_proxy/anthropic")).toBeNull()
    expect(ompModelOf("litellm_proxy/anthropic/")).toBeNull()
  })

  it("lists only the providers picked from omp", () => {
    const models: ModelsResponse = {
      ollama: ["llama3"],
      providers: [
        { label: "OpenAI", litellm_id: "openai", models: ["gpt-5.6"] },
        { label: "OMP", litellm_id: "litellm_proxy", models: ["anthropic/claude-sonnet-5"], source: "omp" },
      ],
      defaults: { default_model: "", small_model: "", vision_model: "", memory_model: "", omp_model: "" },
      tab_order: [],
    }
    expect(ompSelectorModels(models)).toMatchObject({ ollama: [], providers: [{ label: "OMP" }] })
  })
})

describe("ompPrompt", () => {
  afterEach(() => vi.unstubAllGlobals())

  const image: Attachment = { name: "shot.png", mime: "image/png", url: "http://api/shot.png", active: false }
  const pdf: Attachment = { name: "spec.pdf", mime: "application/pdf", url: "http://api/spec.pdf", active: false }

  it("sends images as base64 and keeps the text as typed", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("png", { headers: { "content-type": "image/png" } })))
    const parse = vi.fn()

    expect(await ompPrompt("look", [image], parse)).toEqual({ text: "look", images: [{ data: btoa("png"), mimeType: "image/png" }] })
    expect(parse).not.toHaveBeenCalled()
  })

  it("puts parsed documents ahead of the prompt", async () => {
    const parse = vi.fn(async (documents: Attachment[]) => documents.map((document) => ({ ...document, parsedMd: "# Spec" })))

    const prompt = await ompPrompt("summarize", [pdf], parse)

    expect(prompt.images).toEqual([])
    expect(prompt.text).toMatch(/### spec\.pdf\n\n# Spec[\s\S]*summarize$/)
  })
})
