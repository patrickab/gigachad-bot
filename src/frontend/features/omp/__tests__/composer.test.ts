import { afterEach, describe, expect, it, vi } from "vitest"
import type { Attachment, ModelsResponse } from "@/lib/domain/types"
import { composerImageFile, isComposerOf, mergeRestoredText, ompModelOf, ompPrompt, ompSelectorModels, selectorModelId, shellHint } from "../composer"

describe("isComposerOf", () => {
  it("matches the session a view shows", () => {
    expect(isComposerOf({ kind: "session", sessionId: "a" }, "a")).toBe(true)
    expect(isComposerOf({ kind: "session", sessionId: "a" }, "b")).toBe(false)
  })

  it("matches the new-session screen only to a composer without session", () => {
    expect(isComposerOf({ kind: "new", cwd: "/repo" }, null)).toBe(true)
    expect(isComposerOf({ kind: "new", cwd: "/repo" }, "a")).toBe(false)
  })

  it("never offers a composer on the settings view", () => {
    expect(isComposerOf({ kind: "settings" }, null)).toBe(false)
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

describe("mergeRestoredText", () => {
  it("puts restored text before what the user typed meanwhile", () => {
    expect(mergeRestoredText("", { text: "queued", replace: null })).toBe("queued")
    expect(mergeRestoredText("typed", { text: "queued", replace: null })).toBe("queued\n\ntyped")
    expect(mergeRestoredText("typed", { text: "", replace: null })).toBe("typed")
  })

  it("replaces the block an earlier restore put in front", () => {
    const replace = { lead: "first", fallback: "second" }
    expect(mergeRestoredText("first\n\ntyped", { text: "first\n\nsecond", replace })).toBe("first\n\nsecond\n\ntyped")
  })

  it("prepends only the new part once the user edited the earlier block", () => {
    const replace = { lead: "first", fallback: "second" }
    expect(mergeRestoredText("edited", { text: "first\n\nsecond", replace })).toBe("second\n\nedited")
  })
})

describe("composerImageFile", () => {
  it("decodes the base64 image omp hands back", async () => {
    const file = composerImageFile({ data: btoa("png"), mimeType: "image/png" }, 0)
    expect(file).toMatchObject({ name: "restored-image-1.png", type: "image/png" })
    const content = await new Promise((resolve) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.readAsText(file) })
    expect(content).toBe("png")
  })
})

describe("shellHint", () => {
  it("tells a shell command from a local one", () => {
    expect(shellHint("  !ls", 0)).toBe("Shell · output sent to the model")
    expect(shellHint("!!ls", 0)).toBe("Shell · output stays local")
    expect(shellHint("ls !x", 0)).toBeNull()
  })

  it("treats a draft with attachments as a prompt, like omp-web", () => {
    expect(shellHint("!ls", 1)).toBeNull()
  })
})
