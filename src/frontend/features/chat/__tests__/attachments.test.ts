import { afterEach, describe, expect, it, vi } from "vitest"

import { getApiBase, setApiBase } from "@/lib/backend/config"
import { applyReferenceReads, buildAttachedSend, composeUserMessage, liveReferencePaths, normalizeAttachment, normalizeMessageAttachments, readReferences, toolArtifactDocument, withAttachments, type AttachReference } from "@/features/chat/attachments"
import { ApiError } from "@/lib/backend/api"
import type { Attachment, Message } from "@/lib/domain/types"

const originalApiBase = getApiBase()

afterEach(() => setApiBase(originalApiBase))

describe("normalizeAttachment", () => {
  it("rebases a stale persisted upload URL onto the current API origin, keeping its path", () => {
    setApiBase("https://gigachad-backend.tail8cc40f.ts.net/api")
    const stale: Attachment = {
      name: "image.png",
      mime: "image/png",
      url: "http://127.0.0.1:8001/chat-histories/my-project/_uploads/chat-123/image.png",
      active: true,
    }

    // Deliberately mismatched chat_id/slug: a branched or moved chat's stored
    // path is the only reliable locator, so it must win over any (chatId,
    // slug) reconstruction that would guess a different, nonexistent path.
    const fresh = normalizeAttachment(stale, "some-other-chat-id", "different-project")

    expect(fresh.url).toBe(
      "https://gigachad-backend.tail8cc40f.ts.net/chat-histories/my-project/_uploads/chat-123/image.png",
    )
  })

  it("rebases a file root attachment's fileviewer URL, preserving its query", () => {
    setApiBase("https://gigachad-backend.tail8cc40f.ts.net/api")
    const stale: Attachment = {
      name: "doc.pdf",
      mime: "application/pdf",
      url: "http://127.0.0.1:8001/api/fileviewer/raw?path=%2Fold%2Fpath.pdf",
      active: true,
      vaultPath: "/nextcloud/vault/doc.pdf",
    }

    const fresh = normalizeAttachment(stale, "chat-123", null)

    expect(fresh.url).toBe(
      "https://gigachad-backend.tail8cc40f.ts.net/api/fileviewer/raw?path=%2Fold%2Fpath.pdf",
    )
  })

  it("falls back to (chatId, slug) reconstruction only when there is no stored URL", () => {
    setApiBase("https://gigachad-backend.tail8cc40f.ts.net/api")
    const bare: Attachment = { name: "image.png", mime: "image/png", url: "", active: true }

    const fresh = normalizeAttachment(bare, "chat-123", "my-project")

    expect(fresh.url).toBe(
      "https://gigachad-backend.tail8cc40f.ts.net/api/assets/attachment/project/my-project/chat/chat-123/image.png",
    )
  })

  it("keeps a stored URL untouched when it isn't a parseable absolute URL and no chat id is known", () => {
    const stale: Attachment = { name: "image.png", mime: "image/png", url: "not a url", active: true }

    expect(normalizeAttachment(stale, null, "my-project").url).toBe("not a url")
  })
})

describe("normalizeMessageAttachments", () => {
  it("rebases every attachment URL on a loaded message onto the current origin", () => {
    setApiBase("https://gigachad-backend.tail8cc40f.ts.net/api")
    const msg: Message = {
      role: "user",
      content: "see attached",
      attachments: [
        { name: "image.png", mime: "image/png", url: "http://127.0.0.1:8001/chat-uploads/chat-123/image.png", active: true },
      ],
    }

    const normalized = normalizeMessageAttachments(msg, "chat-123", null)

    expect(normalized.attachments?.[0].url).toBe(
      "https://gigachad-backend.tail8cc40f.ts.net/chat-uploads/chat-123/image.png",
    )
  })
})

describe("toolArtifactDocument", () => {
  it("creates reusable Mermaid, mindmap, and Plotly documents", () => {
    expect(toolArtifactDocument({
      id: "diagram/a",
      name: "diagram",
      arguments: {},
      status: "done",
      detail: { mermaid: "flowchart LR\nA --> B" },
    })).toEqual({
      name: "diagram-diagram-a.md",
      content: "# Mermaid diagram\n\n```mermaid\nflowchart LR\nA --> B\n```\n",
    })
    expect(toolArtifactDocument({
      id: "mindmap-1",
      name: "mindmap",
      arguments: {},
      status: "done",
      detail: { mindmap: "```markmap\n# Topic\n```" },
    })).toEqual({ name: "mindmap-mindmap-1.md", content: "```markmap\n# Topic\n```" })
    expect(toolArtifactDocument({
      id: "plot-1",
      name: "sandbox_plot",
      arguments: {},
      status: "done",
      detail: { figure: { data: [{ type: "bar" }] } },
    })).toEqual({ name: "sandbox_plot-plot-1.plot.json", content: '{\n  "data": [\n    {\n      "type": "bar"\n    }\n  ]\n}' })
  })

})

describe("withAttachments", () => {
  it("replaces attachments and rebuilds hiddenContent from the active text ones", () => {
    const msg: Message = { role: "user", content: "q", hiddenContent: "stale" }
    const attachments: Attachment[] = [
      { name: "a.txt", mime: "text/plain", url: "", active: true, content: "alpha" },
      { name: "b.txt", mime: "text/plain", url: "", active: false, content: "beta" },
    ]

    const patched = withAttachments(msg, attachments)

    expect(patched.attachments).toBe(attachments)
    expect(patched.hiddenContent).toBe("**Attached files:**\n\n### a.txt\n\nalpha\n\n## END")
    expect(withAttachments(msg, []).hiddenContent).toBeUndefined()
  })
})

describe("composeUserMessage", () => {
  it("puts hidden content ahead of the prompt", () => {
    expect(composeUserMessage("hello", "HIDDEN")).toBe("HIDDEN\n\nhello")
  })

  it("falls back to a default prompt for attachment-only turns", () => {
    expect(composeUserMessage("  ", undefined)).toBe("Please review the attached document and provide a summary.")
    expect(composeUserMessage("", "HIDDEN")).toBe("HIDDEN\n\nPlease review the attached document and provide a summary.")
  })
})

describe("buildAttachedSend", () => {
  it("parses unparsed documents and assembles the request with prior and new image paths", async () => {
    const attachments: Attachment[] = [
      { name: "doc.pdf", mime: "application/pdf", url: "", active: true },
      { name: "new.png", mime: "image/png", url: "", active: true },
    ]
    const priorMessages: Message[] = [
      { role: "user", content: "earlier", attachments: [{ name: "old.png", mime: "image/png", url: "", active: true }] },
      { role: "assistant", content: "ok" },
    ]
    const parse = vi.fn(async () => [{ name: "doc.pdf", parsedMd: "parsed text" }])

    const { enriched, request } = await buildAttachedSend(
      {
        text: "",
        attachments,
        priorMessages,
        chatId: "chat-1",
        slug: "proj",
        baseParams: { model: "m", downscale_images: true },
      },
      parse,
    )

    expect(parse).toHaveBeenCalledWith("chat-1", ["doc.pdf"], "proj")
    expect(enriched[0].parsedMd).toBe("parsed text")
    expect(request).toEqual({
      model: "m",
      downscale_images: true,
      chat_id: "chat-1",
      user_msg: "**Attached files:**\n\n### doc.pdf\n\nparsed text\n\n## END\n\nPlease review the attached document and provide a summary.",
      img_paths: ["old.png", "new.png"],
      project_slug: "proj",
    })
  })
})

describe("reference refresh", () => {
  const reference = (vaultPath: string, content: string): Attachment => ({ name: "notes.md", mime: "text/markdown", url: "", active: true, vaultPath, content })
  const asked = (path: string): Message => withAttachments({ role: "user", content: "q" }, [reference(path, "old")])
  // What a send does: re-read the chat's live references, then patch them in.
  const refresh = async (messages: Message[], read: AttachReference) =>
    applyReferenceReads(messages, await readReferences(liveReferencePaths(messages.flatMap((m) => m.attachments ?? [])), read))

  it("puts the source's current text into every earlier message the model reads", async () => {
    const history: Message[] = [asked("/notes.md"), { role: "assistant", content: "a" }, asked("/notes.md")]
    const read = vi.fn(async (path: string): Promise<Attachment> => ({ ...reference(path, "new"), url: "fresh" }))

    const refreshed = await refresh(history, read)

    expect(read).toHaveBeenCalledTimes(1)
    expect(refreshed[0].hiddenContent).toContain("new")
    expect(refreshed[2].hiddenContent).not.toContain("old")
    expect(refreshed[1]).toBe(history[1])
  })

  it("follows a moved file to its new path", async () => {
    const refreshed = await refresh([asked("/old/notes.md")], async () => reference("/new/notes.md", "moved"))

    expect(refreshed[0].attachments?.[0].vaultPath).toBe("/new/notes.md")
  })

  it("keeps the last known text of a vanished source and marks it missing", async () => {
    const refreshed = await refresh([asked("/gone.md")], async () => { throw new ApiError("gone", 404) })

    expect(refreshed[0].attachments?.[0]).toMatchObject({ missing: true, content: "old" })
    expect(refreshed[0].hiddenContent).toContain("old")
  })

  it("does not mark a source missing when the backend merely failed", async () => {
    const refreshed = await refresh([asked("/notes.md")], async () => { throw new ApiError("down", 500) })

    expect(refreshed[0].attachments?.[0].missing).toBeUndefined()
  })

  it("leaves inactive and missing references alone on a send, but re-reading one by path revives it", async () => {
    const missing = withAttachments({ role: "user", content: "q" }, [{ ...reference("/gone.md", "old"), missing: true }])
    const inactive = withAttachments({ role: "user", content: "q" }, [{ ...reference("/off.md", "old"), active: false }])
    const read = vi.fn(async (path: string) => reference(path, "back"))

    const sent = await refresh([missing, inactive], read)
    const previewed = applyReferenceReads([missing, inactive], await readReferences(["/gone.md"], read))

    expect(sent).toEqual([missing, inactive])
    expect(previewed[0].attachments?.[0]).toMatchObject({ content: "back", missing: undefined })
    expect(read).toHaveBeenCalledTimes(1)
  })

  it("leaves a chat without references untouched", async () => {
    const history: Message[] = [{ role: "user", content: "q" }]

    expect(await refresh(history, vi.fn())).toBe(history)
  })
})
