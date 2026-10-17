import { afterEach, describe, expect, it, vi } from "vitest"

import { getApiBase, setApiBase } from "@/lib/backend/config"
import { buildAttachedSend, composeUserMessage, normalizeAttachment, normalizeMessageAttachments, toolArtifactDocument, withAttachments } from "@/features/chat/attachments"
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
