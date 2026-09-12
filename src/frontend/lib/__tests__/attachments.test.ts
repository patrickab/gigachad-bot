import { afterEach, describe, expect, it } from "vitest"

import { getApiBase, setApiBase } from "@/lib/config"
import { normalizeAttachment, normalizeMessageAttachments } from "@/lib/attachments"
import type { Attachment, Message } from "@/lib/types"

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

  it("rebases a vault attachment's fileviewer URL, preserving its query", () => {
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
      "https://gigachad-backend.tail8cc40f.ts.net/chat-histories/my-project/_uploads/chat-123/image.png",
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
