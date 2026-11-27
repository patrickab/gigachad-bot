import { describe, expect, it, vi } from "vitest"

import { applyReferenceReads, liveReferencePaths, normalizeAttachment, readReferences, withAttachments, type AttachReference } from "@/features/chat/attachments"
import { ApiError } from "@/lib/backend/api"
import type { Attachment, Message } from "@/lib/domain/types"

vi.mock("@/lib/backend/config", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  getApiBase: () => "https://gigachad-backend.tail8cc40f.ts.net/api",
}))

describe("normalizeAttachment", () => {
  it("rebases a stale persisted upload URL onto the current API origin, keeping its path", () => {
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
    const bare: Attachment = { name: "image.png", mime: "image/png", url: "", active: true }

    const fresh = normalizeAttachment(bare, "chat-123", "my-project")

    expect(fresh.url).toBe(
      "https://gigachad-backend.tail8cc40f.ts.net/api/assets/attachment/project/my-project/chat/chat-123/image.png",
    )
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
})
