import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PiTranscript } from "../PiTranscript"
import { EMPTY_TRANSCRIPT, messagesOfEntries, type SessionEntry } from "../transcript"
import type { CustomMessage, UserMessage } from "../types"

const fileLinks = { base: "/repo", open: () => {} }
const video = { path: ".pi/pastes/20261119-120000-clip.mp4", durationSeconds: 3, width: 320, height: 240, fps: 25, columns: 4, rows: 3, image: { data: "iVBORw0KGgo=", mimeType: "image/png" } }

function entry(id: string, parentId: string | null, fields: { type: string } & Record<string, unknown>): SessionEntry {
  return { id, parentId, timestamp: "2026-11-19T10:00:00.000Z", ...fields }
}

describe("video-preview card", () => {
  const prompt: UserMessage = { role: "user", content: "What does @.pi/pastes/20261119-120000-clip.mp4 show?", timestamp: 1 }

  it("shows the frame grid and metadata from the session entry's details", () => {
    const entries = [
      entry("a", null, { type: "message", message: prompt }),
      entry("b", "a", { type: "custom_message", customType: "video-preview", content: "Video preview: clip", display: true, details: { videos: [video] } }),
    ]
    const messages = messagesOfEntries({ entries, leafId: "b" })
    expect((messages[1] as CustomMessage).details).toEqual({ videos: [video] })

    render(<PiTranscript transcript={{ ...EMPTY_TRANSCRIPT, messages, entryIds: ["a", "b"] }} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: `Frames of ${video.path}` })).toHaveAttribute("src", "data:image/png;base64,iVBORw0KGgo=")
    expect(screen.getByText("20261119-120000-clip.mp4 · 3.0 s · 320×240 · 25 fps")).toBeInTheDocument()
    expect(screen.queryByText("Video preview: clip")).toBeNull()
  })

  it("falls back to the message text without usable details", () => {
    const bare: CustomMessage = { role: "custom", customType: "video-preview", content: "Video preview: clip", display: true, details: { videos: [{ path: "x" }] }, timestamp: 2 }
    render(<PiTranscript transcript={{ ...EMPTY_TRANSCRIPT, messages: [prompt, bare], entryIds: ["a", "b"] }} fileLinks={fileLinks} />)
    expect(screen.getByText("Video preview: clip")).toBeInTheDocument()
    expect(screen.queryByRole("img")).toBeNull()
  })
})
