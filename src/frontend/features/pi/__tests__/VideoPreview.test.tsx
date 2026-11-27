import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { PiTranscript } from "../PiTranscript"
import { EMPTY_TRANSCRIPT, transcriptOfEntries, type SessionEntry } from "../transcript"
import type { CustomMessage, UserMessage } from "../types"

const fileLinks = { base: "/repo", open: () => {} }
const facts = { path: ".pi/pastes/20261119-120000-clip.mp4", durationSeconds: 3, width: 320, height: 240, fps: 25 }
const frame = (seconds: number) => ({ seconds, image: { data: `frame${seconds}`, mimeType: "image/jpeg" } })
const video = { ...facts, frames: [frame(0), frame(62.5)] }

function entry(id: string, parentId: string | null, fields: { type: string } & Record<string, unknown>): SessionEntry {
  return { id, parentId, timestamp: "2026-11-19T10:00:00.000Z", ...fields }
}

describe("video-preview card", () => {
  const prompt: UserMessage = { role: "user", content: "What does @.pi/pastes/20261119-120000-clip.mp4 show?", timestamp: 1 }

  it("shows each frame with its time and the metadata from the session entry's details", () => {
    const entries = [
      entry("a", null, { type: "message", message: prompt }),
      entry("b", "a", { type: "custom_message", customType: "video-preview", content: "Video preview: clip", display: true, details: { videos: [video] } }),
    ]
    const messages = transcriptOfEntries({ entries, leafId: "b" }).messages
    expect((messages[1] as CustomMessage).details).toEqual({ videos: [video] })

    render(<PiTranscript transcript={{ ...EMPTY_TRANSCRIPT, messages, entryIds: ["a", "b"] }} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: `${video.path} at 0:00.0` })).toHaveAttribute("src", "data:image/jpeg;base64,frame0")
    expect(screen.getByRole("img", { name: `${video.path} at 1:02.5` })).toBeInTheDocument()
    expect(screen.getByText("1:02.5")).toBeInTheDocument()
    expect(screen.getByText("20261119-120000-clip.mp4 · 3.0 s · 320×240 · 25 fps")).toBeInTheDocument()
    expect(screen.queryByText("Video preview: clip")).toBeNull()
  })

  it("shows the single frame grid older sessions stored", () => {
    const grid: CustomMessage = { role: "custom", customType: "video-preview", content: "Video preview: clip", display: true, details: { videos: [{ ...facts, columns: 4, rows: 3, image: { data: "grid", mimeType: "image/png" } }] }, timestamp: 2 }
    render(<PiTranscript transcript={{ ...EMPTY_TRANSCRIPT, messages: [prompt, grid], entryIds: ["a", "b"] }} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: `Frames of ${facts.path}` })).toHaveAttribute("src", "data:image/png;base64,grid")
  })

  it("falls back to the message text without usable details", () => {
    const bare: CustomMessage = { role: "custom", customType: "video-preview", content: "Video preview: clip", display: true, details: { videos: [{ path: "x" }] }, timestamp: 2 }
    render(<PiTranscript transcript={{ ...EMPTY_TRANSCRIPT, messages: [prompt, bare], entryIds: ["a", "b"] }} fileLinks={fileLinks} />)
    expect(screen.getByText("Video preview: clip")).toBeInTheDocument()
    expect(screen.queryByRole("img")).toBeNull()
  })
})
