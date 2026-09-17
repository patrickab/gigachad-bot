import { act, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentEditor } from "@/components/DocumentEditor"

const api = vi.hoisted(() => ({
  loadFileViewerText: vi.fn(),
  readFileVaultRendered: vi.fn(),
  writeDocument: vi.fn(),
  writeBinaryDocument: vi.fn(),
  storeDrawing: vi.fn(),
  fileViewerRawUrl: vi.fn(),
}))

const sync = vi.hoisted(() => {
  let listener: ((event: { seq: number; resource_kind: "document" | "asset" | "vault_root"; resource_key: string; version: number | null; device_id: string | null }) => void) | undefined
  return {
    emit(event: { seq: number; resource_kind: "document" | "asset" | "vault_root"; resource_key: string; version: number | null; device_id: string | null }) { listener?.(event) },
    subscribeToChanges: vi.fn((next) => { listener = next; return () => { listener = undefined } }),
  }
})

vi.mock("@/lib/api", () => api)
vi.mock("@/lib/syncStream", () => ({ subscribeToChanges: sync.subscribeToChanges }))
vi.mock("@/components/CanvasEditor", () => ({
  CanvasEditor: ({ doc }: { doc: { strokes: Array<{ points: string[][] }> } }) => <output data-testid="canvas">{doc.strokes[0]?.points[0]?.[0] ?? "empty"}</output>,
  emptyCanvasDoc: () => ({ version: 1, frames: [], strokes: [], attachments: [], texts: [] }),
  parseCanvasDoc: JSON.parse,
  serializeCanvasDoc: JSON.stringify,
}))

describe("DocumentEditor live canvas sync", () => {
  beforeEach(() => {
    api.loadFileViewerText.mockReset()
    sync.subscribeToChanges.mockClear()
  })

  it("reloads an open database-native path from its logical change key", async () => {
    const key = "project/project/document/notes.canvas"
    const absolutePath = key
    api.loadFileViewerText
      .mockResolvedValueOnce(JSON.stringify({ version: 1, frames: [], strokes: [], attachments: [], texts: [] }))
      .mockResolvedValueOnce(JSON.stringify({ version: 1, frames: [], strokes: [{ points: [["remote"]] }], attachments: [], texts: [] }))

    render(<DocumentEditor path={absolutePath} slug="project" onClose={vi.fn()} />)
    await screen.findByTestId("canvas")

    await act(async () => {
      sync.emit({ seq: 42, resource_kind: "document", resource_key: key, version: 2, device_id: "other-device" })
    })

    await waitFor(() => expect(screen.getByTestId("canvas")).toHaveTextContent("remote"))
  })
})
