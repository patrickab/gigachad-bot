import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentEditor } from "@/components/DocumentEditor"

const api = vi.hoisted(() => ({
  loadFileViewerText: vi.fn(),
  readFileVaultRendered: vi.fn(),
  writeDocument: vi.fn(),
  writeBinaryDocument: vi.fn(),
  storeDrawing: vi.fn(),
  fileViewerRawUrl: vi.fn(),
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status: number) {
      super(message)
      this.status = status
    }
  },
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
const collaboration = vi.hoisted(() => ({
  useCollaborativeCanvas: vi.fn(),
  replace: vi.fn(),
  retry: vi.fn(),
}))

vi.mock("@/hooks/useCollaborativeCanvas", () => collaboration)
vi.mock("@/components/CanvasEditor", () => ({
  CanvasEditor: ({ doc, onChange }: { doc: { strokes: Array<{ id: string; points: string[][] }> }; onChange: (next: unknown) => void }) => (
    <button type="button" data-testid="canvas" onClick={() => onChange({ ...doc, strokes: [] })}>
      {doc.strokes[0]?.points[0]?.[0] ?? "empty"}
    </button>
  ),
  emptyCanvasDoc: () => ({ version: 1, frames: [], strokes: [], attachments: [], texts: [] }),
  parseCanvasDoc: JSON.parse,
  serializeCanvasDoc: JSON.stringify,
}))

describe("DocumentEditor canvas collaboration", () => {
  beforeEach(() => {
    collaboration.replace.mockReset()
    collaboration.retry.mockReset()
    collaboration.useCollaborativeCanvas.mockReturnValue({
      document: { version: 1, frames: [], strokes: [{ id: "stroke-1", points: [["remote"]], color: "#000", width: 2 }], attachments: [], texts: [] },
      replace: collaboration.replace,
      retry: collaboration.retry,
      status: "ready",
    })
  })

  it("keeps vault canvases on the file-viewer load and persistOverride save path", async () => {
    api.loadFileViewerText.mockResolvedValueOnce(JSON.stringify({ version: 1, frames: [], strokes: [{ id: "s", points: [["vault"]], color: "#000", width: 2 }], attachments: [], texts: [] }))
    const persistOverride = vi.fn().mockResolvedValue(undefined)
    const { unmount } = render(<DocumentEditor path="notes/sketch.canvas" slug="" overlay persistOverride={persistOverride} onClose={vi.fn()} />)

    expect(await screen.findByTestId("canvas")).toHaveTextContent("vault")
    expect(collaboration.useCollaborativeCanvas).not.toHaveBeenCalledWith("notes/sketch.canvas")

    fireEvent.click(screen.getByTestId("canvas"))
    unmount() // leaving flushes the pending autosave
    await waitFor(() => expect(persistOverride).toHaveBeenCalledWith(expect.stringContaining("\"strokes\":[]")))
    expect(api.writeDocument).not.toHaveBeenCalled()
  })

  it("keeps a loaded canvas mounted when sync is transiently unavailable", async () => {
    collaboration.useCollaborativeCanvas.mockReturnValue({
      document: { version: 1, frames: [], strokes: [{ id: "stroke-1", points: [["remote"]], color: "#000", width: 2 }], attachments: [], texts: [] },
      replace: collaboration.replace,
      retry: collaboration.retry,
      status: "error",
    })

    render(<DocumentEditor path="project/project/document/notes.canvas" slug="project" onClose={vi.fn()} />)

    expect(await screen.findByTestId("canvas")).toHaveTextContent("remote")
    expect(screen.getByRole("status")).toHaveTextContent("Canvas sync interrupted")
    fireEvent.click(screen.getByText("Retry"))
    expect(collaboration.retry).toHaveBeenCalledOnce()
  })
})
