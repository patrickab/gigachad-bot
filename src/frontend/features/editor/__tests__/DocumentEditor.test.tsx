import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DocumentEditor } from "@/features/editor/DocumentEditor"

const api = vi.hoisted(() => ({
  loadFileViewerText: vi.fn(),
  readFileRootRendered: vi.fn(),
  writeDocument: vi.fn(),
  writeBinaryDocument: vi.fn(),
  storeDrawing: vi.fn(),
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

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: sync.subscribeToChanges }))
const collaboration = vi.hoisted(() => ({
  useCollaborativeCanvas: vi.fn(),
  replace: vi.fn(),
  retry: vi.fn(),
}))

vi.mock("@/features/editor/canvas/useCollaborativeCanvas", () => collaboration)
vi.mock("@/features/editor/canvas/CanvasEditor", () => ({
  CanvasEditor: ({ doc, onChange }: { doc: { strokes: Array<{ id: string; points: string[][] }> }; onChange: (next: unknown) => void }) => (
    <button type="button" data-testid="canvas" onClick={() => onChange({ ...doc, strokes: [] })}>
      {doc.strokes[0]?.points[0]?.[0] ?? "empty"}
    </button>
  ),
}))
vi.mock("@/features/editor/AssistedConsoleEditor", () => ({
  AssistedConsoleEditor: ({ value, children }: { value: string; children: React.ReactNode }) => <div><textarea aria-label="Document text" value={value} readOnly />{children}</div>,
}))
vi.mock("@/components/content/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => <output data-testid="rendered-plot">{JSON.stringify(figure)}</output>,
}))
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <output data-testid="rendered-markdown">{content}</output>,
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

  it("keeps file root canvases on the file-viewer load and persistOverride save path", async () => {
    api.loadFileViewerText.mockResolvedValueOnce(JSON.stringify({ version: 1, frames: [], strokes: [{ id: "s", points: [["root"]], color: "#000", width: 2 }], attachments: [], texts: [] }))
    const persistOverride = vi.fn().mockResolvedValue(undefined)
    const { unmount } = render(<DocumentEditor path="notes/sketch.canvas" slug="" overlay persistOverride={persistOverride} onClose={vi.fn()} />)

    expect(await screen.findByTestId("canvas")).toHaveTextContent("root")
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

describe("DocumentEditor tool artifacts", () => {
  it("renders plot JSON by default and switches between render and text", async () => {
    const figure = { data: [{ type: "bar", y: [1, 2] }] }
    api.loadFileViewerText.mockResolvedValueOnce(JSON.stringify(figure))
    render(<DocumentEditor path="project/proj/document/sandbox_plot-call-1.plot.json" slug="proj" onClose={vi.fn()} />)

    expect(await screen.findByTestId("rendered-plot")).toHaveTextContent(JSON.stringify(figure))
    fireEvent.click(screen.getByRole("button", { name: "Text" }))
    expect(screen.getByRole("textbox", { name: "Document text" })).toHaveValue(JSON.stringify(figure))
    fireEvent.click(screen.getByRole("button", { name: "Render" }))
    expect(screen.getByTestId("rendered-plot")).toBeInTheDocument()
  })

  it("renders markmap markdown by default", async () => {
    const mindmap = "```markmap\n# Topic\n```"
    api.loadFileViewerText.mockResolvedValueOnce(mindmap)
    render(<DocumentEditor path="project/proj/document/mindmap-call-1.md" slug="proj" onClose={vi.fn()} />)

    expect(await screen.findByTestId("rendered-markdown")).toHaveTextContent("```markmap # Topic ```")
    expect(screen.getByRole("button", { name: "Text" })).toHaveAttribute("aria-pressed", "false")
  })
})
