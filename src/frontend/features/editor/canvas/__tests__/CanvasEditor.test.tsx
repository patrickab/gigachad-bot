/**
 * Canvas editor regression tests: undo/redo history.
 *
 * History used to be a stroke-only stack: deleting a frame, moving anything or
 * erasing was unrecoverable, and Ctrl+Z after one of those silently ate an
 * unrelated stroke. It now records only entities touched by a local gesture, so
 * undo/redo leave remote additions and later remote changes alone.
 *
 * The pre-change document must be captured when the mutation is dispatched, NOT
 * inside the setState updater — those run during the next render, after liveRef
 * already points at the post-change document, which would make undo a no-op.
 */
import { useState } from "react"
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { render, act, fireEvent, waitFor } from "@testing-library/react"
import { TabContext, type TabContextValue } from "@/contexts/TabContext"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import type { CanvasAttachment, CanvasDocument } from "@/lib/domain/canvasDocument"
import type { ChangeEvent } from "@/lib/backend/syncStream"

const noop = () => {}

/** Provides a tab whose frontmost state is `value`. */
function TabActiveProvider({ value, children }: { value: boolean; children: React.ReactNode }) {
  const tab: TabContextValue = {
    tab: { id: "t1", chatId: "c1", historyFile: null, title: null, config: DEFAULT_TAB_CONFIG, appMode: "canvas", projectSlug: null, workspaceItem: null },
    isActive: value,
    setModeLabel: noop,
    updateConfig: noop,
    setAppMode: noop,
    setWorkspaceItem: noop,
    setHistoryFile: noop,
    setTitle: noop,
  }
  return <TabContext.Provider value={tab}>{children}</TabContext.Provider>
}

const renders = vi.hoisted(() => ({ pdf: 0, markdown: 0 }))
vi.mock("@/features/editor/pdf/LazyAnnotatedPdfViewer", () => ({
  LazyAnnotatedPdfViewer: ({ initialPage = 1, onPageChange }: { initialPage?: number; onPageChange?: (page: number) => void }) => {
    renders.pdf++
    return <button type="button" data-testid="pdf-viewer" onClick={() => onPageChange?.(4)}>{initialPage}</button>
  },
}))
// Memoized like the real renderer, so a note re-renders its markdown only when its props change.
vi.mock("@/components/content/LaTeXMarkdown", async () => ({
  LaTeXMarkdown: (await import("react")).memo(function LaTeXMarkdown({ content }: { content: string }) {
    renders.markdown++
    return <div data-testid="document-markdown">{content}</div>
  }),
}))
vi.mock("@/components/content/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => <div data-testid="document-plot">{JSON.stringify(figure)}</div>,
}))
// The architecture window and the text editor have their own tests; here they only report what the canvas hands them.
vi.mock("@/features/editor/architecture/LazyArchitectureWindow", () => ({
  LazyArchitectureWindow: ({ path, onNavigate }: { path: string; onNavigate: (path: string) => void }) =>
    <button type="button" data-testid="architecture-window" onClick={() => onNavigate("graph/proj/backend/backend.c4")}>{path}</button>,
}))
vi.mock("@/features/editor/DocumentEditor", () => ({
  DocumentEditor: ({ path }: { path: string }) => <div data-testid="text-editor">{path}</div>,
}))
const api = vi.hoisted(() => ({
  canvasEndpoint: (key: string) => key,
  writeBinaryDocument: vi.fn(),
  listProjectDocuments: vi.fn(async () => [{ path: "project/proj/document/notes.canvas", name: "notes.canvas", mime: "application/json" }]),
  loadFileViewerText: vi.fn(async () => ""),
  listNotes: vi.fn(async () => [] as { path: string; name: string; mime: string }[]),
  writeDocument: vi.fn<typeof import("@/lib/backend/api").writeDocument>(async () => ({ path: "project/proj/document/notes.canvas", name: "notes.canvas", mime: "application/json" })),
  renameDocument: vi.fn(async (_slug: string, path: string, name: string) => ({ path, name: `${name}.md`, mime: "text/markdown", changed: [] as string[] })),
  removeDocument: vi.fn<typeof import("@/lib/backend/api").removeDocument>(async () => []),
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status: number) {
      super(message)
      this.status = status
    }
  },
}))
vi.mock("@/lib/backend/api", () => api)

const collaboration = vi.hoisted(() => {
  const replace = vi.fn()
  const retry = vi.fn()
  return {
    replace,
    retry,
    useCollaborativeCanvas: vi.fn<() => {
      document: CanvasDocument | null
      replace: typeof replace
      retry: typeof retry
      status: string
    }>(() => ({
      document: { version: 1, frames: [], strokes: [], attachments: [], texts: [] },
      replace,
      retry,
      status: "ready",
    })),
  }
})
vi.mock("@/features/editor/canvas/useCollaborativeCanvas", () => collaboration)
const changes = vi.hoisted(() => ({ listeners: new Set<(event: ChangeEvent) => void>() }))
vi.mock("@/lib/backend/syncStream", () => ({
  syncCursorReady: () => Promise.resolve(),
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    changes.listeners.add(listener)
    return () => { changes.listeners.delete(listener) }
  },
}))

import { CanvasEditor } from "@/features/editor/canvas/CanvasEditor"
import { FileCanvasEditor } from "@/features/editor/canvas/FileCanvasEditor"
import { emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc } from "@/lib/domain/canvasDocument"
import { watchDocumentLists } from "@/lib/backend/resources"
import { onFileChange, onFileMove, type FileMove } from "@/lib/backend/fileEvents"

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= RO as unknown as typeof ResizeObserver

function Harness({ seen, slug, active = true, initialDoc = emptyCanvasDoc() }: { seen: CanvasDocument[]; slug?: string; active?: boolean; initialDoc?: CanvasDocument }) {
  const [doc, setDoc] = useState<CanvasDocument>(initialDoc)
  seen.push(doc)
  return (
    <UndoDeleteProvider>
      <TabActiveProvider value={active}>
        <CanvasEditor doc={doc} onChange={setDoc} slug={slug} docPath="project/proj/document/host.canvas" />
      </TabActiveProvider>
    </UndoDeleteProvider>
  )
}

function CollaborativeHarness({ seen }: { seen: CanvasDocument[] }) {
  const [doc, setDoc] = useState<CanvasDocument>(emptyCanvasDoc())
  seen.push(doc)
  return (
    <UndoDeleteProvider>
    <TabActiveProvider value={true}>
      <CanvasEditor doc={doc} onChange={setDoc} docPath="project/proj/document/host.canvas" />
      <button
        type="button"
        onClick={() => setDoc((current) => ({
          ...current,
          frames: [...current.frames, { id: "remote-frame", kind: "page", x: 100, y: 100, width: 794 }],
        }))}
      >
        Remote add
      </button>
      <button
        type="button"
        onClick={() => setDoc((current) => ({
          ...current,
          frames: current.frames.map((frame) => frame.id === "remote-frame" ? frame : { ...frame, width: 999 }),
        }))}
      >
        Remote update
      </button>
    </TabActiveProvider>
    </UndoDeleteProvider>
  )
}

const toolbarButton = (c: HTMLElement, label: string) => c.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!
const latest = (seen: CanvasDocument[]) => seen[seen.length - 1]!

/** The attachment as a window of `kind`, failing the test when it is another kind. */
function ofKind<K extends CanvasAttachment["kind"]>(att: CanvasAttachment | undefined, kind: K): Extract<CanvasAttachment, { kind: K }> {
  if (att?.kind !== kind) throw new Error(`expected a ${kind} attachment`)
  return att as Extract<CanvasAttachment, { kind: K }>
}

function addPage(container: HTMLElement) {
  act(() => { toolbarButton(container, "Add to canvas").click() }) // open the + menu
  const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "Page")!
  act(() => { item.click() })
}

describe("canvas schema", () => {
  it("assigns UUIDs to legacy strokes while preserving existing IDs", () => {
    const doc = parseCanvasDoc(JSON.stringify({
      version: 1,
      frames: [],
      strokes: [
        { points: [[0, 0]], color: "#000", width: 2 },
        { id: "kept", points: [[1, 1]], color: "#000", width: 2 },
      ],
      attachments: [],
      texts: [],
    }))

    expect(doc.strokes[0]!.id).toMatch(/^[0-9a-f-]{36}$/)
    expect(doc.strokes[1]!.id).toBe("kept")
  })
})

describe("canvas history", () => {
  it("undoes a frame add — not just strokes", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    expect(latest(seen).frames).toHaveLength(0)
    expect(toolbarButton(container, "Undo")).toBeDisabled() // nothing to undo yet

    addPage(container)
    expect(latest(seen).frames).toHaveLength(1)
    expect(toolbarButton(container, "Undo")).not.toBeDisabled()

    // the ordering bug would leave the page in place here
    act(() => { toolbarButton(container, "Undo").click() })
    expect(latest(seen).frames).toHaveLength(0)

    act(() => { toolbarButton(container, "Redo").click() })
    expect(latest(seen).frames).toHaveLength(1)
  })

  it("does not undo or redo entities added or changed remotely after a local gesture", () => {
    const seen: CanvasDocument[] = []
    const { container, getByText } = render(<CollaborativeHarness seen={seen} />)

    addPage(container)
    act(() => { fireEvent.click(getByText("Remote add")) })
    act(() => { fireEvent.click(getByText("Remote update")) })

    act(() => { toolbarButton(container, "Undo").click() })
    expect(latest(seen).frames).toHaveLength(2)
    expect(latest(seen).frames).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "remote-frame" }),
      expect.objectContaining({ width: 999 }),
    ]))

    act(() => { toolbarButton(container, "Redo").click() })
    expect(latest(seen).frames).toHaveLength(2)
    expect(latest(seen).frames).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "remote-frame" }),
      expect.objectContaining({ width: 999 }),
    ]))
  })

  it("stacks and unwinds multiple mutations in order", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addPage(container)
    addPage(container)
    addPage(container)
    expect(latest(seen).frames).toHaveLength(3)

    for (const remaining of [2, 1, 0]) {
      act(() => { toolbarButton(container, "Undo").click() })
      expect(latest(seen).frames).toHaveLength(remaining)
    }
    expect(toolbarButton(container, "Undo")).toBeDisabled()
  })

  // Ctrl+Shift+Z arrives with e.key === "Z" (shift uppercases it), so a case-sensitive
  // match on "z" left it dead — and with it the only redo shortcut browsers agree on.
  it("redoes from the keyboard, Ctrl+Y or Ctrl+Shift+Z", () => {
    const press = (key: string, shiftKey = false) => act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key, ctrlKey: true, shiftKey, bubbles: true }))
    })
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addPage(container)
    press("z")
    expect(latest(seen).frames).toHaveLength(0)
    press("y")
    expect(latest(seen).frames).toHaveLength(1)

    press("z")
    expect(latest(seen).frames).toHaveLength(0)
    press("Z", true)
    expect(latest(seen).frames).toHaveLength(1)
  })

  it("drops the redo stack once a new mutation lands", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addPage(container)
    act(() => { toolbarButton(container, "Undo").click() })
    expect(toolbarButton(container, "Redo")).not.toBeDisabled()

    addPage(container)
    expect(toolbarButton(container, "Redo")).toBeDisabled()
  })
})

// Backgrounded canvas tabs stay mounted and must ignore keys meant for another tab.
describe("background tab isolation", () => {
  const press = (key: string) => act(() => {
    window.dispatchEvent(new KeyboardEvent("keydown", { key, ctrlKey: true, bubbles: true }))
  })

  it("ignores Ctrl+Z from a backgrounded canvas tab", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} active={false} />)

    addPage(container)
    expect(latest(seen).frames).toHaveLength(1)

    press("z")
    expect(latest(seen).frames).toHaveLength(1) // still there — the tab isn't active
  })

  it("ignores a paste meant for another tab", async () => {
    api.writeBinaryDocument.mockResolvedValue({ path: "project/proj/document/pasted.png", name: "pasted.png", mime: "image/png" })
    const file = new File(["fake"], "pasted.png", { type: "image/png" })
    const item = { type: "image/png", getAsFile: () => file }
    const dispatchPaste = () => act(() => {
      window.dispatchEvent(Object.assign(new Event("paste", { bubbles: true, cancelable: true }), { clipboardData: { items: [item] } }))
    })

    const seen: CanvasDocument[] = []
    render(<Harness seen={seen} slug="proj" active={false} />)
    dispatchPaste()
    expect(api.writeBinaryDocument).not.toHaveBeenCalled()

    render(<Harness seen={seen} slug="proj" active />)
    dispatchPaste()
    expect(api.writeBinaryDocument).toHaveBeenCalledTimes(1)
  })
})

describe("keyboard text entry", () => {
  it("starts one focused text note at the pointer and keeps the first edit undoable as one action", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)
    const surface = container.querySelector("[tabindex=\"0\"]") as HTMLDivElement

    act(() => {
      fireEvent(surface, new MouseEvent("pointermove", { bubbles: true, clientX: 100, clientY: 50 }))
      fireEvent.keyDown(surface, { key: "H" })
    })

    const note = container.querySelector("textarea") as HTMLTextAreaElement
    expect(latest(seen).texts).toMatchObject([{ x: 200, y: 100, text: "H" }])
    expect(note).toHaveFocus()

    act(() => { fireEvent.change(note, { target: { value: "Hi" } }) })
    act(() => { toolbarButton(container, "Undo").click() })
    expect(latest(seen).texts).toHaveLength(0)
  })

  it("leaves Space and Tab to the tab picker and tab switching", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)
    const surface = container.querySelector("[tabindex=\"0\"]") as HTMLDivElement
    let spaceKept = true
    act(() => {
      spaceKept = fireEvent.keyDown(surface, { key: " ", code: "Space" })
      fireEvent.keyDown(surface, { key: "Tab" })
    })
    expect(spaceKept).toBe(true)
    expect(container.querySelector("textarea")).toBeNull()
  })

  it("closes text notes with Enter while Ctrl+Enter keeps the native line break", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)
    const surface = container.querySelector("[tabindex=\"0\"]") as HTMLDivElement

    act(() => { fireEvent.keyDown(surface, { key: "H" }) })
    const note = container.querySelector("textarea") as HTMLTextAreaElement

    act(() => { fireEvent.keyDown(note, { key: "Enter" }) })
    expect(container.querySelector("textarea")).toBeNull()

    const preview = container.querySelector("[data-testid=\"document-markdown\"]") as HTMLElement
    act(() => { fireEvent.click(preview) })
    const reopened = container.querySelector("textarea") as HTMLTextAreaElement
    expect(reopened).toHaveFocus()
    expect(fireEvent.keyDown(reopened, { key: "Enter", ctrlKey: true })).toBe(true)
    expect(reopened).toHaveFocus()
    act(() => { fireEvent.change(reopened, { target: { value: "H\nI" } }) })
    expect(latest(seen).texts[0]!.text).toBe("H\nI")
  })
})

describe("Escape inside a canvas overlay", () => {
  it("blurs the focused text note first, then closes the overlay", async () => {
    const onClose = vi.fn()
    const { container } = render(
      <UndoDeleteProvider>
        <TabActiveProvider value>
          <FileCanvasEditor path="project/proj/document/overlay.canvas" slug="proj" onClose={onClose} persistOverride={async () => {}} />
        </TabActiveProvider>
      </UndoDeleteProvider>,
    )
    const surface = await waitFor(() => {
      const found = container.querySelector<HTMLDivElement>("[tabindex=\"0\"]")
      expect(found).not.toBeNull()
      return found!
    })
    act(() => { fireEvent.keyDown(surface, { key: "H" }) })
    const note = container.querySelector("textarea") as HTMLTextAreaElement
    expect(note).toHaveFocus()

    act(() => { fireEvent.keyDown(note, { key: "Escape" }) })
    expect(container.querySelector("textarea")).toBeNull()
    expect(onClose).not.toHaveBeenCalled()

    act(() => { fireEvent.keyDown(document.body, { key: "Escape" }) })
    expect(onClose).toHaveBeenCalledTimes(1)
  })
})

describe("nested canvas", () => {
  const addCanvas = (container: HTMLElement) => {
    act(() => { toolbarButton(container, "Add to canvas").click() })
    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "New canvas")!
    act(() => { item.click() })
  }

  it("routes edits into the nested document, not the parent", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addCanvas(container)
    addPage(container.querySelector("[data-canvas-attachment]") as HTMLElement)

    expect(latest(seen).frames).toHaveLength(0)
    expect(ofKind(latest(seen).attachments[0], "canvas").canvas!.frames).toHaveLength(1)
  })

  it("survives a save/load round-trip", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addCanvas(container)
    addPage(container.querySelector("[data-canvas-attachment]") as HTMLElement)

    const reloaded = parseCanvasDoc(serializeCanvasDoc(latest(seen)))
    expect(ofKind(reloaded.attachments[0], "canvas").canvas!.frames).toHaveLength(1)
  })
})

describe("nested canvas file", () => {
  const openCanvas = async (container: HTMLElement, name: string) => {
    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})
    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === name)!
    act(() => { item.click() })
  }

  it("opens a project canvas as a window holding only its path", async () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    await openCanvas(container, "notes.canvas")
    const [att] = latest(seen).attachments
    expect(att).toMatchObject({ kind: "canvas", path: "project/proj/document/notes.canvas" })
    expect(ofKind(att, "canvas").canvas).toBeUndefined()
    expect(collaboration.useCollaborativeCanvas).toHaveBeenCalledWith("project/proj/document/notes.canvas")
  })

  it("does not fall back to generic writes when collaboration loading fails", async () => {
    api.writeDocument.mockClear()
    collaboration.useCollaborativeCanvas.mockReturnValueOnce({
      document: null,
      replace: collaboration.replace,
      retry: collaboration.retry,
      status: "error",
    })
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    await openCanvas(container, "notes.canvas")
    expect(container.querySelector("[data-canvas-attachment]")!.textContent).toMatch(/Failed to load/i)
    expect(api.writeDocument).not.toHaveBeenCalled()
  })
})

describe("PDF attachments", () => {
  it("persists the selected page and restores it when reopening", () => {
    const doc: CanvasDocument = {
      ...emptyCanvasDoc(),
      attachments: [{ id: "pdf-1", kind: "pdf", path: "project/proj/document/reference.pdf", x: 0, y: 0, width: 500 }],
    }
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} initialDoc={doc} />)
    const viewer = container.querySelector("[data-testid=\"pdf-viewer\"]") as HTMLButtonElement

    expect(viewer).toHaveTextContent("1")
    act(() => { viewer.click() })
    expect(ofKind(latest(seen).attachments[0], "pdf").page).toBe(4)

    addPage(container)
    act(() => { toolbarButton(container, "Undo").click() })
    expect(ofKind(latest(seen).attachments[0], "pdf").page).toBe(4)

    const restored = parseCanvasDoc(serializeCanvasDoc(latest(seen)))
    const reopened = render(<Harness seen={[]} initialDoc={restored} />)
    expect(reopened.container.querySelector("[data-testid=\"pdf-viewer\"]")).toHaveTextContent("4")
  })
})

describe("panning", () => {
  // jsdom has no PointerEvent: without it the canvas never sees that a mouse pans
  beforeEach(() => vi.stubGlobal("PointerEvent", class extends MouseEvent {
    pointerType: string
    constructor(type: string, init: PointerEventInit = {}) { super(type, init); this.pointerType = init.pointerType ?? "" }
  }))
  afterEach(() => vi.unstubAllGlobals())

  it("moves attachment windows and notes without re-rendering their bodies", () => {
    const doc: CanvasDocument = {
      ...emptyCanvasDoc(),
      attachments: [{ id: "pdf-1", kind: "pdf", path: "project/proj/document/reference.pdf", x: 0, y: 0, width: 500 }],
      texts: [{ id: "note-1", x: 0, y: 600, width: 200, height: 40, text: "note", color: "#000", size: 16 }],
    }
    const { container } = render(<Harness seen={[]} initialDoc={doc} />)
    const surface = container.querySelector<HTMLElement>("[tabindex=\"0\"]")!
    const note = container.querySelector<HTMLElement>("[data-testid=\"document-markdown\"]")!.parentElement!.parentElement!
    const left = parseFloat(note.style.left)
    surface.setPointerCapture = noop
    fireEvent.pointerDown(surface, { button: 0, pointerType: "mouse", clientX: 0, clientY: 0 })
    renders.pdf = 0
    renders.markdown = 0
    for (let i = 1; i <= 20; i++) fireEvent.pointerMove(surface, { pointerType: "mouse", clientX: i * 5, clientY: i * 3 })
    fireEvent.pointerUp(surface, { pointerType: "mouse" })

    expect(parseFloat(note.style.left)).toBe(left + 100)
    expect(renders).toEqual({ pdf: 0, markdown: 0 })
  })
})

describe("document attachments", () => {
  it("lists a generated markdown artifact and adds it as a document attachment", async () => {
    api.listProjectDocuments.mockResolvedValueOnce([
      { path: "project/proj/document/summary.md", name: "summary.md", mime: "text/markdown" },
    ])
    api.loadFileViewerText.mockResolvedValueOnce("# Summary\n\nGenerated notes.")
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})

    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "summary.md")
    expect(item).toBeTruthy()
    act(() => { item!.click() })

    expect(latest(seen).attachments).toHaveLength(1)
    expect(latest(seen).attachments[0]!.kind).toBe("document")
    expect(latest(seen).attachments[0]!.path).toBe("project/proj/document/summary.md")

    await act(async () => {})
    expect(container.textContent).toContain("Generated notes.")
  })

  it("adds an Architecture window that remembers the file it moves to", async () => {
    const seen: CanvasDocument[] = []
    const { container, getByTestId } = render(<Harness seen={seen} slug="proj" />)

    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})
    act(() => { Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "Architecture")!.click() })
    expect(latest(seen).attachments[0]).toMatchObject({ kind: "architecture-graph", path: "graph/proj/" })

    act(() => { getByTestId("architecture-window").click() })
    expect(latest(seen).attachments[0]!.path).toBe("graph/proj/backend/backend.c4")
  })

  it("names a document's file on a header edit without moving it, but only retitles an architecture window", async () => {
    const seen: CanvasDocument[] = []
    const initialDoc: CanvasDocument = {
      version: 1, frames: [], strokes: [], texts: [],
      attachments: [
        { id: "mindmap", kind: "document", path: "project/proj/document/mindmap.md", x: 0, y: 0, width: 720, height: 480 },
        { id: "graph", kind: "architecture-graph", path: "graph/proj/model.c4", x: 0, y: 500, width: 720, height: 480 },
      ],
    }
    const { container, getByText } = render(<Harness seen={seen} slug="proj" initialDoc={initialDoc} />)

    fireEvent.doubleClick(getByText("mindmap.md"))
    let input = container.querySelector("input")!
    fireEvent.change(input, { target: { value: "Research map" } })
    fireEvent.blur(input)
    await waitFor(() => expect(latest(seen).attachments[0]).toMatchObject({
      path: "project/proj/document/mindmap.md", title: "Research map",
    }))
    expect(api.renameDocument).toHaveBeenLastCalledWith("proj", "project/proj/document/mindmap.md", "Research map")

    api.renameDocument.mockClear()
    fireEvent.doubleClick(getByText("Architecture"))
    input = container.querySelector("input")!
    fireEvent.change(input, { target: { value: "System design" } })
    fireEvent.blur(input)
    await waitFor(() => expect(latest(seen).attachments[1]).toMatchObject({
      path: "graph/proj/model.c4", title: "System design",
    }))
    expect(api.renameDocument).not.toHaveBeenCalled()
  })

  it("renames and deletes documents from the add menu, its list following the project's changes and attachments keeping the renamed file's key", async () => {
    const plot = { path: "project/proj/document/sandbox_plot-toolu_1.plot.json", name: "sandbox_plot-toolu_1.plot.json", mime: "application/json" }
    const diagram = { path: "project/proj/document/diagram-toolu_2.md", name: "diagram-toolu_2.md", mime: "text/markdown" }
    api.listProjectDocuments.mockResolvedValueOnce([plot, diagram]).mockResolvedValueOnce([{ ...plot, name: "Loss.plot.json" }, diagram])
    api.renameDocument.mockResolvedValueOnce({ ...plot, name: "Loss.plot.json", changed: [] })
    api.removeDocument.mockResolvedValueOnce([diagram.path])
    const stopWatching = watchDocumentLists()
    const seen: CanvasDocument[] = []
    const initialDoc: CanvasDocument = {
      version: 1, frames: [], strokes: [], texts: [],
      attachments: [
        { id: "plot", kind: "document", path: "project/proj/document/sandbox_plot-toolu_1.plot.json", x: 0, y: 0, width: 720, height: 480 },
        { id: "diagram", kind: "document", path: "project/proj/document/diagram-toolu_2.md", x: 0, y: 500, width: 720, height: 480 },
      ],
    }
    const { container, getByLabelText, queryByLabelText, getByText } = render(<Harness seen={seen} slug="proj" initialDoc={initialDoc} />)

    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})

    fireEvent.click(getByLabelText("Rename sandbox_plot-toolu_1.plot.json"))
    const input = getByLabelText("New name for sandbox_plot-toolu_1.plot.json") as HTMLInputElement
    fireEvent.change(input, { target: { value: "Loss" } })
    fireEvent.blur(input)
    await waitFor(() => expect(api.renameDocument).toHaveBeenLastCalledWith("proj", plot.path, "Loss"))
    // The rename's change to the project's catalog entry reads the list again.
    act(() => {
      for (const listener of changes.listeners) listener({ seq: 1, resource_kind: "document", resource_key: "chat/proj/project.json", version: null, operation: "write", device_id: "this" })
    })
    await waitFor(() => expect(container.textContent).toContain("Loss.plot.json"))
    expect(latest(seen).attachments[0]!.path).toBe(plot.path)

    // Deleting hides the row at once; the request and the attachment removal wait for the undo window.
    fireEvent.click(getByLabelText("Delete diagram-toolu_2.md"))
    expect(queryByLabelText("Delete diagram-toolu_2.md")).toBeNull()
    expect(api.removeDocument).not.toHaveBeenCalled()
    fireEvent.click(getByText("Undo"))
    expect(queryByLabelText("Delete diagram-toolu_2.md")).not.toBeNull()

    vi.useFakeTimers()
    try {
      fireEvent.click(getByLabelText("Delete diagram-toolu_2.md"))
      await act(async () => { vi.advanceTimersByTime(6000) })
    } finally {
      vi.useRealTimers()
      stopWatching()
    }
    expect(api.removeDocument).toHaveBeenCalledWith("proj", diagram.path)
    expect(latest(seen).attachments.map((a) => a.id)).toEqual(["plot"])
    expect(queryByLabelText("Delete diagram-toolu_2.md")).toBeNull()
  })

  it("announces a rename from the add menu that moved the file, so this device's tabs of it follow", async () => {
    const notes = { path: "/work/notes.md", name: "notes.md", mime: "text/markdown" }
    api.listProjectDocuments.mockResolvedValueOnce([notes])
    api.renameDocument.mockResolvedValueOnce({ path: "/work/ideas.md", name: "ideas.md", mime: "text/markdown", changed: ["/work/paper.tex"] })
    const moves: FileMove[] = []
    const changed: string[] = []
    const stopMoves = onFileMove((move) => moves.push(move))
    const stopChanges = onFileChange(() => true, (path) => changed.push(path))
    const { container, getByLabelText } = render(<Harness seen={[]} slug="proj" />)

    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})
    fireEvent.click(getByLabelText("Rename notes.md"))
    const input = getByLabelText("New name for notes.md")
    fireEvent.change(input, { target: { value: "ideas" } })
    fireEvent.blur(input)

    await waitFor(() => expect(moves).toEqual([{ from: notes.path, to: "/work/ideas.md" }]))
    expect(changed).toEqual(["/work/paper.tex"])
    stopMoves()
    stopChanges()
  })

  it("lists notes-scoped artifacts when the canvas has no project slug", async () => {
    api.listNotes.mockResolvedValueOnce([
      { path: "note/chart.plot.json", name: "chart.plot.json", mime: "application/json" },
    ])
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})

    expect(container.textContent).toContain("chart.plot.json")
    expect(api.listProjectDocuments).not.toHaveBeenCalledWith(undefined)
  })

  it("shows a readable failure state for malformed or invalid plot documents", async () => {
    api.listProjectDocuments.mockResolvedValueOnce([
      { path: "project/proj/document/broken.plot.json", name: "broken.plot.json", mime: "application/json" },
    ])
    api.loadFileViewerText.mockResolvedValueOnce("null")
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    act(() => { toolbarButton(container, "Add to canvas").click() })
    await act(async () => {})
    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "broken.plot.json")
    act(() => { item!.click() })
    await act(async () => {})

    expect(latest(seen).attachments[0]!.kind).toBe("document")
    expect(container.textContent).toContain("Malformed plot document")
  })
})
