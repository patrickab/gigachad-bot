/**
 * Canvas editor regression tests: undo/redo history, lasso selection geometry, and the
 * idle-loop guard.
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
import { describe, it, expect, vi } from "vitest"
import { render, act, fireEvent, waitFor } from "@testing-library/react"
import { TabActiveProvider } from "@/components/TabManager"
import type { CanvasDocument } from "@/components/CanvasEditor"

vi.mock("@/components/PdfViewer", () => ({
  PdfViewer: ({ initialPage = 1, onPageChange }: { initialPage?: number; onPageChange?: (page: number) => void }) =>
    <button type="button" data-testid="pdf-viewer" onClick={() => onPageChange?.(4)}>{initialPage}</button>,
}))
vi.mock("./PdfViewer", () => ({
  PdfViewer: ({ initialPage = 1, onPageChange }: { initialPage?: number; onPageChange?: (page: number) => void }) =>
    <button type="button" data-testid="pdf-viewer" onClick={() => onPageChange?.(4)}>{initialPage}</button>,
}))
vi.mock("@/components/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <div data-testid="document-markdown">{content}</div>,
}))
vi.mock("./LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <div data-testid="document-markdown">{content}</div>,
}))
vi.mock("@/components/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => <div data-testid="document-plot">{JSON.stringify(figure)}</div>,
}))
vi.mock("./PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => <div data-testid="document-plot">{JSON.stringify(figure)}</div>,
}))
const api = vi.hoisted(() => ({
  fileViewerRawUrl: (p: string) => `/raw/${p}`,
  writeBinaryDocument: vi.fn(),
  listProjectDocuments: vi.fn(async () => [{ path: "project/proj/document/notes.canvas", name: "notes.canvas", mime: "application/json" }]),
  loadFileViewerText: vi.fn(async () => ""),
  listArchitectureGraphs: vi.fn(async () => []),
  listNotes: vi.fn(async () => [] as { path: string; name: string; mime: string }[]),
  writeDocument: vi.fn(async (_slug: string, _name: string, _content: string) => ({ path: "project/proj/document/notes.canvas", name: "notes.canvas", mime: "application/json" })),
  renameDocument: vi.fn(async (_slug: string, _path: string, name: string) => ({ path: `project/proj/document/${name}.md`, name: `${name}.md`, mime: "text/markdown" })),
  removeDocument: vi.fn(async (_slug: string, _path: string) => undefined),
  renameArchitectureGraph: vi.fn(async (_name: string, name: string) => ({ path: `graph/${name}.architecture.yaml`, name: `${name}.architecture.yaml`, content: "", hasDraft: false, revision: "r" })),
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status: number) {
      super(message)
      this.status = status
    }
  },
}))
vi.mock("@/lib/api", () => api)

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
vi.mock("@/hooks/useCollaborativeCanvas", () => collaboration)

import {
  CanvasEditor, appendStrokePoint, emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc, polyBounds, resizeBox, scaleStroke, scalePoly, remapAcrossCanvases, inOwnAttachment,
  type SelBox,
} from "@/components/CanvasEditor"
import type { StrokeData } from "@/lib/drawing"

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
    <TabActiveProvider value={active}>
      <CanvasEditor doc={doc} onChange={setDoc} slug={slug} docPath="project/proj/document/host.canvas" />
    </TabActiveProvider>
  )
}

function CollaborativeHarness({ seen }: { seen: CanvasDocument[] }) {
  const [doc, setDoc] = useState<CanvasDocument>(emptyCanvasDoc())
  seen.push(doc)
  return (
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
  )
}

// toolbar order: [+, undo, redo, size, color, lasso, camera, text]
const toolbar = (c: HTMLElement) => Array.from(c.querySelectorAll("button"))
const latest = (seen: CanvasDocument[]) => seen[seen.length - 1]!

function addPage(container: HTMLElement) {
  act(() => { toolbar(container)[0]!.click() }) // open the + menu
  const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "Page")!
  act(() => { item.click() })
}

describe("lasso selection", () => {
  const stroke = (pts: number[][], width = 4): StrokeData => ({ id: crypto.randomUUID(), points: pts, color: "#000", width })
  const box = (x: number, y: number, w: number, h: number): SelBox => ({ x, y, w, h })

  it("bounds the lasso shape", () => {
    expect(polyBounds([[10, 10], [30, 50], [5, 20]])).toEqual({ x: 5, y: 10, w: 25, h: 40 })
    expect(polyBounds([])).toBeNull()
  })

  it("grows from the top-left when the handle is dragged", () => {
    const b = box(10, 20, 100, 100)
    expect(resizeBox(b, 50, 20)).toEqual({ x: 10, y: 20, w: 150, h: 120 })
    expect(resizeBox(b, -40, -60)).toEqual({ x: 10, y: 20, w: 60, h: 40 }) // origin never moves
  })

  it("clamps instead of collapsing or flipping", () => {
    const b = resizeBox(box(0, 0, 100, 100), -500, -500)
    expect(b).toEqual({ x: 0, y: 0, w: 24, h: 24 })
  })

  it("deforms the lasso outline exactly like the ink it holds", () => {
    const from = box(0, 0, 10, 10)
    const to = box(0, 0, 20, 10)
    const poly: [number, number][] = [[0, 0], [10, 0], [10, 10]]
    expect(scalePoly(poly, from, to)).toEqual([[0, 0], [20, 0], [20, 10]])
    // same mapping the strokes get, so the outline keeps hugging them
    expect(scaleStroke(stroke([[10, 10]]), [[10, 10]], from, to).points[0]!.slice(0, 2)).toEqual([20, 10])
  })

  it("scales geometry but never the pen width", () => {
    const s = stroke([[0, 0], [10, 10]], 7)
    const scaled = scaleStroke(s, s.points, box(0, 0, 10, 10), box(0, 0, 30, 5))
    expect(scaled.points).toEqual([[0, 0, 0.5], [30, 5, 0.5]]) // 3x wider, squeezed to half
    expect(scaled.width).toBe(7)
    expect(scaled.color).toBe("#000")
  })

  it("keeps pressure per point through a scale", () => {
    const s = stroke([[0, 0, 0.2], [10, 10, 0.9]])
    const scaled = scaleStroke(s, s.points, box(0, 0, 10, 10), box(5, 5, 20, 20))
    expect(scaled.points).toEqual([[5, 5, 0.2], [25, 25, 0.9]])
  })
})

describe("stroke sampling", () => {
  it("keeps the final pen sample while ignoring a duplicate browser sample", () => {
    const points = [[10, 20, 0.5]]

    appendStrokePoint(points, [30, 40, 0.7])
    appendStrokePoint(points, [30, 40, 0.7])

    expect(points).toEqual([[10, 20, 0.5], [30, 40, 0.7]])
  })
})
describe("stroke pointer ownership", () => {
  it("ignores a different pointer ending an active stroke", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)
    const surface = container.querySelectorAll("svg.absolute.inset-0.w-full.h-full")[1] as SVGSVGElement
    surface.setPointerCapture = vi.fn()

    const pointer = (type: string, pointerId: number, clientX: number, clientY: number) => {
      const event = new MouseEvent(type, { bubbles: true, button: 0, clientX, clientY })
      Object.defineProperty(event, "pointerId", { value: pointerId })
      return event
    }
    act(() => {
      fireEvent(surface, pointer("pointerdown", 1, 10, 10))
      fireEvent(surface, pointer("pointerup", 2, 900, 1000))
    })

    expect(latest(seen).strokes).toHaveLength(0)

    act(() => {
      fireEvent(surface, pointer("pointerup", 1, 20, 20))
    })

    expect(latest(seen).strokes).toHaveLength(1)
    expect(latest(seen).strokes[0]!.points.at(-1)?.slice(0, 2)).toEqual([40, 40])
  })
})

describe("stroke capture loss", () => {
  it("does not append a synthetic origin point after pointer capture is lost", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)
    const surface = container.querySelectorAll("svg.absolute.inset-0.w-full.h-full")[1] as SVGSVGElement
    surface.setPointerCapture = vi.fn()
    const pointer = (type: string, clientX: number, clientY: number) => {
      const event = new MouseEvent(type, { bubbles: true, button: 0, clientX, clientY })
      Object.defineProperty(event, "pointerId", { value: 1 })
      return event
    }

    act(() => { fireEvent(surface, pointer("pointerdown", 10, 10)) })
    act(() => { fireEvent(surface, pointer("pointermove", 20, 20)) })
    act(() => { fireEvent(surface, pointer("lostpointercapture", 0, 0)) })

    expect(latest(seen).strokes).toHaveLength(1)
    expect(latest(seen).strokes[0]!.points.at(-1)?.slice(0, 2)).toEqual([40, 40])
  })
})

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
    expect(toolbar(container)[1]!).toBeDisabled() // nothing to undo yet

    addPage(container)
    expect(latest(seen).frames).toHaveLength(1)
    expect(toolbar(container)[1]!).not.toBeDisabled()

    // the ordering bug would leave the page in place here
    act(() => { toolbar(container)[1]!.click() })
    expect(latest(seen).frames).toHaveLength(0)

    act(() => { toolbar(container)[2]!.click() }) // redo
    expect(latest(seen).frames).toHaveLength(1)
  })

  it("does not undo or redo entities added or changed remotely after a local gesture", () => {
    const seen: CanvasDocument[] = []
    const { container, getByText } = render(<CollaborativeHarness seen={seen} />)

    addPage(container)
    act(() => { fireEvent.click(getByText("Remote add")) })
    act(() => { fireEvent.click(getByText("Remote update")) })

    act(() => { toolbar(container)[1]!.click() })
    expect(latest(seen).frames).toHaveLength(2)
    expect(latest(seen).frames).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "remote-frame" }),
      expect.objectContaining({ width: 999 }),
    ]))

    act(() => { toolbar(container)[2]!.click() })
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
      act(() => { toolbar(container)[1]!.click() })
      expect(latest(seen).frames).toHaveLength(remaining)
    }
    expect(toolbar(container)[1]!).toBeDisabled()
  })

  // The viewport save callback used to depend on `doc`, so every document change re-armed
  // its timer, which wrote a new doc, which re-armed the timer: a 2Hz mutate/re-render
  // loop for as long as a canvas was open — and it starved the 1s autosave debounce.
  it("stops touching the document once the view settles", () => {
    vi.useFakeTimers()
    try {
      const seen: CanvasDocument[] = []
      render(<Harness seen={seen} />)
      act(() => { vi.advanceTimersByTime(600) }) // viewport persists once
      const settled = seen.length
      // each flush lets one more loop iteration through, so step rather than jump
      for (let i = 0; i < 5; i++) act(() => { vi.advanceTimersByTime(1000) })
      expect(seen.length).toBe(settled)
    } finally {
      vi.useRealTimers()
    }
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
    act(() => { toolbar(container)[1]!.click() })
    expect(toolbar(container)[2]!).not.toBeDisabled()

    addPage(container)
    expect(toolbar(container)[2]!).toBeDisabled()
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
    act(() => { toolbar(container)[1]!.click() })
    expect(latest(seen).texts).toHaveLength(0)
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

  it("renders Markdown after leaving a text note", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)
    const surface = container.querySelector("[tabindex=\"0\"]") as HTMLDivElement

    act(() => { fireEvent.keyDown(surface, { key: "H" }) })
    const note = container.querySelector("textarea") as HTMLTextAreaElement
    act(() => { fireEvent.change(note, { target: { value: "**bold**" } }) })
    act(() => { fireEvent.keyDown(note, { key: "Enter" }) })

    expect(container.querySelector("[data-testid=\"document-markdown\"]")).toHaveTextContent("**bold**")
  })
})

describe("nested canvas", () => {
  const addCanvas = (container: HTMLElement) => {
    act(() => { toolbar(container)[0]!.click() })
    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "New canvas")!
    act(() => { item.click() })
  }

  it("adds a canvas window carrying its own document", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addCanvas(container)
    const [att] = latest(seen).attachments
    expect(att!.kind).toBe("canvas")
    expect(att!.canvas).toEqual(emptyCanvasDoc())
  })

  it("routes edits into the nested document, not the parent", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addCanvas(container)
    addPage(container.querySelector("[data-canvas-attachment]") as HTMLElement)

    expect(latest(seen).frames).toHaveLength(0)
    expect(latest(seen).attachments[0]!.canvas!.frames).toHaveLength(1)
  })

  it("survives a save/load round-trip", () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    addCanvas(container)
    addPage(container.querySelector("[data-canvas-attachment]") as HTMLElement)

    const reloaded = parseCanvasDoc(serializeCanvasDoc(latest(seen)))
    expect(reloaded.attachments[0]!.canvas!.frames).toHaveLength(1)
  })
})

describe("nested canvas file", () => {
  const openCanvas = async (container: HTMLElement, name: string) => {
    act(() => { toolbar(container)[0]!.click() })
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
    expect(att!.canvas).toBeUndefined()
    expect(collaboration.useCollaborativeCanvas).toHaveBeenCalledWith("project/proj/document/notes.canvas")
  })

  it("keeps a loaded nested canvas mounted through a transient sync error", async () => {
    collaboration.replace.mockClear()
    collaboration.useCollaborativeCanvas.mockReturnValueOnce({
      document: emptyCanvasDoc(),
      replace: collaboration.replace,
      retry: collaboration.retry,
      status: "error",
    })
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    await openCanvas(container, "notes.canvas")
    addPage(container.querySelector("[data-canvas-attachment]") as HTMLElement)

    expect(collaboration.replace).toHaveBeenCalledWith(expect.objectContaining({
      frames: [expect.objectContaining({ id: expect.stringMatching(/^[0-9a-f-]{36}$/) })],
    }))
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
    expect(latest(seen).attachments[0]!.page).toBe(4)

    addPage(container)
    act(() => { toolbar(container)[1]!.click() })
    expect(latest(seen).attachments[0]!.page).toBe(4)

    const restored = parseCanvasDoc(serializeCanvasDoc(latest(seen)))
    const reopened = render(<Harness seen={[]} initialDoc={restored} />)
    expect(reopened.container.querySelector("[data-testid=\"pdf-viewer\"]")).toHaveTextContent("4")
  })

  it("lets screenshot drags pass through a PDF attachment", () => {
    const doc: CanvasDocument = {
      ...emptyCanvasDoc(),
      attachments: [{ id: "pdf-1", kind: "pdf", path: "project/proj/document/reference.pdf", x: 0, y: 0, width: 500 }],
    }
    const { container } = render(<Harness seen={[]} initialDoc={doc} />)

    act(() => { toolbar(container)[6]!.click() })

    expect(container.querySelector("[data-canvas-attachment]")?.parentElement).toHaveClass("pointer-events-none")
  })

  it("shows the screenshot selection over a PDF attachment", async () => {
    const doc: CanvasDocument = {
      ...emptyCanvasDoc(),
      attachments: [{ id: "pdf-1", kind: "pdf", path: "project/proj/document/reference.pdf", x: 0, y: 0, width: 500 }],
    }
    const { container } = render(<Harness seen={[]} initialDoc={doc} />)
    act(() => { toolbar(container)[6]!.click() })
    const surface = container.querySelectorAll("svg.absolute.inset-0.w-full.h-full")[1] as SVGSVGElement
    surface.setPointerCapture = vi.fn()

    act(() => {
      fireEvent(surface, new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: 10, clientY: 10 }))
      fireEvent(surface, new MouseEvent("pointermove", { bubbles: true, clientX: 50, clientY: 40 }))
    })

    await waitFor(() => expect(container.querySelector("[data-testid=\"screenshot-selection\"]")).toBeTruthy())
    const attachment = container.querySelector("[data-canvas-attachment]")!.parentElement!
    const selection = container.querySelector("[data-testid=\"screenshot-selection\"]")!
    expect(attachment.compareDocumentPosition(selection) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0)
  })
})

describe("project assets", () => {
  it("loads PDFs and images from the canvas project instead of its parent surface", async () => {
    api.listProjectDocuments.mockResolvedValueOnce([
      { path: "project/proj/document/reference.pdf", name: "reference.pdf", mime: "application/pdf" },
      { path: "project/proj/document/diagram.png", name: "diagram.png", mime: "image/png" },
    ])
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    act(() => { toolbar(container)[0]!.click() })
    await act(async () => {})

    expect(container.textContent).toContain("reference.pdf")
    expect(container.textContent).toContain("diagram.png")
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

    act(() => { toolbar(container)[0]!.click() })
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

  it("renames the backing document for document and Architecture Graph header edits", async () => {
    const seen: CanvasDocument[] = []
    const initialDoc: CanvasDocument = {
      version: 1, frames: [], strokes: [], texts: [],
      attachments: [
        { id: "mindmap", kind: "document", path: "project/proj/document/mindmap.md", x: 0, y: 0, width: 720, height: 480 },
        { id: "graph", kind: "architecture-graph", path: "graph/system.architecture.yaml", x: 0, y: 500, width: 720, height: 480 },
      ],
    }
    const { container, getByText } = render(<Harness seen={seen} slug="proj" initialDoc={initialDoc} />)

    fireEvent.doubleClick(getByText("mindmap.md"))
    let input = container.querySelector("input")!
    fireEvent.change(input, { target: { value: "Research map" } })
    fireEvent.blur(input)
    await waitFor(() => expect(latest(seen).attachments[0]).toMatchObject({
      path: "project/proj/document/Research map.md", title: "Research map",
    }))

    fireEvent.doubleClick(getByText("system.architecture.yaml"))
    input = container.querySelector("input")!
    fireEvent.change(input, { target: { value: "System design" } })
    fireEvent.blur(input)
    await waitFor(() => expect(latest(seen).attachments[1]).toMatchObject({
      path: "graph/System design.architecture.yaml", title: "System design",
    }))
  })

  it("renames and deletes documents from the add menu, keeping canvas attachments in step", async () => {
    api.listProjectDocuments.mockResolvedValueOnce([
      { path: "project/proj/document/sandbox_plot-toolu_1.plot.json", name: "sandbox_plot-toolu_1.plot.json", mime: "application/json" },
      { path: "project/proj/document/diagram-toolu_2.md", name: "diagram-toolu_2.md", mime: "text/markdown" },
    ])
    api.renameDocument.mockResolvedValueOnce({ path: "project/proj/document/Loss.plot.json", name: "Loss.plot.json", mime: "application/json" })
    const seen: CanvasDocument[] = []
    const initialDoc: CanvasDocument = {
      version: 1, frames: [], strokes: [], texts: [],
      attachments: [
        { id: "plot", kind: "document", path: "project/proj/document/sandbox_plot-toolu_1.plot.json", x: 0, y: 0, width: 720, height: 480 },
        { id: "diagram", kind: "document", path: "project/proj/document/diagram-toolu_2.md", x: 0, y: 500, width: 720, height: 480 },
      ],
    }
    const { container, getByLabelText } = render(<Harness seen={seen} slug="proj" initialDoc={initialDoc} />)

    act(() => { toolbar(container)[0]!.click() })
    await act(async () => {})

    fireEvent.click(getByLabelText("Rename sandbox_plot-toolu_1.plot.json"))
    const input = getByLabelText("New name for sandbox_plot-toolu_1.plot.json") as HTMLInputElement
    fireEvent.change(input, { target: { value: "Loss" } })
    fireEvent.blur(input)
    await waitFor(() => expect(container.textContent).toContain("Loss.plot.json"))
    expect(api.renameDocument).toHaveBeenLastCalledWith("proj", "project/proj/document/sandbox_plot-toolu_1.plot.json", "Loss")
    expect(latest(seen).attachments[0]!.path).toBe("project/proj/document/Loss.plot.json")

    fireEvent.click(getByLabelText("Delete diagram-toolu_2.md"))
    await waitFor(() => expect(latest(seen).attachments.map((a) => a.id)).toEqual(["plot"]))
    expect(api.removeDocument).toHaveBeenCalledWith("proj", "project/proj/document/diagram-toolu_2.md")
  })

  it("lays plots out at a zoom-independent width and hides them only while a zoom gesture runs", async () => {
    api.loadFileViewerText.mockResolvedValueOnce(JSON.stringify({ data: [] }))
    const seen: CanvasDocument[] = []
    const initialDoc: CanvasDocument = {
      version: 1, frames: [], strokes: [], texts: [],
      attachments: [{ id: "plot", kind: "document", path: "project/proj/document/p.plot.json", x: 0, y: 0, width: 480, height: 320 }],
    }
    const { container, getByTestId } = render(<Harness seen={seen} slug="proj" initialDoc={initialDoc} />)
    await act(async () => {})
    const frame = () => getByTestId("document-plot").parentElement!
    const layoutWidth = frame().style.width
    expect(parseFloat(layoutWidth)).toBeGreaterThanOrEqual(960)
    expect(frame().style.visibility).toBe("")

    vi.useFakeTimers()
    const canvas = container.querySelector<HTMLElement>("[tabindex='0']")!
    act(() => { canvas.dispatchEvent(new WheelEvent("wheel", { deltaY: -100, bubbles: true, cancelable: true })) })
    expect(frame().style.visibility).toBe("hidden")
    expect(frame().style.width).toBe(layoutWidth)

    act(() => { vi.advanceTimersByTime(300) })
    expect(frame().style.visibility).toBe("")
    vi.useRealTimers()
  })

  it("lists notes-scoped artifacts when the canvas has no project slug", async () => {
    api.listNotes.mockResolvedValueOnce([
      { path: "note/chart.plot.json", name: "chart.plot.json", mime: "application/json" },
    ])
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} />)

    act(() => { toolbar(container)[0]!.click() })
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

    act(() => { toolbar(container)[0]!.click() })
    await act(async () => {})
    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === "broken.plot.json")
    act(() => { item!.click() })
    await act(async () => {})

    expect(latest(seen).attachments[0]!.kind).toBe("document")
    expect(container.textContent).toContain("Malformed plot document")
  })
})

describe("cross-canvas stroke transfer", () => {
  const view = (left: number, top: number, scale: number, ox = 0, oy = 0) =>
    ({ rect: { left, top }, scale, offset: { x: ox, y: oy } })

  it("keeps the ink where it was dropped, at the size it looked", () => {
    // source: 2x zoom, panned; target: 1x, offset window — a point at source (10,10)
    // sits at client (100+10*2+5, 50+10*2+5) = (125, 75)
    const from = view(100, 50, 2, 5, 5)
    const to = view(25, 25, 1, 0, 0)
    const moved = remapAcrossCanvases({ id: crypto.randomUUID(), points: [[10, 10]], color: "#000", width: 4 }, from, to)
    expect(moved.points[0]!.slice(0, 2)).toEqual([100, 50]) // 125-25, 75-25 at scale 1
    expect(moved.width).toBe(8) // half the zoom means twice the units for the same px
    expect(moved.color).toBe("#000")
  })

  it("is a no-op between identical views", () => {
    const v = view(40, 40, 1.5, 12, -8)
    const s: StrokeData = { id: crypto.randomUUID(), points: [[3, 4, 0.7], [9, 1, 0.2]], color: "#f00", width: 2 }
    expect(remapAcrossCanvases(s, v, v)).toEqual(s)
  })
})

// Wheel and two-finger gestures are native listeners on the host surface, so they fire
// before any JSX stopPropagation can stop them. The host must ignore events inside a
// nested window, and the nested canvas must NOT ignore its own — its container sits
// inside an attachment element too, which a plain `closest` test can't tell apart.
describe("gesture scoping between canvases", () => {
  const build = () => {
    const host = document.createElement("div")
    const window_ = document.createElement("div")
    window_.setAttribute("data-canvas-attachment", "")
    const nested = document.createElement("div")
    const ink = document.createElement("span")
    host.append(window_)
    window_.append(nested)
    nested.append(ink)
    return { host, nested, ink }
  }

  it("keeps the host out of its own nested window", () => {
    const { host, ink } = build()
    expect(inOwnAttachment(ink, host)).toBe(true)
  })

  it("still lets the nested canvas zoom and pan itself", () => {
    const { nested, ink } = build()
    expect(inOwnAttachment(ink, nested)).toBe(false)
  })

  it("leaves the bare canvas surface alone", () => {
    const host = document.createElement("div")
    const ink = document.createElement("span")
    host.append(ink)
    expect(inOwnAttachment(ink, host)).toBe(false)
  })
})
