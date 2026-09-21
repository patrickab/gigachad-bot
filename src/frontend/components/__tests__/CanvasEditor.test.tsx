/**
 * Canvas editor regression tests: undo/redo history, lasso selection geometry, and the
 * idle-loop guard.
 *
 * History used to be a stroke-only stack: deleting a frame, moving anything or
 * erasing was unrecoverable, and Ctrl+Z after one of those silently ate an
 * unrelated stroke. It is now whole-doc snapshots.
 *
 * The subtle failure this guards is snapshot ordering: the pre-change doc must be
 * captured when the mutation is dispatched, NOT inside the setState updater — those
 * run during the next render, after liveRef already points at the post-change doc,
 * which would make every undo a no-op.
 */
import { useState } from "react"
import { describe, it, expect, vi } from "vitest"
import { render, act, fireEvent } from "@testing-library/react"
import { TabActiveProvider } from "@/components/TabManager"

vi.mock("@/components/PdfViewer", () => ({ PdfViewer: () => null }))
vi.mock("./PdfViewer", () => ({ PdfViewer: () => null }))
const api = vi.hoisted(() => ({
  fileViewerRawUrl: (p: string) => `/raw/${p}`,
  writeBinaryDocument: vi.fn(),
  listProjectDocuments: vi.fn(async () => [{ path: "project/proj/document/notes.canvas", name: "notes.canvas", mime: "application/json" }]),
  loadFileViewerText: vi.fn(async () => ""),
  listArchitectureGraphs: vi.fn(async () => []),
  writeDocument: vi.fn(async (_slug: string, _name: string, _content: string) => ({ path: "project/proj/document/notes.canvas", name: "notes.canvas", mime: "application/json" })),
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status: number) {
      super(message)
      this.status = status
    }
  },
}))
vi.mock("@/lib/api", () => api)

import {
  CanvasEditor, appendStrokePoint, emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc, polyBounds, resizeBox, scaleStroke, scalePoly, remapAcrossCanvases, inOwnAttachment,
  type CanvasDocument, type SelBox,
} from "@/components/CanvasEditor"
import type { StrokeData } from "@/lib/drawing"

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= RO as unknown as typeof ResizeObserver

function Harness({ seen, slug, active = true }: { seen: CanvasDocument[]; slug?: string; active?: boolean }) {
  const [doc, setDoc] = useState<CanvasDocument>(emptyCanvasDoc())
  seen.push(doc)
  return (
    <TabActiveProvider value={active}>
      <CanvasEditor doc={doc} onChange={setDoc} slug={slug} docPath="project/proj/document/host.canvas" />
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
  const stroke = (pts: number[][], width = 4): StrokeData => ({ points: pts, color: "#000", width })
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
    expect(note).not.toHaveFocus()

    act(() => { note.focus() })
    expect(fireEvent.keyDown(note, { key: "Enter", ctrlKey: true })).toBe(true)
    expect(note).toHaveFocus()
    act(() => { fireEvent.change(note, { target: { value: "H\nI" } }) })
    expect(latest(seen).texts[0]!.text).toBe("H\nI")
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
    await act(async () => {}) // the picker fetches the project's canvases on open
    const item = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === name)!
    act(() => { item.click() })
    await act(async () => {}) // …and the window loads the file it points at
  }

  it("opens a project canvas as a window holding only its path", async () => {
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    await openCanvas(container, "notes.canvas")
    const [att] = latest(seen).attachments
    expect(att).toMatchObject({ kind: "canvas", path: "project/proj/document/notes.canvas" })
    expect(att!.canvas).toBeUndefined() // contents stay in the file, not in the host doc
    expect(api.loadFileViewerText).toHaveBeenCalledWith("project/proj/document/notes.canvas")
  })

  it("writes drawings back to the file when the window closes", async () => {
    api.writeDocument.mockClear()
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    await openCanvas(container, "notes.canvas")
    const window_ = container.querySelector("[data-canvas-attachment]") as HTMLElement
    addPage(window_)
    expect(latest(seen).attachments[0]!.canvas).toBeUndefined() // still not in the host

    // closing flushes before the autosave debounce would have fired
    const close = Array.from(container.querySelectorAll("button")).find((b) => b.closest("[data-canvas-attachment]") === null && b.querySelector("svg.lucide-x"))!
    act(() => { close.click() })

    const [slug, name, written] = api.writeDocument.mock.calls.at(-1)!
    expect([slug, name]).toEqual(["proj", "notes.canvas"])
    expect(parseCanvasDoc(written).frames).toHaveLength(1)
  })

  it("does not overwrite the file when its window fails to load", async () => {
    api.writeDocument.mockClear()
    api.loadFileViewerText.mockRejectedValueOnce(new api.ApiError("boom", 500))
    const seen: CanvasDocument[] = []
    const { container } = render(<Harness seen={seen} slug="proj" />)

    await openCanvas(container, "notes.canvas")
    const window_ = container.querySelector("[data-canvas-attachment]") as HTMLElement
    expect(window_.textContent).toMatch(/Failed to load/i)

    // closing would normally flush a save — it must stay inert since nothing loaded
    const close = Array.from(container.querySelectorAll("button")).find((b) => b.closest("[data-canvas-attachment]") === null && b.querySelector("svg.lucide-x"))!
    act(() => { close.click() })

    expect(api.writeDocument).not.toHaveBeenCalled()
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

describe("cross-canvas stroke transfer", () => {
  const view = (left: number, top: number, scale: number, ox = 0, oy = 0) =>
    ({ rect: { left, top }, scale, offset: { x: ox, y: oy } })

  it("keeps the ink where it was dropped, at the size it looked", () => {
    // source: 2x zoom, panned; target: 1x, offset window — a point at source (10,10)
    // sits at client (100+10*2+5, 50+10*2+5) = (125, 75)
    const from = view(100, 50, 2, 5, 5)
    const to = view(25, 25, 1, 0, 0)
    const moved = remapAcrossCanvases({ points: [[10, 10]], color: "#000", width: 4 }, from, to)
    expect(moved.points[0]!.slice(0, 2)).toEqual([100, 50]) // 125-25, 75-25 at scale 1
    expect(moved.width).toBe(8) // half the zoom means twice the units for the same px
    expect(moved.color).toBe("#000")
  })

  it("is a no-op between identical views", () => {
    const v = view(40, 40, 1.5, 12, -8)
    const s: StrokeData = { points: [[3, 4, 0.7], [9, 1, 0.2]], color: "#f00", width: 2 }
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
