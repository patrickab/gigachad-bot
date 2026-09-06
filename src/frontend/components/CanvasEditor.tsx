"use client"

import { memo, useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { getStroke } from "perfect-freehand"
import { type StrokeData, type EmbedRect, getSvgPathFromStroke, renderPageToPng } from "@/lib/drawing"
import { createArchitectureGraph, fileViewerRawUrl, writeBinaryDocument, listArchitectureGraphs, listProjectDocuments, loadFileViewerText, readArchitectureGraph, writeArchitectureGraph, writeDocument } from "@/lib/api"
import { emptyArchitectureGraph, parseArchitectureGraph, serializeArchitectureGraph, type ArchitectureGraph } from "@/lib/architectureGraph"
import { useGraphAutosave } from "@/lib/graphAutosave"
import { activeThemeName } from "@/lib/palette"
import { cn } from "@/lib/utils"
import { Plus, Undo2, Redo2, Trash2, Copy, FileType, ImageIcon, X, Camera, CircleDashed, Type, SquarePen, PenLine, Maximize2, Minimize2 } from "lucide-react"
import { PdfViewer } from "./PdfViewer"
import { ArchitectureGraphSurface } from "./ArchitectureGraphSurface"

const A4_W = 794
const A4_H = 1123
const A4_ASPECT = A4_H / A4_W // height / width — page frames are locked to this
const PAGE_GAP = 40
const DEFAULT_PDF_WIDTH = 500 // attachment default width (canvas units)
const DEFAULT_IMAGE_WIDTH = 400 // image-frame default width (canvas units)
const DEFAULT_CANVAS_WIDTH = 600 // nested canvas window default width (canvas units)
const CANVAS_ASPECT = 0.7 // nested canvas windows have no intrinsic aspect — fix one
const DEFAULT_GRAPH_WIDTH = 720
const DEFAULT_GRAPH_HEIGHT = 480
const NESTED_SAVE_MS = 1000 // quiet time before a file-backed canvas window writes back
const MAX_NEST_DEPTH = 2 // canvas windows stop opening files here, so A→B→A can't recurse
const MIN_FRAME_WIDTH = 120
const MIN_ATTACH_WIDTH = 200
const MIN_GRAPH_HEIGHT = 180
const FALLBACK_ASPECT = 1.3 // height/width used until the real aspect is measured
const MIN_SCALE = 0.15
const MAX_SCALE = 3
const THIN_WIDTH = 2
const MEDIUM_WIDTH = 5
const THICK_WIDTH = 10
const PEN_HOLD_MS = 400 // press-and-hold the pen button to activate selection mode
const PEN_DOUBLE_CLICK_MS = 350 // double-click the pen button to activate screenshot mode
const TEXT_DEFAULT_SIZE = 28 // canvas units — font size for new text notes
const TEXT_DEFAULT_WIDTH = 240 // box size for a plain click (no drag) in text mode
const TEXT_DEFAULT_HEIGHT = 80
const MIN_TEXT_WIDTH = 80
const MIN_TEXT_HEIGHT = 40
const STRAIGHTEN_HOLD_MS = 500 // hold the pen still mid-stroke to snap it into a straight line
const PEN_TOUCH_SUPPRESS_MS = 700 // touch gestures stay suppressed this long after the last pen event (hover included)
const ZOOM_SETTLE_MS = 250 // attachments re-rasterize at full resolution this long after the zoom stops changing
const PDF_LAYOUT_CAP = 916 // PdfViewer caps rendering at 900px + scrollbar gutter — wider layout gains no resolution
const STRAIGHTEN_MOVE_TOLERANCE = 5 // px of client movement that resets the "holding still" timer
const HISTORY_LIMIT = 50 // undo depth — whole-doc snapshots, so this is the memory knob
const LASSO_COVERAGE = 0.6 // fraction of a stroke's points that must fall inside the lasso to select it
const SELECTION_MIN = 24 // canvas units — selection box can be squeezed no smaller
const COPY_OFFSET_PX = 24 // screen px a duplicated selection is nudged by, so it reads as a new object

const STROKE_OPTIONS = {
  smoothing: 0.5,
  streamline: 0.5,
  simulatePressure: false,
  last: true,
} as const

// Stroke objects are immutable once committed, so their outline path and bounds are
// computed once per object — not on every render, and not on every eraser sample.
// Weak keys, so a stroke dropped by an edit takes its cached geometry with it.
const strokePathCache = new WeakMap<StrokeData, string>()
const strokeBoundsCache = new WeakMap<StrokeData, [number, number, number, number]>()

// The eraser tests every stroke on the canvas against every pointermove, so rejecting a
// stroke has to be cheap: four comparisons against a cached box instead of a distance
// computation per segment. On a long session's canvas this is the difference between
// scanning every point ever drawn and scanning the handful actually under the tip.
function strokeBounds(s: StrokeData): [number, number, number, number] {
  let b = strokeBoundsCache.get(s)
  if (b === undefined) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const p of s.points) {
      minX = Math.min(minX, p[0]!); maxX = Math.max(maxX, p[0]!)
      minY = Math.min(minY, p[1]!); maxY = Math.max(maxY, p[1]!)
    }
    b = [minX, minY, maxX, maxY]
    strokeBoundsCache.set(s, b)
  }
  return b
}
function strokePath(stroke: StrokeData): string {
  let d = strokePathCache.get(stroke)
  if (d === undefined) {
    d = getSvgPathFromStroke(getStroke(stroke.points, { ...STROKE_OPTIONS, size: stroke.width }))
    strokePathCache.set(stroke, d)
  }
  return d
}

const PRESET_COLORS = [
  { name: "Black", value: "#000000" },
  { name: "Red", value: "#dc2626" },
  { name: "Green", value: "#16a34a" },
  { name: "Blue", value: "#2563eb" },
] as const

type StrokeSize = "thin" | "medium" | "thick"
const SIZE_MAP: Record<StrokeSize, number> = { thin: THIN_WIDTH, medium: MEDIUM_WIDTH, thick: THICK_WIDTH }

// --- Canvas primitives ---------------------------------------------------
// CanvasFrame  = content that IS the artwork (page or image). Rasterized into
//                the PDF/image export. Shared minimalistic chrome: drag the top
//                border to move, bottom-right corner to resize, top-right to delete.
// CanvasAttachment = a live, scrollable file reference (PDF; later text/LaTeX), or a
//                nested canvas window carrying its own CanvasDocument inline.
//                NOT baked into export. Filename header doubles as the move handle.

export interface CanvasFrame {
  id: string
  kind: "page" | "image"
  x: number
  y: number
  width: number
  path?: string // image source (library/project path); pages have none
}

export interface CanvasAttachment {
  id: string
  kind: "pdf" | "canvas" | "architecture-graph"
  path?: string // pdf source; nested canvases have none
  canvas?: CanvasDocument // nested canvas contents (kind === "canvas")
  x: number
  y: number
  width: number
  height?: number // Architecture Graphs own their viewport height; legacy attachments retain their aspect
}

// CanvasText = handwriting-style note in a resizable box. Rasterized into export
// like CanvasFrame content (word-wrapped and clipped to width/height).
export interface CanvasText {
  id: string
  x: number
  y: number
  width: number
  height: number
  text: string
  color: string
  size: number
}

export interface CanvasDocument {
  version: 1
  viewport?: { scale: number; centerX: number; centerY: number }
  frames: CanvasFrame[]
  strokes: StrokeData[]
  attachments: CanvasAttachment[]
  texts: CanvasText[]
}

export function emptyCanvasDoc(): CanvasDocument {
  return { version: 1, frames: [], strokes: [], attachments: [], texts: [] }
}

// Stable fallback for a nested canvas saved without contents — a fresh object per
// render would reset the nested editor's identity on every parent re-render.
const EMPTY_NESTED = emptyCanvasDoc()

// Legacy on-disk shape (pages / pdfEmbeds / imageEmbeds) — migrated on load.
type LegacyEmbed = { id: string; path: string; x: number; y: number; width: number }
// earlier point-text model saved notes without width/height
type LegacyText = Omit<CanvasText, "width" | "height"> & { width?: number; height?: number }
type LegacyDoc = {
  version: 1
  viewport?: CanvasDocument["viewport"]
  strokes?: StrokeData[]
  frames?: CanvasFrame[]
  attachments?: CanvasAttachment[]
  texts?: LegacyText[]
  pages?: { id: string; x: number; y: number }[]
  pdfEmbeds?: LegacyEmbed[]
  imageEmbeds?: LegacyEmbed[]
}

// backfills width/height for text notes saved under the earlier point-text model
function migrateTexts(texts: LegacyText[] | undefined): CanvasText[] {
  return (texts ?? []).map((t) => ({ ...t, width: t.width ?? TEXT_DEFAULT_WIDTH, height: t.height ?? TEXT_DEFAULT_HEIGHT }))
}

function migrate(d: LegacyDoc): CanvasDocument {
  if (Array.isArray(d.frames)) {
    return { version: 1, viewport: d.viewport, frames: d.frames, strokes: d.strokes ?? [], attachments: d.attachments ?? [], texts: migrateTexts(d.texts) }
  }
  const frames: CanvasFrame[] = []
  for (const p of d.pages ?? []) frames.push({ id: p.id, kind: "page", x: p.x, y: p.y, width: A4_W })
  for (const im of d.imageEmbeds ?? []) frames.push({ id: im.id, kind: "image", x: im.x, y: im.y, width: im.width, path: im.path })
  const attachments: CanvasAttachment[] = (d.pdfEmbeds ?? []).map((e) => ({ id: e.id, kind: "pdf", x: e.x, y: e.y, width: e.width, path: e.path }))
  return { version: 1, viewport: d.viewport, frames, strokes: d.strokes ?? [], attachments, texts: migrateTexts(d.texts) }
}

export function parseCanvasDoc(text: string): CanvasDocument {
  try {
    const parsed = JSON.parse(text) as LegacyDoc
    if (parsed && parsed.version === 1) return migrate(parsed)
  } catch { /* */ }
  return emptyCanvasDoc()
}

export function serializeCanvasDoc(doc: CanvasDocument): string {
  return JSON.stringify(doc)
}

interface CanvasEditorProps {
  doc: CanvasDocument
  onChange: (doc: CanvasDocument) => void
  availablePdfs?: { path: string; name: string }[]
  availableImages?: { path: string; name: string }[]
  slug?: string
  onImageAdded?: (path: string) => void
  // When given, the toolbar renders into this external element (e.g. the app
  // header in canvas mode) instead of above the drawing surface.
  toolbarSlot?: HTMLElement | null
  // Path of the `.canvas` file being edited — kept out of the "open canvas" picker so a
  // canvas can't embed itself.
  docPath?: string
  // Nesting depth, set only by canvas windows on themselves. Past MAX_NEST_DEPTH a
  // canvas window stops opening its file, so a cycle (A embeds B, B embeds A) can't
  // recurse forever.
  depth?: number
}

// convert between center (canvas-space point at view center) and offset (SVG translate)
function centerToOffset(cx: number, cy: number, s: number, w: number, h: number) {
  return { x: w / 2 - cx * s, y: h / 2 - cy * s }
}
function offsetToCenter(ox: number, oy: number, s: number, w: number, h: number) {
  return { cx: (w / 2 - ox) / s, cy: (h / 2 - oy) / s }
}

// shortest distance from (px,py) to segment (ax,ay)-(bx,by) — straight strokes (e.g.
// straightened lines) can be as sparse as 2 points, so the eraser must hit-test the
// segment between points, not just the points themselves
function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy
  const t = lenSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

// ray-casting point-in-polygon test — used by the lasso selection tool
function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!
    const [xj, yj] = poly[j]!
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

// --- Selection ------------------------------------------------------------
// The lasso keeps its freehand shape — that outline is what you drag to move, and it
// deforms along with the ink when scaled. A single handle at the shape's bottom-right
// resizes diagonally, anchored at the top-left. Stroke `width` is never touched by a
// scale — squeezing a drawing must not thin the pen that drew it.
export interface SelBox { x: number; y: number; w: number; h: number }
type Poly = [number, number][]

export function polyBounds(poly: Poly): SelBox | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of poly) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x)
    minY = Math.min(minY, y); maxY = Math.max(maxY, y)
  }
  if (minX === Infinity) return null
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
}

// The bottom-right handle: top-left stays put, both axes scale independently. Clamped to
// SELECTION_MIN so the shape can't collapse (zero width ⇒ infinite scale factor) or flip.
export function resizeBox(b: SelBox, dx: number, dy: number): SelBox {
  return { x: b.x, y: b.y, w: Math.max(SELECTION_MIN, b.w + dx), h: Math.max(SELECTION_MIN, b.h + dy) }
}

// Map a point from one box into another. Pure scale+translate — `width` is not a factor.
function remap(p: number[], from: SelBox, to: SelBox): number[] {
  return [
    to.x + (p[0]! - from.x) * (to.w / from.w),
    to.y + (p[1]! - from.y) * (to.h / from.h),
    p[2] ?? 0.5,
  ]
}

export function scalePoly(poly: Poly, from: SelBox, to: SelBox): Poly {
  return poly.map((p) => { const [x, y] = remap(p, from, to); return [x!, y!] as [number, number] })
}

// Rewrite a stroke's geometry from one selection box into another. `width` is carried
// over deliberately untouched: squeezing a drawing must not thin the pen that drew it.
export function scaleStroke(stroke: StrokeData, points: number[][], from: SelBox, to: SelBox): StrokeData {
  return { ...stroke, points: points.map((p) => remap(p, from, to)) }
}

// --- Cross-canvas stroke transfer ------------------------------------------------
// Every mounted canvas registers its drawing surface here, so a selection dragged out of
// one canvas can be handed to whichever canvas is under the pointer on release — into a
// nested window or back out of one. The handoff is in client coordinates, so the ink
// lands exactly where it was dropped and comes out at the target's zoom: same on-screen
// size, which means `width` scales by the zoom ratio (a lasso resize deliberately leaves
// the pen alone; a canvas with a different zoom is the opposite case).
// A canvas surface as the transfer sees it: where it sits on screen, and how it maps its
// own units onto that. `rect` only needs left/top, so tests can pass a bare object.
export interface CanvasView { rect: { left: number; top: number }; scale: number; offset: { x: number; y: number } }
type StrokeDrop = (strokes: StrokeData[], from: CanvasView) => void
const canvasDropTargets = new Map<HTMLElement, StrokeDrop>()

// Rewrite a stroke from one canvas's units into another's, going through client px, so
// it keeps both its position and its size on screen. `width` is in canvas units, so it
// takes the zoom ratio too — dropping into a canvas zoomed out 2× would otherwise double
// the apparent pen thickness.
export function remapAcrossCanvases(stroke: StrokeData, from: CanvasView, to: CanvasView): StrokeData {
  return {
    ...stroke,
    width: stroke.width * (from.scale / to.scale),
    points: stroke.points.map((p) => [
      (from.rect.left + p[0]! * from.scale + from.offset.x - to.rect.left - to.offset.x) / to.scale,
      (from.rect.top + p[1]! * from.scale + from.offset.y - to.rect.top - to.offset.y) / to.scale,
      p[2] ?? 0.5,
    ]),
  }
}

// Innermost registered surface under the point, skipping the one being dragged from —
// walking up from the topmost element is what picks the nested window over its host.
function dropTargetAt(x: number, y: number, self: HTMLElement | null): StrokeDrop | null {
  let el: Element | null = document.elementFromPoint(x, y)
  while (el) {
    const drop = canvasDropTargets.get(el as HTMLElement)
    if (drop && el !== self) return drop
    el = el.parentElement
  }
  return null
}

// True when an event landed in an attachment window that *this* surface owns — those
// (PDFs, nested canvases) handle their own scroll and zoom, so the host must keep its
// hands off. Containment is the whole point: a nested canvas's own container also sits
// inside an attachment element, its own, and that one must still zoom itself. Native
// listeners run before React's synthetic dispatch, so a JSX stopPropagation can't do it.
export function inOwnAttachment(target: EventTarget | null, el: HTMLElement): boolean {
  const att = (target as Element | null)?.closest?.("[data-canvas-attachment]")
  return !!att && el.contains(att)
}

// ponytail: swap black↔white for display only, stored colour stays unchanged
function inkColor(c: string, isDark: boolean): string {
  return isDark && c === "#000000" ? "#ffffff" : c
}

// The committed ink is by far the biggest thing on the canvas, and it does not change
// when you pan, zoom, type in a note or drag a frame — memoizing it keeps those off an
// O(strokes) reconciliation. `hidden` lets a drag preview take a few strokes over
// without touching the layer, so a selection drag re-renders k strokes, not all of them.
const StrokeLayer = memo(function StrokeLayer({ strokes, hidden, isDark }: { strokes: StrokeData[]; hidden: Set<number> | null; isDark: boolean }) {
  return (
    <>
      {strokes.map((stroke, i) => {
        if (hidden?.has(i)) return null
        const d = strokePath(stroke)
        return d ? <path key={i} d={d} fill={inkColor(stroke.color, isDark)} /> : null
      })}
    </>
  )
})

export function CanvasEditor({ doc, onChange, availablePdfs, availableImages, slug, onImageAdded, toolbarSlot, docPath, depth = 0 }: CanvasEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // latest doc/onChange for the long-lived window drag listeners and for history
  const liveRef = useRef({ doc, onChange })
  liveRef.current = { doc, onChange }

  const initScale = doc.viewport?.scale ?? 0.5
  // offset is derived once the container mounts; fall back to a reasonable default
  const [scale, setScale] = useState(initScale)
  const [offset, setOffset] = useState({ x: 40, y: 40 })
  const scaleRef = useRef(scale)
  const offsetRef = useRef(offset)
  scaleRef.current = scale
  offsetRef.current = offset

  // on mount (and, top-level only, on resize) recompute offset from the stored center so
  // the same canvas point stays centered regardless of container dimensions
  const restoredRef = useRef(false)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const apply = () => {
      const vp = doc.viewport
      const cx = vp?.centerX ?? 0
      const cy = vp?.centerY ?? 0
      const s = vp?.scale ?? 0.5
      const o = centerToOffset(cx, cy, s, el.clientWidth, el.clientHeight)
      offsetRef.current = o
      scaleRef.current = s
      setOffset(o)
      setScale(s)
      restoredRef.current = true
    }
    apply()
    const ro = new ResizeObserver(() => {
      if (!restoredRef.current) return
      // A nested window is sized by its host's zoom, so re-centering would slide its ink
      // on every host zoom step — the two views must be independent. Anchor the top-left
      // instead: the ink stays put and the box reveals more or less of it. Top-level
      // canvases keep centering; there the container only resizes with the browser window.
      if (depth > 0) return
      // re-derive offset from current center so resize keeps the same canvas point centered
      const { cx, cy } = offsetToCenter(offsetRef.current.x, offsetRef.current.y, scaleRef.current, el.clientWidth, el.clientHeight)
      const o = centerToOffset(cx, cy, scaleRef.current, el.clientWidth, el.clientHeight)
      offsetRef.current = o
      setOffset(o)
    })
    ro.observe(el)
    return () => ro.disconnect()
  // only re-run on mount, not on doc changes (viewport is persisted separately)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const [isPanning, setIsPanning] = useState(false)
  const panStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 })

  const [isDrawing, setIsDrawing] = useState(false)
  // The in-progress stroke lives in a ref and renders imperatively via livePathRef,
  // so a pointermove never triggers a React re-render of the whole editor.
  const currentPointsRef = useRef<number[][]>([])
  const livePathRef = useRef<SVGPathElement>(null)
  const [isErasing, setIsErasing] = useState(false)
  const [strokeWidth, setStrokeWidth] = useState<StrokeSize>("thin")
  const [color, setColor] = useState("#000000")
  const [recentColors, setRecentColors] = useState<string[]>([])
  const [colorOpen, setColorOpen] = useState(false)
  const colorRef = useRef<HTMLDivElement>(null)

  // --- History: whole-doc snapshots, so frame/attachment deletes, moves, resizes,
  // erases and text edits are all undoable — not just strokes. `viewport` is excluded
  // on restore (a view concern; undoing shouldn't teleport the camera). ---
  const [undoStack, setUndoStack] = useState<CanvasDocument[]>([])
  const [redoStack, setRedoStack] = useState<CanvasDocument[]>([])

  // Push the pre-change doc. Call once per user gesture, before its first onChange.
  // `prev` must be read here, not inside the updater — updaters run during the next
  // render, by which point liveRef already holds the post-change doc.
  const snapshot = useCallback(() => {
    const prev = liveRef.current.doc
    setUndoStack((s) => [...s, prev].slice(-HISTORY_LIMIT))
    setRedoStack([])
  }, [])

  // Writing through this keeps `liveRef` in step within the same tick: two mutations can
  // land back to back before React re-renders (a cross-canvas stroke drop removes here
  // and adds there), and the second must not read the pre-first document.
  const applyChange = useCallback((next: CanvasDocument) => {
    liveRef.current.doc = next
    liveRef.current.onChange(next)
  }, [])

  // Discrete (non-drag) mutation: snapshot, then apply.
  const commit = useCallback((next: CanvasDocument) => {
    snapshot()
    applyChange(next)
  }, [snapshot, applyChange])

  // A drag fires onChange on every pointermove, so it must snapshot only once — and
  // only on the move that actually changes something, so a click that moves nothing
  // (or an eraser pass over empty space) leaves no dead undo step. Reset on pointerup.
  const gestureSnapped = useRef(false)
  const snapshotOnce = useCallback(() => {
    if (gestureSnapped.current) return
    gestureSnapped.current = true
    snapshot()
  }, [snapshot])

  const baseWidth = SIZE_MAP[strokeWidth]

  // Redraw the in-progress stroke.
  const redrawLive = useCallback(() => {
    const el = livePathRef.current
    if (!el) return
    const pts = currentPointsRef.current
    el.setAttribute("d", pts.length >= 2 ? getSvgPathFromStroke(getStroke(pts, { ...STROKE_OPTIONS, size: baseWidth })) : "")
  }, [baseWidth])

  // --- Select, move, scale & duplicate strokes. Draw a freehand lasso (not a
  // rectangle, so any shaped area works); once released the lasso is replaced by the
  // bounding box of what it caught, which is what you then drag, resize or copy.
  // Activated either via the toolbar icon or by press-and-holding the pen's side button
  // (which used to pan the canvas — panning is still available via space+drag / touch). ---
  type RectDrag = { x0: number; y0: number; x1: number; y1: number }
  const [selectionMode, setSelectionMode] = useState(false)
  const [lassoPoints, setLassoPoints] = useState<[number, number][] | null>(null) // in-progress lasso being drawn
  const lassoActive = useRef(false)
  const [selectionLasso, setSelectionLasso] = useState<Poly | null>(null) // frozen shape backing the current selection
  const [selectedStrokes, setSelectedStrokes] = useState<Set<number>>(new Set())
  const selDragRef = useRef<{ handle: "move" | "resize"; startX: number; startY: number; originals: Map<number, number[][]>; origLasso: Poly; origBox: SelBox; box: SelBox | null } | null>(null)
  // The strokes under an in-flight drag, as [index, stroke]. While this is set the
  // committed layer hides those indices and only this small list re-renders per frame;
  // the document itself is not touched until the gesture ends.
  const [dragPreview, setDragPreview] = useState<[number, StrokeData][] | null>(null)

  const clearSelection = useCallback(() => {
    setSelectedStrokes((prev) => (prev.size === 0 ? prev : new Set()))
    setSelectionLasso(null)
  }, [])

  // Receive a selection dropped from another canvas: source canvas units → client px →
  // our canvas units, so the ink keeps its screen position and screen size.
  const dropStrokes = useCallback<StrokeDrop>((incoming, from) => {
    const el = containerRef.current
    if (!el) return
    const to = { rect: el.getBoundingClientRect(), scale: scaleRef.current, offset: offsetRef.current }
    const cur = liveRef.current.doc
    commit({ ...cur, strokes: [...cur.strokes, ...incoming.map((s) => remapAcrossCanvases(s, from, to))] })
  }, [commit])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    canvasDropTargets.set(el, dropStrokes)
    return () => { canvasDropTargets.delete(el) }
  }, [dropStrokes])

  // Pen side-button gesture: hold => activate selection mode, double-click => activate screenshot mode
  const penHoldTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const penHoldFired = useRef(false)
  const penClickPending = useRef(false)
  const penClickTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // --- Screenshot: drag a rect, release copies the rasterized region to clipboard ---
  const [screenshotMode, setScreenshotMode] = useState(false)
  const [shotRect, setShotRect] = useState<RectDrag | null>(null)
  const shotStart = useRef<{ x: number; y: number } | null>(null)

  // --- Text: drag a box to define where a handwriting-style note goes. A plain click
  // (no real drag) still drops a default-size box. Clicking an existing note focuses its
  // textarea and places the cursor natively; a thin top band moves it, corner resizes it. ---
  const [textMode, setTextMode] = useState(false)
  const [textDragRect, setTextDragRect] = useState<RectDrag | null>(null)
  const textDragStart = useRef<{ x: number; y: number } | null>(null)
  const [autoFocusId, setAutoFocusId] = useState<string | null>(null)

  const [isDark, setIsDark] = useState(() => activeThemeName() === "dark")
  useEffect(() => {
    const obs = new MutationObserver(() => setIsDark(activeThemeName() === "dark"))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => obs.disconnect()
  }, [])

  // ponytail: close color popover on outside click, skip if native picker is active
  const nativePickerOpen = useRef(false)
  useEffect(() => {
    if (!colorOpen) return
    const handler = (e: MouseEvent) => {
      if (nativePickerOpen.current) return
      if (colorRef.current && !colorRef.current.contains(e.target as Node)) setColorOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [colorOpen])

  const pickColor = useCallback((c: string, close = true) => {
    setColor(c)
    setRecentColors((prev) => {
      const without = prev.filter((x) => x !== c)
      return [c, ...without].slice(0, 3)
    })
    if (close) setColorOpen(false)
  }, [])

  // Reads the doc through liveRef so this stays identity-stable: depending on `doc`
  // meant every document change re-armed the save timer, which then wrote a new doc,
  // which re-armed the timer — a 2Hz mutate/re-render loop that ran for as long as a
  // canvas was open. The no-op guard stops a write when the view hasn't actually moved.
  const persistViewport = useCallback((s: number, o: { x: number; y: number }) => {
    const el = containerRef.current
    if (!el) return
    const { cx, cy } = offsetToCenter(o.x, o.y, s, el.clientWidth, el.clientHeight)
    const { doc: cur, onChange: change } = liveRef.current
    const vp = cur.viewport
    if (vp && vp.scale === s && Math.abs(vp.centerX - cx) < 0.5 && Math.abs(vp.centerY - cy) < 0.5) return
    change({ ...cur, viewport: { scale: s, centerX: cx, centerY: cy } })
  }, [])

  const screenToCanvas = useCallback((clientX: number, clientY: number): [number, number] => {
    const el = containerRef.current
    if (!el) return [0, 0]
    const rect = el.getBoundingClientRect()
    const sx = clientX - rect.left
    const sy = clientY - rect.top
    return [(sx - offsetRef.current.x) / scaleRef.current, (sy - offsetRef.current.y) / scaleRef.current]
  }, [])

  // Wheel zoom
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (inOwnAttachment(e.target, el)) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const mx = e.clientX - rect.left
      const my = e.clientY - rect.top
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08
      const prev = scaleRef.current
      const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, prev * factor))
      const ratio = next / prev
      const newOx = mx - (mx - offsetRef.current.x) * ratio
      const newOy = my - (my - offsetRef.current.y) * ratio
      scaleRef.current = next
      offsetRef.current = { x: newOx, y: newOy }
      setScale(next)
      setOffset({ x: newOx, y: newOy })
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [])

  // Save viewport on idle
  const viewportTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    clearTimeout(viewportTimer.current)
    viewportTimer.current = setTimeout(() => {
      persistViewport(scale, offset)
    }, 500)
    return () => clearTimeout(viewportTimer.current)
  }, [scale, offset, persistViewport])

  // --- Pan (middle mouse or space+drag) ---
  const spaceDown = useRef(false)
  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.code === "Space" && !e.repeat) spaceDown.current = true }
    const up = (e: KeyboardEvent) => { if (e.code === "Space") spaceDown.current = false }
    window.addEventListener("keydown", down)
    window.addEventListener("keyup", up)
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up) }
  }, [])

  const handleContainerPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "touch" || screenshotMode || selectionMode) return
    // any pointerdown that bubbles this far didn't hit the selection lasso or its
    // handles (they stopPropagation) — so it's "outside" and drops the selection
    clearSelection()
    if (e.button === 1) {
      // pen side button: hold = activate selection mode & start the lasso right on this
      // same press (no lift-and-retouch needed), double-click = activate screenshot mode
      e.preventDefault()
      ;(e.target as Element).setPointerCapture(e.pointerId)
      penHoldFired.current = false
      clearTimeout(penHoldTimer.current)
      const clientX = e.clientX, clientY = e.clientY
      penHoldTimer.current = setTimeout(() => {
        penHoldFired.current = true
        setSelectionMode(true)
        setScreenshotMode(false)
        const [cx, cy] = screenToCanvas(clientX, clientY)
        lassoActive.current = true
        setLassoPoints([[cx, cy]])
      }, PEN_HOLD_MS)
      return
    }
    if (e.button === 0 && spaceDown.current) {
      e.preventDefault()
      setIsPanning(true)
      panStart.current = { x: e.clientX, y: e.clientY, ox: offsetRef.current.x, oy: offsetRef.current.y }
      ;(e.target as Element).setPointerCapture(e.pointerId)
    }
  }, [screenshotMode, selectionMode, clearSelection, screenToCanvas])

  const handleContainerPointerMove = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "touch") return
    if (!isPanning) return
    const dx = e.clientX - panStart.current.x
    const dy = e.clientY - panStart.current.y
    const newOffset = { x: panStart.current.ox + dx, y: panStart.current.oy + dy }
    offsetRef.current = newOffset
    setOffset(newOffset)
  }, [isPanning])

  const handleContainerPointerUp = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "touch") return
    if (e.button === 1) {
      clearTimeout(penHoldTimer.current)
      if (penHoldFired.current) { penHoldFired.current = false; return }
      if (penClickPending.current) {
        penClickPending.current = false
        clearTimeout(penClickTimer.current)
        setScreenshotMode(true)
        setSelectionMode(false)
      } else {
        penClickPending.current = true
        penClickTimer.current = setTimeout(() => { penClickPending.current = false }, PEN_DOUBLE_CLICK_MS)
      }
      return
    }
    if (isPanning) setIsPanning(false)
  }, [isPanning])

  // Test every stroke against the finished lasso loop and keep the ones it mostly
  // covers — full enclosure meant a stroke poking one pixel out was silently missed.
  // ponytail: coverage is measured on the raw sample points, so a 2-point straightened
  // line still effectively needs both ends inside; resample if that ever bites.
  const finalizeLasso = useCallback((pts: [number, number][] | null) => {
    if (pts && pts.length >= 3) {
      const picked = new Set<number>()
      doc.strokes.forEach((s, i) => {
        if (s.points.length === 0) return
        const inside = s.points.reduce((n, p) => n + (pointInPolygon(p[0]!, p[1]!, pts) ? 1 : 0), 0)
        if (inside / s.points.length > LASSO_COVERAGE) picked.add(i)
      })
      setSelectedStrokes(picked)
      setSelectionLasso(picked.size > 0 ? pts : null)
    }
  }, [doc.strokes])

  // --- Two-finger touch: pinch-zoom + pan. Implemented with pointer events, not
  // TouchEvents — some Linux browsers (Firefox) never deliver TouchEvents even
  // though touch pointer events fire fine. ---
  const touchRef = useRef<{ dist: number; cx: number; cy: number; scale: number; ox: number; oy: number } | null>(null)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const touches = new Map<number, { x: number; y: number }>()

    // Palm rejection: touch gestures are suppressed while the pen is in contact
    // and for PEN_TOUCH_SUPPRESS_MS after any pen event, hover included — the
    // palm lands just before the tip touches and lingers after it lifts. A pen
    // hover with no buttons also clears the contact flag, so a missed pointerup
    // can never leave touch permanently disabled. Capture phase so a
    // stopPropagation in some handler can't hide pen events from us.
    let penContact = false
    let penLastSeen = 0
    const onPen = (e: PointerEvent) => {
      if (e.pointerType !== "pen") return
      penLastSeen = performance.now()
      penContact = e.type === "pointerdown" || (e.type === "pointermove" && e.buttons !== 0)
      if (penContact && (touchRef.current || touches.size > 0)) { touchRef.current = null; touches.clear() }
    }
    const penEvents = ["pointerdown", "pointermove", "pointerup", "pointercancel"] as const
    penEvents.forEach((t) => window.addEventListener(t, onPen, true))
    const penNear = () => penContact || performance.now() - penLastSeen < PEN_TOUCH_SUPPRESS_MS

    const beginGesture = () => {
      const [a, b] = [...touches.values()]
      const rect = el.getBoundingClientRect()
      touchRef.current = {
        dist: Math.hypot(a!.x - b!.x, a!.y - b!.y),
        cx: (a!.x + b!.x) / 2 - rect.left,
        cy: (a!.y + b!.y) / 2 - rect.top,
        scale: scaleRef.current,
        ox: offsetRef.current.x,
        oy: offsetRef.current.y,
      }
    }
    const onDown = (e: PointerEvent) => {
      if (e.pointerType !== "touch" || penNear()) return
      // fingers that land in a nested window belong to that window's pinch/pan, not ours
      if (inOwnAttachment(e.target, el)) return
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY })
      touchRef.current = null
      if (touches.size === 2) beginGesture()
    }
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "touch" || !touches.has(e.pointerId)) return
      if (penNear()) { touches.clear(); touchRef.current = null; return }
      touches.set(e.pointerId, { x: e.clientX, y: e.clientY })
      const t = touchRef.current
      if (!t || touches.size !== 2) return
      const [a, b] = [...touches.values()]
      const rect = el.getBoundingClientRect()
      const newDist = Math.hypot(a!.x - b!.x, a!.y - b!.y)
      const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, t.scale * (newDist / t.dist)))
      const scaleRatio = next / t.scale
      const mx = (a!.x + b!.x) / 2 - rect.left
      const my = (a!.y + b!.y) / 2 - rect.top
      const newOx = t.cx - (t.cx - t.ox) * scaleRatio + (mx - t.cx)
      const newOy = t.cy - (t.cy - t.oy) * scaleRatio + (my - t.cy)
      scaleRef.current = next
      offsetRef.current = { x: newOx, y: newOy }
      setScale(next)
      setOffset({ x: newOx, y: newOy })
    }
    const onUp = (e: PointerEvent) => {
      if (e.pointerType !== "touch") return
      touches.delete(e.pointerId)
      touchRef.current = null
    }

    el.addEventListener("pointerdown", onDown)
    el.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    window.addEventListener("pointercancel", onUp)
    return () => {
      penEvents.forEach((t) => window.removeEventListener(t, onPen, true))
      el.removeEventListener("pointerdown", onDown)
      el.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      window.removeEventListener("pointercancel", onUp)
    }
  }, [])

  // Attachments scale with canvas zoom, but re-rendering the PDF at every zoom
  // frame re-rasterizes all pages (flicker). Instead the gesture scales them as
  // bitmaps via CSS transform, and once the zoom settles the layout width snaps
  // to the true screen size for one sharp re-render — PdfViewer preserves the
  // reading position across that width change.
  const [settledScale, setSettledScale] = useState(scale)
  useEffect(() => {
    const t = setTimeout(() => setSettledScale(scale), ZOOM_SETTLE_MS)
    return () => clearTimeout(t)
  }, [scale])

  // --- Aspect cache (image frames + pdf attachments), keyed by element id ---
  const [aspects, setAspects] = useState<Record<string, number>>({})
  const setAspect = useCallback((id: string, r: number) => {
    setAspects((prev) => (Math.abs((prev[id] ?? 0) - r) < 0.001 ? prev : { ...prev, [id]: r }))
  }, [])
  const aspectFor = useCallback((f: CanvasFrame) => f.kind === "page" ? A4_ASPECT : (aspects[f.id] ?? FALLBACK_ASPECT), [aspects])

  // --- Straighten-on-hold: pausing mid-stroke for STRAIGHTEN_HOLD_MS snaps the
  // in-progress freehand stroke to a straight line from its start to the hold point. ---
  const straightenTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const straightenAnchor = useRef<{ clientX: number; clientY: number } | null>(null)
  const straightened = useRef(false)

  const armStraighten = useCallback((clientX: number, clientY: number) => {
    straightenAnchor.current = { clientX, clientY }
    clearTimeout(straightenTimer.current)
    straightenTimer.current = setTimeout(() => {
      const pts = currentPointsRef.current
      if (pts.length < 2) return
      straightened.current = true
      currentPointsRef.current = [pts[0]!, pts[pts.length - 1]!]
      redrawLive()
    }, STRAIGHTEN_HOLD_MS)
  }, [redrawLive])

  const cancelStraighten = useCallback(() => {
    clearTimeout(straightenTimer.current)
    straightenAnchor.current = null
    straightened.current = false
  }, [])

  // --- Long-press context menu ---
  const [contextMenu, setContextMenu] = useState<{ screenX: number; screenY: number; canvasX: number; canvasY: number } | null>(null)
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const longPressPos = useRef<{ clientX: number; clientY: number } | null>(null)

  const startLongPress = useCallback((clientX: number, clientY: number) => {
    longPressPos.current = { clientX, clientY }
    longPressTimer.current = setTimeout(() => {
      const [cx, cy] = screenToCanvas(clientX, clientY)
      setIsDrawing(false)
      currentPointsRef.current = []
      redrawLive()
      cancelStraighten()
      setContextMenu({ screenX: clientX, screenY: clientY, canvasX: cx, canvasY: cy })
    }, 500)
  }, [screenToCanvas, cancelStraighten, redrawLive])

  const cancelLongPress = useCallback(() => {
    clearTimeout(longPressTimer.current)
    longPressPos.current = null
  }, [])

  useEffect(() => {
    if (!contextMenu) return
    const close = () => setContextMenu(null)
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") close() }
    document.addEventListener("pointerdown", close)
    document.addEventListener("keydown", esc)
    return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", esc) }
  }, [contextMenu])

  // --- Screenshot capture: rasterize a canvas-space rect and copy it to the clipboard ---
  const captureScreenshot = useCallback(async (rect: RectDrag) => {
    const x = Math.min(rect.x0, rect.x1)
    const y = Math.min(rect.y0, rect.y1)
    const w = Math.abs(rect.x1 - rect.x0)
    const h = Math.abs(rect.y1 - rect.y0)
    if (w < 4 || h < 4) return
    const images: EmbedRect[] = doc.frames
      .filter((f) => f.kind === "image" && f.path)
      .map((f) => ({ url: fileViewerRawUrl(f.path!), x: f.x, y: f.y, width: f.width, aspect: aspectFor(f) }))
    try {
      const pngBytes = await renderPageToPng(doc.strokes, x, y, w, h, images, doc.texts)
      const blob = new Blob([pngBytes.buffer as ArrayBuffer], { type: "image/png" })
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
    } catch { /* clipboard write can fail without permission — silently drop */ }
  }, [doc.frames, doc.strokes, doc.texts, aspectFor])

  // --- Drawing (on SVG) ---
  const handleSvgPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (spaceDown.current) return
    if (screenshotMode) {
      if (e.button !== 0) return
      e.stopPropagation()
      ;(e.target as Element).setPointerCapture(e.pointerId)
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      shotStart.current = { x: cx, y: cy }
      setShotRect({ x0: cx, y0: cy, x1: cx, y1: cy })
      return
    }
    if (selectionMode) {
      if (e.button !== 0) return
      e.stopPropagation()
      ;(e.target as Element).setPointerCapture(e.pointerId)
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      lassoActive.current = true
      setLassoPoints([[cx, cy]])
      return
    }
    if (textMode) {
      if (e.button !== 0) return
      e.stopPropagation()
      ;(e.target as Element).setPointerCapture(e.pointerId)
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      textDragStart.current = { x: cx, y: cy }
      setTextDragRect({ x0: cx, y0: cy, x1: cx, y1: cy })
      return
    }
    // drawing/erasing directly on the canvas counts as "outside" the selection
    clearSelection()
    if (e.button === 5) { setIsErasing(true); return }
    if (e.button !== 0) return
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    setIsDrawing(true)
    const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
    currentPointsRef.current = [[cx, cy, e.pressure > 0 ? e.pressure : 0.5]]
    redrawLive()
    // the long-press context menu (incl. "Paste image") is a touch/mouse affordance —
    // a pen stroke that pauses mid-draw must never be mistaken for a long-press
    if (e.pointerType !== "pen") startLongPress(e.clientX, e.clientY)
    armStraighten(e.clientX, e.clientY)
  }, [screenToCanvas, startLongPress, armStraighten, screenshotMode, selectionMode, textMode, clearSelection, redrawLive])

  const handleSvgPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (lassoActive.current) {
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      setLassoPoints((prev) => (prev ? [...prev, [cx, cy]] : [[cx, cy]]))
      return
    }
    if (shotStart.current) {
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      setShotRect({ x0: shotStart.current.x, y0: shotStart.current.y, x1: cx, y1: cy })
      return
    }
    if (textDragStart.current) {
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      setTextDragRect({ x0: textDragStart.current.x, y0: textDragStart.current.y, x1: cx, y1: cy })
      return
    }
    // cancel long-press if pointer moves more than a few pixels
    if (longPressPos.current) {
      const dx = e.clientX - longPressPos.current.clientX
      const dy = e.clientY - longPressPos.current.clientY
      if (dx * dx + dy * dy > 25) cancelLongPress()
    }
    if (isErasing) {
      const [ex, ey] = screenToCanvas(e.clientX, e.clientY)
      const threshold = 15 / scaleRef.current
      const updated = doc.strokes.filter((stroke) => {
        const [minX, minY, maxX, maxY] = strokeBounds(stroke)
        if (ex < minX - threshold || ex > maxX + threshold || ey < minY - threshold || ey > maxY + threshold) return true
        const pts = stroke.points
        if (pts.length === 1) return Math.hypot(pts[0]![0]! - ex, pts[0]![1]! - ey) >= threshold
        for (let i = 1; i < pts.length; i++) {
          if (distToSegment(ex, ey, pts[i - 1]![0]!, pts[i - 1]![1]!, pts[i]![0]!, pts[i]![1]!) < threshold) return false
        }
        return true
      })
      if (updated.length !== doc.strokes.length) {
        snapshotOnce()
        onChange({ ...doc, strokes: updated })
      }
      return
    }
    if (!isDrawing) return
    if (straightened.current) {
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      currentPointsRef.current = [currentPointsRef.current[0]!, [cx, cy, e.pressure > 0 ? e.pressure : 0.5]]
      redrawLive()
      return
    }
    if (straightenAnchor.current) {
      const dx = e.clientX - straightenAnchor.current.clientX
      const dy = e.clientY - straightenAnchor.current.clientY
      if (dx * dx + dy * dy > STRAIGHTEN_MOVE_TOLERANCE * STRAIGHTEN_MOVE_TOLERANCE) armStraighten(e.clientX, e.clientY)
    }
    // coalesced events surface the pen's full sample rate (~240Hz on pen hardware)
    // instead of one point per display frame
    const native = e.nativeEvent
    for (const ev of native.getCoalescedEvents?.() ?? [native]) {
      const [cx, cy] = screenToCanvas(ev.clientX, ev.clientY)
      currentPointsRef.current.push([cx, cy, ev.pressure > 0 ? ev.pressure : 0.5])
    }
    redrawLive()
  }, [isDrawing, isErasing, screenToCanvas, doc, onChange, cancelLongPress, armStraighten, redrawLive, snapshotOnce])

  const handleSvgPointerUp = useCallback((e?: React.PointerEvent<SVGSVGElement>) => {
    cancelLongPress()
    cancelStraighten()
    if (e?.pointerType === "touch") return
    if (lassoActive.current) {
      lassoActive.current = false
      setLassoPoints((pts) => { finalizeLasso(pts); return null })
      setSelectionMode(false)
      return
    }
    if (shotStart.current) {
      shotStart.current = null
      setShotRect((rect) => {
        if (rect) captureScreenshot(rect)
        return null
      })
      setScreenshotMode(false)
      return
    }
    if (textDragStart.current) {
      const start = textDragStart.current
      textDragStart.current = null
      setTextDragRect(null)
      if (e) {
        const [ex, ey] = screenToCanvas(e.clientX, e.clientY)
        const x = Math.min(start.x, ex)
        const y = Math.min(start.y, ey)
        const dragW = Math.abs(ex - start.x)
        const dragH = Math.abs(ey - start.y)
        // a plain click (no real drag) gets a default-size box instead of a sliver
        const width = dragW < MIN_TEXT_WIDTH ? TEXT_DEFAULT_WIDTH : dragW
        const height = dragH < MIN_TEXT_HEIGHT ? TEXT_DEFAULT_HEIGHT : dragH
        const id = `txt-${Date.now()}`
        commit({ ...doc, texts: [...doc.texts, { id, x, y, width, height, text: "", color, size: TEXT_DEFAULT_SIZE }] })
        setAutoFocusId(id)
      }
      return
    }
    if (isErasing) { setIsErasing(false); return }
    if (!isDrawing) return
    setIsDrawing(false)
    const pts = currentPointsRef.current
    currentPointsRef.current = []
    redrawLive()
    if (pts.length < 2) return
    commit({ ...doc, strokes: [...doc.strokes, { points: pts, color, width: baseWidth }] })
  }, [isDrawing, isErasing, color, baseWidth, doc, onChange, cancelLongPress, cancelStraighten, captureScreenshot, finalizeLasso, screenToCanvas, redrawLive, commit])

  // --- Undo / Redo ---
  // Selections index into doc.strokes, so a restore that shifts the array would leave
  // them pointing at the wrong strokes — both directions drop the selection.
  const restore = useCallback((target: CanvasDocument) => {
    const cur = liveRef.current.doc
    liveRef.current.onChange({ ...target, viewport: cur.viewport })
    clearSelection()
    return cur
  }, [clearSelection])

  const undo = useCallback(() => {
    if (undoStack.length === 0) return
    const cur = restore(undoStack[undoStack.length - 1]!)
    setUndoStack((s) => s.slice(0, -1))
    setRedoStack((s) => [...s, cur])
  }, [undoStack, restore])

  const redo = useCallback(() => {
    if (redoStack.length === 0) return
    const cur = restore(redoStack[redoStack.length - 1]!)
    setRedoStack((s) => s.slice(0, -1))
    setUndoStack((s) => [...s, cur].slice(-HISTORY_LIMIT))
  }, [redoStack, restore])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return
      const key = e.key.toLowerCase() // shift uppercases it, so Ctrl+Shift+Z arrives as "Z"
      // Redo also matches `code` — the physical key — for layouts where `key` isn't "y".
      // The undo branch must stay first: on QWERTZ the KeyY position is labelled Z.
      if (key === "z" && !e.shiftKey) {
        e.preventDefault()
        undo()
      } else if (key === "y" || e.code === "KeyY" || (key === "z" && e.shiftKey)) {
        e.preventDefault()
        redo()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [undo, redo])

  // One history snapshot per focus session of a text note, so typing a sentence is
  // one undo step, not one per keystroke. The first keystroke into a freshly created
  // (empty) box is skipped — the creation snapshot already covers it.
  const textEdited = useRef(false)

  // Removes a text note if its content is blank — called on blur so an
  // untouched or fully-cleared box doesn't linger as an empty artifact.
  const dropIfEmptyText = useCallback((id: string) => {
    const { doc: cur, onChange: change } = liveRef.current
    const item = cur.texts.find((t) => t.id === id)
    if (item && item.text.trim() === "") {
      change({ ...cur, texts: cur.texts.filter((t) => t.id !== id) })
    }
  }, [])

  // Escape cancels an in-progress screenshot/selection-drawing/text-box-drag, or drops the current stroke selection
  useEffect(() => {
    if (!screenshotMode && !selectionMode && !textMode && selectedStrokes.size === 0) return
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return
      shotStart.current = null
      setShotRect(null)
      setScreenshotMode(false)
      lassoActive.current = false
      setLassoPoints(null)
      setSelectionMode(false)
      textDragStart.current = null
      setTextDragRect(null)
      setTextMode(false)
      clearSelection()
    }
    document.addEventListener("keydown", esc)
    return () => document.removeEventListener("keydown", esc)
  }, [screenshotMode, selectionMode, textMode, selectedStrokes, clearSelection])

  const nextSize = useCallback(() => {
    setStrokeWidth((w) => w === "thin" ? "medium" : w === "medium" ? "thick" : "thin")
  }, [])

  // --- Add menu (pages / images / pdf attachments) ---
  const [addMenuOpen, setAddMenuOpen] = useState(false)
  const addMenuRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!addMenuOpen) return
    const handler = (e: MouseEvent) => {
      if (addMenuRef.current && !addMenuRef.current.contains(e.target as Node)) setAddMenuOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [addMenuOpen])

  // The project's other `.canvas` files, fetched when the menu opens — unlike PDFs and
  // images the parent has no such list, and it changes outside this component. Only the
  // outermost editor offers them: a canvas window can hold a file, not hand out more.
  const [projectCanvases, setProjectCanvases] = useState<{ path: string; name: string }[]>([])
  const [architectureGraphs, setArchitectureGraphs] = useState<{ path: string; name: string }[]>([])
  useEffect(() => {
    if (!addMenuOpen || slug == null || depth > 0) return
    listProjectDocuments(slug)
      .then((docs) => setProjectCanvases(docs
        .filter((d) => d.path.endsWith(".canvas") && d.path !== docPath)
        .map((d) => ({ path: d.path, name: d.name }))))
      .catch(() => { /* picker just stays empty */ })
  }, [addMenuOpen, slug, depth, docPath])

  useEffect(() => {
    if (!addMenuOpen || depth > 0) return
    listArchitectureGraphs().then((graphs) => setArchitectureGraphs(graphs)).catch(() => { /* picker just stays empty */ })
  }, [addMenuOpen, depth])

  const pointAtCenter = useCallback((w: number, aspect: number) => {
    const el = containerRef.current
    const vw = el ? el.clientWidth : 800
    const vh = el ? el.clientHeight : 600
    const cx = (-offsetRef.current.x + vw / 2) / scaleRef.current - w / 2
    const cy = (-offsetRef.current.y + vh / 2) / scaleRef.current - (w * aspect) / 2
    return { cx, cy }
  }, [])

  // --- Frames (pages + images) ---
  const addPage = useCallback(() => {
    const pages = doc.frames.filter((f) => f.kind === "page")
    const last = pages[pages.length - 1]
    const newX = last ? last.x + last.width + PAGE_GAP : 0
    commit({ ...doc, frames: [...doc.frames, { id: `p-${Date.now()}`, kind: "page", x: newX, y: last?.y ?? 0, width: A4_W }] })
  }, [doc, commit])

  const addImageFrameAt = useCallback((path: string, x: number, y: number) => {
    commit({ ...doc, frames: [...doc.frames, { id: `img-${Date.now()}`, kind: "image", x, y, width: DEFAULT_IMAGE_WIDTH, path }] })
  }, [doc, commit])

  const addImageFrame = useCallback((path: string) => {
    const { cx, cy } = pointAtCenter(DEFAULT_IMAGE_WIDTH, FALLBACK_ASPECT)
    addImageFrameAt(path, cx, cy)
    setAddMenuOpen(false)
  }, [pointAtCenter, addImageFrameAt])

  const removeFrame = useCallback((id: string) => {
    commit({ ...doc, frames: doc.frames.filter((f) => f.id !== id) })
    setAspects(({ [id]: _gone, ...rest }) => rest)
  }, [doc, commit])

  // --- Attachments (pdf) ---
  const addAttachment = useCallback((path: string) => {
    if (doc.attachments.some((a) => a.path === path)) { setAddMenuOpen(false); return }
    const { cx, cy } = pointAtCenter(DEFAULT_PDF_WIDTH, FALLBACK_ASPECT)
    commit({ ...doc, attachments: [...doc.attachments, { id: `pdf-${Date.now()}`, kind: "pdf", path, x: cx, y: cy, width: DEFAULT_PDF_WIDTH }] })
    setAddMenuOpen(false)
  }, [doc, commit, pointAtCenter])

  // Nested canvas window — a full CanvasEditor living inside this one, contents stored
  // inline in the parent document.
  const addCanvasWindow = useCallback(() => {
    const { cx, cy } = pointAtCenter(DEFAULT_CANVAS_WIDTH, CANVAS_ASPECT)
    commit({ ...doc, attachments: [...doc.attachments, { id: `cv-${Date.now()}`, kind: "canvas", canvas: emptyCanvasDoc(), x: cx, y: cy, width: DEFAULT_CANVAS_WIDTH }] })
    setAddMenuOpen(false)
  }, [doc, commit, pointAtCenter])

  // Same window, but bound to an existing project canvas: it loads, autosaves and
  // flushes on close by itself (`NestedCanvasFile`), so nothing of it lands in this doc.
  const addCanvasFile = useCallback((path: string) => {
    if (doc.attachments.some((a) => a.path === path)) { setAddMenuOpen(false); return }
    const { cx, cy } = pointAtCenter(DEFAULT_CANVAS_WIDTH, CANVAS_ASPECT)
    commit({ ...doc, attachments: [...doc.attachments, { id: `cv-${Date.now()}`, kind: "canvas", path, x: cx, y: cy, width: DEFAULT_CANVAS_WIDTH }] })
    setAddMenuOpen(false)
  }, [doc, commit, pointAtCenter])

  const addArchitectureGraph = useCallback((path: string) => {
    if (doc.attachments.some((attachment) => attachment.path === path)) { setAddMenuOpen(false); return }
    const { cx, cy } = pointAtCenter(DEFAULT_GRAPH_WIDTH, DEFAULT_GRAPH_HEIGHT / DEFAULT_GRAPH_WIDTH)
    commit({ ...doc, attachments: [...doc.attachments, {
      id: `ag-${Date.now()}`, kind: "architecture-graph", path, x: cx, y: cy, width: DEFAULT_GRAPH_WIDTH, height: DEFAULT_GRAPH_HEIGHT,
    }] })
    setAddMenuOpen(false)
  }, [doc, commit, pointAtCenter])

  const createArchitectureGraphWindow = useCallback(async () => {
    const title = window.prompt("Architecture Graph name")?.trim()
    if (!title) return
    const stem = title.replace(/\.architecture\.yaml$/, "").trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "")
    if (!stem) return
    const name = `${stem}.architecture.yaml`
    try {
      const created = await createArchitectureGraph(name, serializeArchitectureGraph(emptyArchitectureGraph(title)), slug)
      addArchitectureGraph(created.path)
    } catch { /* name already exists or graph storage unavailable */ }
  }, [addArchitectureGraph, slug])

  // ponytail: nested edits bypass the parent's history — the nested editor has its own
  // undo/redo, and snapshotting the whole parent per nested stroke would be absurd.
  const updateNestedCanvas = useCallback((id: string, nested: CanvasDocument) => {
    const cur = liveRef.current.doc
    applyChange({ ...cur, attachments: cur.attachments.map((a) => a.id === id ? { ...a, canvas: nested } : a) })
  }, [applyChange])

  // Which attachment window, if any, is blown up over the whole surface. Exiting drops
  // straight back onto the canvas that holds it.
  const [fullscreenId, setFullscreenId] = useState<string | null>(null)

  const removeAttachment = useCallback((id: string) => {
    commit({ ...doc, attachments: doc.attachments.filter((a) => a.id !== id) })
    setAspects(({ [id]: _gone, ...rest }) => rest)
    setFullscreenId((cur) => (cur === id ? null : cur))
  }, [doc, commit])

  // --- Paste image (Ctrl+V or long-press menu) ---
  const pasteImageAtPoint = useCallback(async (canvasX: number, canvasY: number) => {
    if (!slug) return
    try {
      const items = await navigator.clipboard.read()
      for (const item of items) {
        const imageType = item.types.find((t) => t.startsWith("image/"))
        if (!imageType) continue
        const blob = await item.getType(imageType)
        const ext = imageType.split("/")[1] || "png"
        const name = `pasted-${Date.now()}.${ext}`
        const res = await writeBinaryDocument(slug, name, blob)
        addImageFrameAt(res.path, canvasX, canvasY)
        onImageAdded?.(res.path)
        return
      }
    } catch { /* clipboard empty or no permission */ }
  }, [slug, addImageFrameAt, onImageAdded])

  useEffect(() => {
    const handler = (e: ClipboardEvent) => {
      // every canvas window listens on `window`; only the outermost may act, or one
      // paste imports the clipboard image once per open window
      if (!slug || depth > 0) return
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (!item.type.startsWith("image/")) continue
        e.preventDefault()
        const blob = item.getAsFile()
        if (!blob) continue
        const ext = item.type.split("/")[1] || "png"
        const name = `pasted-${Date.now()}.${ext}`
        const { cx, cy } = pointAtCenter(DEFAULT_IMAGE_WIDTH, FALLBACK_ASPECT)
        writeBinaryDocument(slug, name, blob).then((res) => {
          addImageFrameAt(res.path, cx, cy)
          onImageAdded?.(res.path)
        }).catch(() => {})
        return
      }
    }
    window.addEventListener("paste", handler)
    return () => window.removeEventListener("paste", handler)
  }, [slug, depth, pointAtCenter, addImageFrameAt, onImageAdded])

  // --- Shared move / resize for frames + attachments + text notes ---
  const dragRef = useRef<{
    id: string; target: "frame" | "attachment" | "text"; mode: "move" | "resize"
    startX: number; startY: number; origX: number; origY: number; origW: number; origH: number
  } | null>(null)

  const startInteraction = useCallback((id: string, target: "frame" | "attachment" | "text", mode: "move" | "resize", clientX: number, clientY: number) => {
    const list = target === "frame" ? liveRef.current.doc.frames : target === "attachment" ? liveRef.current.doc.attachments : liveRef.current.doc.texts
    const item = list.find((e) => e.id === id)
    if (!item) return
    dragRef.current = {
      id, target, mode, startX: clientX, startY: clientY, origX: item.x, origY: item.y,
      origW: "width" in item ? item.width : 0, origH: "height" in item ? (item.height ?? 0) : 0,
    }
  }, [])

  // Drag the selection box body to translate every selected stroke together, or a
  // handle to scale them. Originals are snapshotted at gesture start so every frame
  // maps from the same source — repeated relative deltas would drift.
  const startSelDrag = useCallback((handle: "move" | "resize", clientX: number, clientY: number) => {
    const originals = new Map<number, number[][]>()
    for (const i of selectedStrokes) {
      const s = liveRef.current.doc.strokes[i]
      if (s) originals.set(i, s.points.map((p) => [...p]))
    }
    const origBox = selectionLasso && polyBounds(selectionLasso)
    if (originals.size === 0 || !selectionLasso || !origBox) return
    selDragRef.current = { handle, startX: clientX, startY: clientY, originals, origLasso: selectionLasso.map((p) => [...p] as [number, number]), origBox, box: null }
  }, [selectedStrokes, selectionLasso])

  const deleteSelection = useCallback(() => {
    if (selectedStrokes.size === 0) return
    commit({ ...liveRef.current.doc, strokes: liveRef.current.doc.strokes.filter((_, i) => !selectedStrokes.has(i)) })
    clearSelection()
  }, [selectedStrokes, clearSelection, commit])

  // Duplicate the selection, offset so the copy is visibly its own object, and hand the
  // selection over to it — the originals drop out, the copies are what you keep dragging.
  const copySelection = useCallback(() => {
    const cur = liveRef.current.doc
    if (selectedStrokes.size === 0 || !selectionLasso) return
    const off = COPY_OFFSET_PX / scaleRef.current
    const copies = [...selectedStrokes]
      .map((i) => cur.strokes[i])
      .filter((s): s is StrokeData => Boolean(s))
      .map((s) => ({ ...s, points: s.points.map((p) => [p[0]! + off, p[1]! + off, p[2] ?? 0.5]) }))
    if (copies.length === 0) return
    commit({ ...cur, strokes: [...cur.strokes, ...copies] })
    setSelectedStrokes(new Set(copies.map((_, k) => cur.strokes.length + k)))
    setSelectionLasso(selectionLasso.map(([x, y]) => [x + off, y + off]))
  }, [selectedStrokes, selectionLasso, commit])

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const s = selDragRef.current
      if (s) {
        const dx = (e.clientX - s.startX) / scaleRef.current
        const dy = (e.clientY - s.startY) / scaleRef.current
        if (dx === 0 && dy === 0) return
        // A move translates the shape's box by the delta; the handle grows it. Either way
        // the strokes AND the lasso outline are remapped from the original box into the new
        // one, so the outline keeps hugging the ink. `width` is left alone — scaling ink
        // must not change how thick the pen was.
        const box = s.handle === "move"
          ? { ...s.origBox, x: s.origBox.x + dx, y: s.origBox.y + dy }
          : resizeBox(s.origBox, dx, dy)
        s.box = box
        // Preview only: writing the document here would rebuild every stroke array and
        // re-serialize the whole canvas on every pointermove. The commit happens on release.
        const strokes = liveRef.current.doc.strokes
        setDragPreview([...s.originals].map(([i, pts]) => [i, scaleStroke(strokes[i]!, pts, s.origBox, box)]))
        // the outline (and the buttons anchored to it) rides along with the strokes
        setSelectionLasso(scalePoly(s.origLasso, s.origBox, box))
        return
      }
      const d = dragRef.current
      if (!d) return
      const dx = (e.clientX - d.startX) / scaleRef.current
      const dy = (e.clientY - d.startY) / scaleRef.current
      if (dx === 0 && dy === 0) return
      snapshotOnce()
      const { doc: cur, onChange: change } = liveRef.current
      const minW = d.target === "frame" ? MIN_FRAME_WIDTH : MIN_ATTACH_WIDTH
      if (d.target === "frame") {
        change({ ...cur, frames: cur.frames.map((f) => f.id !== d.id ? f
          : d.mode === "resize" ? { ...f, width: Math.max(minW, d.origW + dx) }
          : { ...f, x: d.origX + dx, y: d.origY + dy }) })
      } else if (d.target === "attachment") {
        change({ ...cur, attachments: cur.attachments.map((a) => a.id !== d.id ? a
          : d.mode === "resize" ? a.kind === "architecture-graph"
            ? { ...a, width: Math.max(minW, d.origW + dx), height: Math.max(MIN_GRAPH_HEIGHT, d.origH + dy) }
            : { ...a, width: Math.max(minW, d.origW + dx) }
          : { ...a, x: d.origX + dx, y: d.origY + dy }) })
      } else {
        change({ ...cur, texts: cur.texts.map((t) => t.id !== d.id ? t
          : d.mode === "resize" ? { ...t, width: Math.max(MIN_TEXT_WIDTH, d.origW + dx), height: Math.max(MIN_TEXT_HEIGHT, d.origH + dy) }
          : { ...t, x: d.origX + dx, y: d.origY + dy }) })
      }
    }
    const onUp = (e: PointerEvent) => {
      const s = selDragRef.current
      if (s?.box) {
        const { doc: cur } = liveRef.current
        const box = s.box
        const moved = cur.strokes.map((stroke, i) => {
          const orig = s.originals.get(i)
          return orig ? scaleStroke(stroke, orig, s.origBox, box) : stroke
        })
        // Released over another canvas: hand the ink over and delete it here. Two
        // documents change, so it costs one undo step on each side.
        const el = containerRef.current
        const drop = s.handle === "move" ? dropTargetAt(e.clientX, e.clientY, el) : null
        if (drop && el) {
          commit({ ...cur, strokes: moved.filter((_, i) => !s.originals.has(i)) })
          drop(moved.filter((_, i) => s.originals.has(i)), { rect: el.getBoundingClientRect(), scale: scaleRef.current, offset: offsetRef.current })
          clearSelection()
        } else {
          commit({ ...cur, strokes: moved })
        }
      }
      setDragPreview(null)
      dragRef.current = null
      selDragRef.current = null
      gestureSnapped.current = false
    }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
    }
  }, [snapshotOnce, commit, clearSelection])

  // Bounds of the frozen lasso — positions the resize handle and the action buttons
  const selectionBounds = selectionLasso ? polyBounds(selectionLasso) : null

  const toolbar = (
      <div className={cn("flex items-center gap-1 shrink-0 min-w-0", toolbarSlot ? "flex-1" : "px-2 py-1.5 border-b border-divider/50")}>
        <div className="relative" ref={addMenuRef}>
          <button
            onClick={() => setAddMenuOpen((o) => !o)}
            className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium text-ink-muted hover:text-ink hover:bg-hover transition-colors"
          >
            <Plus className="h-3 w-3" />
          </button>
          {addMenuOpen && (
            <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] py-1 min-w-[160px] max-h-[60vh] overflow-y-auto">
              <button
                onClick={() => { addPage(); setAddMenuOpen(false) }}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors"
              >
                Page
              </button>
              {(availableImages ?? []).length > 0 && (
                <>
                  <div className="mx-2 my-1 border-t border-divider/50" />
                  <div className="px-3 py-0.5 text-[9px] text-ink-faint uppercase tracking-wider">Images</div>
                  {availableImages!.map((img) => (
                    <button
                      key={img.path}
                      onClick={() => addImageFrame(img.path)}
                      className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors truncate"
                    >
                      <ImageIcon className="h-3 w-3 shrink-0 text-ink-faint" />
                      <span className="truncate">{img.name}</span>
                    </button>
                  ))}
                </>
              )}
              {(availablePdfs ?? []).length > 0 && (
                <>
                  <div className="mx-2 my-1 border-t border-divider/50" />
                  <div className="px-3 py-0.5 text-[9px] text-ink-faint uppercase tracking-wider">PDFs</div>
                  {availablePdfs!.map((pdf) => (
                    <button
                      key={pdf.path}
                      onClick={() => addAttachment(pdf.path)}
                      className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors truncate"
                    >
                      <FileType className="h-3 w-3 shrink-0 text-ink-faint" />
                      <span className="truncate">{pdf.name}</span>
                    </button>
                  ))}
                </>
              )}
              <div className="mx-2 my-1 border-t border-divider/50" />
              {/* section label only earns its place once there is a list under it */}
              {architectureGraphs.length > 0 && (
                <div className="px-3 py-0.5 text-[9px] text-ink-faint uppercase tracking-wider">Architecture Graphs</div>
              )}
              <button
                onClick={createArchitectureGraphWindow}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors"
              >
                <Plus className="h-3 w-3 shrink-0 text-ink-faint" />
                <span>New Architecture Graph</span>
              </button>
              {architectureGraphs.map((graph) => (
                <button
                  key={graph.path}
                  onClick={() => addArchitectureGraph(graph.path)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors truncate"
                >
                  <FileType className="h-3 w-3 shrink-0 text-ink-faint" />
                  <span className="truncate">{graph.name.replace(/\.architecture\.yaml$/, "")}</span>
                </button>
              ))}
              <div className="mx-2 my-1 border-t border-divider/50" />
              <div className="px-3 py-0.5 text-[9px] text-ink-faint uppercase tracking-wider">Canvases</div>
              <button
                onClick={addCanvasWindow}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors"
              >
                <SquarePen className="h-3 w-3 shrink-0 text-ink-faint" />
                <span className="truncate">New canvas</span>
              </button>
              {projectCanvases.map((cv) => (
                <button
                  key={cv.path}
                  onClick={() => addCanvasFile(cv.path)}
                  className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors truncate"
                >
                  <PenLine className="h-3 w-3 shrink-0 text-ink-faint" />
                  <span className="truncate">{cv.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={undo}
          disabled={undoStack.length === 0}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
        >
          <Undo2 className="h-3.5 w-3.5" />
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={redo}
          disabled={redoStack.length === 0}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
        >
          <Redo2 className="h-3.5 w-3.5" />
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={nextSize}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" className="shrink-0">
            <circle cx="5" cy="5" r={strokeWidth === "thin" ? 1.5 : strokeWidth === "medium" ? 2.5 : 4} fill="currentColor" />
          </svg>
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <div className="relative" ref={colorRef}>
          <button
            onClick={() => setColorOpen((o) => !o)}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-[10px] font-medium hover:bg-hover transition-colors"
            style={{ color: inkColor(color, isDark) }}
          >
            Color
          </button>
          {colorOpen && (
            <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] p-2 min-w-[100px]">
              <div className="grid grid-cols-4 gap-1 mb-1.5">
                {PRESET_COLORS.map((c) => (
                  <button
                    key={c.value}
                    onClick={() => pickColor(c.value)}
                    className={cn(
                      "w-5 h-5 rounded-full border-2 transition-all",
                      color === c.value ? "border-ink scale-110" : "border-transparent hover:border-ink-faint"
                    )}
                    style={{ backgroundColor: inkColor(c.value, isDark) }}
                  />
                ))}
              </div>
              {recentColors.length > 0 && (
                <>
                  <div className="text-[9px] text-ink-faint uppercase tracking-wider mb-1">Recent</div>
                  <div className="flex gap-1 mb-1.5">
                    {recentColors.map((c) => (
                      <button
                        key={c}
                        onClick={() => pickColor(c)}
                        className={cn(
                          "w-5 h-5 rounded-full border-2 transition-all",
                          color === c ? "border-ink scale-110" : "border-transparent hover:border-ink-faint"
                        )}
                        style={{ backgroundColor: inkColor(c, isDark) }}
                      />
                    ))}
                  </div>
                </>
              )}
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="color"
                  value={color}
                  onFocus={() => { nativePickerOpen.current = true }}
                  onBlur={() => { nativePickerOpen.current = false }}
                  onChange={(e) => pickColor(e.target.value, false)}
                  className="w-5 h-5 rounded border-0 p-0 cursor-pointer bg-transparent [&::-webkit-color-swatch-wrapper]:p-0 [&::-webkit-color-swatch]:rounded [&::-webkit-color-swatch]:border-divider"
                />
                <span className="text-[10px] text-ink-subtle">Custom</span>
              </label>
            </div>
          )}
        </div>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={() => { setSelectionMode((v) => !v); setScreenshotMode(false); setTextMode(false) }}
          className={cn(
            "rounded p-1 transition-colors",
            selectionMode ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover",
          )}
        >
          <CircleDashed className="h-3.5 w-3.5" />
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={() => { setScreenshotMode((v) => !v); setSelectionMode(false); setTextMode(false) }}
          className={cn(
            "rounded p-1 transition-colors",
            screenshotMode ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover",
          )}
        >
          <Camera className="h-3.5 w-3.5" />
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => { setTextMode((v) => !v); setSelectionMode(false); setScreenshotMode(false) }}
          className={cn(
            "rounded p-1 transition-colors",
            textMode ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover",
          )}
        >
          <Type className="h-3.5 w-3.5" />
        </button>
        <div className="flex-1" />
        <span className="text-[10px] text-ink-faint tabular-nums">{Math.round(scale * 100)}%</span>
        {isErasing && <span className="text-[10px] text-ink-faint ml-1">(eraser)</span>}
        {screenshotMode && <span className="text-[10px] text-ink-faint ml-1">(drag to capture)</span>}
        {selectionMode && <span className="text-[10px] text-ink-faint ml-1">(draw a lasso to select)</span>}
        {textMode && <span className="text-[10px] text-ink-faint ml-1">(drag a box to write in)</span>}
        {selectedStrokes.size > 0 && <span className="text-[10px] text-ink-faint ml-1">({selectedStrokes.size} selected — drag, resize or Esc)</span>}
      </div>
  )

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar — inline above the canvas, or portaled into the app header in canvas mode */}
      {toolbarSlot ? createPortal(toolbar, toolbarSlot) : toolbar}

      {/* Canvas area */}
      <div
        ref={containerRef}
        className={cn("flex-1 min-h-0 overflow-hidden relative", isDrawing ? "cursor-none" : isPanning ? "cursor-grabbing" : "cursor-crosshair")}
        style={{ touchAction: "none", WebkitTouchCallout: "none", WebkitUserSelect: "none", userSelect: "none" }}
        onPointerDown={handleContainerPointerDown}
        onPointerMove={handleContainerPointerMove}
        onPointerUp={handleContainerPointerUp}
        onContextMenu={(e) => e.preventDefault()}
      >
        {/* Dot grid background */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
          <defs>
            <pattern id="canvas-dots" x={offset.x % (30 * scale)} y={offset.y % (30 * scale)} width={30 * scale} height={30 * scale} patternUnits="userSpaceOnUse">
              <circle cx={1.2} cy={1.2} r={1.2} fill="var(--ink-faint)" opacity={0.7} />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#canvas-dots)" />
        </svg>

        {/* Transformed canvas layer — frames + strokes share this space so strokes
            always render over frame content and both export identically. */}
        <svg
          ref={svgRef}
          className="absolute inset-0 w-full h-full"
          style={{ touchAction: "none" }}
          onPointerDown={handleSvgPointerDown}
          onPointerMove={handleSvgPointerMove}
          onPointerUp={handleSvgPointerUp}
          onPointerLeave={handleSvgPointerUp}
        >
          <g transform={`translate(${offset.x},${offset.y}) scale(${scale})`}>
            {/* Frames (pages + images) — the CanvasFrame primitive.
                All chrome is sized in `u` (= one screen px) so it stays a
                constant, light size at any zoom. */}
            {doc.frames.map((f) => {
              const u = 1 / scale // one screen pixel in canvas units
              const w = f.width
              const h = f.width * aspectFor(f)
              const rx = 6 * u
              const band = 16 * u           // top drag-band height
              const gripW = 26 * u          // grip handle pill
              const delC = { x: f.x + w - 11 * u, y: f.y + 11 * u } // delete center (inside top-right)
              const arm = 2.75 * u
              return (
                <g key={f.id} className="group/frame">
                  {/* content — clipped to rounded corners, pointer-transparent so you can draw over it */}
                  <clipPath id={`frame-clip-${f.id}`}>
                    <rect x={f.x} y={f.y} width={w} height={h} rx={rx} ry={rx} />
                  </clipPath>
                  <g clipPath={`url(#frame-clip-${f.id})`} style={{ pointerEvents: "none" }}>
                    {f.kind === "page" ? (
                      <rect x={f.x} y={f.y} width={w} height={h} fill="var(--surface-elevated)" />
                    ) : (
                      <image
                        href={fileViewerRawUrl(f.path!)}
                        x={f.x} y={f.y} width={w} height={h}
                        preserveAspectRatio="none"
                        onLoad={(e) => {
                          const img = e.currentTarget as unknown as SVGImageElement & { naturalWidth?: number; naturalHeight?: number }
                          if (img.naturalWidth && img.naturalHeight) setAspect(f.id, img.naturalHeight / img.naturalWidth)
                          else {
                            const probe = new Image()
                            probe.onload = () => { if (probe.naturalWidth) setAspect(f.id, probe.naturalHeight / probe.naturalWidth) }
                            probe.src = fileViewerRawUrl(f.path!)
                          }
                        }}
                      />
                    )}
                  </g>
                  {/* rounded border */}
                  <rect x={f.x} y={f.y} width={w} height={h} rx={rx} ry={rx} fill="none" stroke="var(--divider-strong)" strokeWidth={u} style={{ pointerEvents: "none" }} />

                  {/* top drag-band hit area (move) */}
                  <rect
                    x={f.x} y={f.y} width={w} height={band}
                    fill="transparent" style={{ cursor: "grab" }}
                    onPointerDown={(e) => { e.stopPropagation(); startInteraction(f.id, "frame", "move", e.clientX, e.clientY) }}
                  />
                  {/* grip handle — the minimal drag affordance, dim until hover */}
                  <rect
                    x={f.x + w / 2 - gripW / 2} y={f.y + 2.5 * u} width={gripW} height={3.5 * u} rx={1.75 * u}
                    fill="var(--ink-faint)"
                    className="opacity-50 group-hover/frame:opacity-90 transition-opacity"
                    style={{ pointerEvents: "none" }}
                  />

                  {/* resize handle (bottom-right) — subtle corner ticks */}
                  <g
                    style={{ cursor: "nwse-resize" }}
                    onPointerDown={(e) => { e.stopPropagation(); startInteraction(f.id, "frame", "resize", e.clientX, e.clientY) }}
                  >
                    <rect x={f.x + w - 16 * u} y={f.y + h - 16 * u} width={16 * u} height={16 * u} fill="transparent" />
                    <path
                      d={`M ${f.x + w - 4.5 * u},${f.y + h - 1.5 * u} L ${f.x + w - 1.5 * u},${f.y + h - 4.5 * u} M ${f.x + w - 9 * u},${f.y + h - 1.5 * u} L ${f.x + w - 1.5 * u},${f.y + h - 9 * u}`}
                      stroke="var(--ink-faint)" strokeWidth={1.25 * u} strokeLinecap="round" fill="none"
                      className="opacity-50 group-hover/frame:opacity-90 transition-opacity"
                      style={{ pointerEvents: "none" }}
                    />
                  </g>

                  {/* delete — always visible, grey → red on hover, no background (matches CanvasAttachment) */}
                  <g
                    className="text-ink-faint hover:text-danger transition-colors"
                    style={{ cursor: "pointer" }}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); removeFrame(f.id) }}
                  >
                    <circle cx={delC.x} cy={delC.y} r={8 * u} fill="transparent" />
                    <path
                      d={`M ${delC.x - arm},${delC.y - arm} L ${delC.x + arm},${delC.y + arm} M ${delC.x - arm},${delC.y + arm} L ${delC.x + arm},${delC.y - arm}`}
                      stroke="currentColor" strokeWidth={1.5 * u} strokeLinecap="round" fill="none"
                    />
                  </g>
                </g>
              )
            })}
            {/* Committed strokes */}
            <StrokeLayer strokes={doc.strokes} hidden={dragPreview && selectedStrokes} isDark={isDark} />
            {/* In-progress stroke — d is set imperatively by redrawLive */}
            <path ref={livePathRef} fill={inkColor(color, isDark)} />

            {/* Stroke selection — highlighted outlines + the freehand lasso fill, which is
                what you drag to move them together. One handle at the shape's bottom-right
                scales it diagonally; it is sized in `u` so it stays a constant,
                thumb-findable size at any zoom. */}
            {selectionLasso && selectionBounds && (() => {
              const u = 1 / scale
              const b = selectionBounds
              const hit = 13 * u  // invisible grab area — bigger than the dot you see
              const dot = 5 * u
              // Mid-drag the committed layer hides these, so they are drawn here from the
              // preview — one gesture frame re-renders the selection, not the whole canvas.
              const picked: [number, StrokeData | undefined][] = dragPreview ?? [...selectedStrokes].map((i) => [i, doc.strokes[i]])
              return (
                <>
                  {picked.map(([i, s]) => {
                    const d = s && strokePath(s)
                    if (!d) return null
                    return (
                      <g key={`sel-${i}`}>
                        {dragPreview && <path d={d} fill={inkColor(s!.color, isDark)} />}
                        <path d={d} fill="none" stroke="var(--ink)" strokeWidth={1.5 * u} strokeDasharray={`${3 * u} ${3 * u}`} />
                      </g>
                    )
                  })}
                  <path
                    d={`M ${selectionLasso.map(([x, y]) => `${x},${y}`).join(" L ")} Z`}
                    fill="var(--ink-faint)" fillOpacity={0.08}
                    stroke="var(--ink-muted)" strokeWidth={u} strokeDasharray={`${4 * u} ${4 * u}`}
                    style={{ cursor: "grab", pointerEvents: "all" }}
                    onPointerDown={(e) => { e.stopPropagation(); startSelDrag("move", e.clientX, e.clientY) }}
                  />
                  <g
                    style={{ cursor: "nwse-resize" }}
                    onPointerDown={(e) => { e.stopPropagation(); startSelDrag("resize", e.clientX, e.clientY) }}
                  >
                    <rect x={b.x + b.w - hit / 2} y={b.y + b.h - hit / 2} width={hit} height={hit} fill="transparent" />
                    <rect
                      x={b.x + b.w - dot / 2} y={b.y + b.h - dot / 2} width={dot} height={dot} rx={1.5 * u}
                      fill="var(--paper)" stroke="var(--ink-muted)" strokeWidth={u}
                      style={{ pointerEvents: "none" }}
                    />
                  </g>
                </>
              )
            })()}

            {/* In-progress lasso — pen-button drag traces the selection outline freehand */}
            {lassoPoints && lassoPoints.length > 1 && (
              <path
                d={`M ${lassoPoints.map(([x, y]) => `${x},${y}`).join(" L ")}`}
                fill="none"
                stroke="var(--ink-muted)" strokeWidth={1.5 / scale} strokeDasharray={`${4 / scale} ${4 / scale}`}
                style={{ pointerEvents: "none" }}
              />
            )}
            {/* Rubber-band rect — screenshot capture area */}
            {shotRect && (
              <rect
                x={Math.min(shotRect.x0, shotRect.x1)} y={Math.min(shotRect.y0, shotRect.y1)}
                width={Math.abs(shotRect.x1 - shotRect.x0)} height={Math.abs(shotRect.y1 - shotRect.y0)}
                fill="var(--ink)" fillOpacity={0.06}
                stroke="var(--ink)" strokeWidth={1.5 / scale} strokeDasharray={`${5 / scale} ${3 / scale}`}
                style={{ pointerEvents: "none" }}
              />
            )}
            {/* Rubber-band rect — text box being dragged out */}
            {textDragRect && (
              <rect
                x={Math.min(textDragRect.x0, textDragRect.x1)} y={Math.min(textDragRect.y0, textDragRect.y1)}
                width={Math.abs(textDragRect.x1 - textDragRect.x0)} height={Math.abs(textDragRect.y1 - textDragRect.y0)}
                fill="var(--ink)" fillOpacity={0.06}
                stroke="var(--ink)" strokeWidth={1.5 / scale} strokeDasharray={`${5 / scale} ${3 / scale}`}
                style={{ pointerEvents: "none" }}
              />
            )}
          </g>
        </svg>

        {/* Selection actions — duplicate / delete, anchored just above the lasso's bounds */}
        {selectionBounds && (
          <div
            className="absolute z-10 flex items-center gap-1 -translate-y-full"
            style={{
              left: selectionBounds.x * scale + offset.x,
              top: selectionBounds.y * scale + offset.y - 6,
            }}
          >
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); copySelection() }}
              className="rounded-full p-1 bg-paper border border-divider-strong text-ink-faint hover:text-ink shadow-[var(--shadow-lg)] transition-colors"
            >
              <Copy className="h-3 w-3" />
            </button>
            <button
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => { e.stopPropagation(); deleteSelection() }}
              className="rounded-full p-1 bg-paper border border-divider-strong text-ink-faint hover:text-danger shadow-[var(--shadow-lg)] transition-colors"
            >
              <Trash2 className="h-3 w-3" />
            </button>
          </div>
        )}

        {/* Attachments (pdf) — the CanvasAttachment primitive. Live, scrollable,
            positioned HTML over the SVG, never baked into export. */}
        {doc.attachments.map((att) => {
          const screenX = att.x * scale + offset.x
          const screenY = att.y * scale + offset.y
          const nested = att.kind === "canvas"
          const architectureGraph = att.kind === "architecture-graph"
          const aspect = nested ? CANVAS_ASPECT : (aspects[att.id] ?? FALLBACK_ASPECT)
          const name = att.path?.split("/").pop() ?? (architectureGraph ? "Architecture Graph" : nested ? "Canvas" : "PDF")
          // Visible size follows canvas zoom, but layout width only follows the
          // settled zoom (capped at what PdfViewer will actually rasterize); the
          // CSS transform bridges the difference. Mid-gesture that means pure
          // bitmap scaling, at rest transform ≈ 1 and the PDF is sharp.
          // ponytail: a nested canvas must not sit under a CSS transform — its pointer
          // math works in layout px, so a scaled wrapper offsets every stroke. It lays
          // out at screen size instead (transform 1); PDFs keep the trick for sharpness.
          const screenW = att.width * scale
          const screenH = (att.height ?? att.width * CANVAS_ASPECT) * scale
          const layoutW = nested || architectureGraph ? screenW : Math.min(att.width * settledScale, PDF_LAYOUT_CAP)
          // Fullscreen only restyles this same wrapper — moving the window elsewhere in
          // the tree would remount the editor inside it and lose whatever it holds.
          const full = fullscreenId === att.id
          return (
            <div
              key={att.id}
              className={cn(
                "absolute flex flex-col border border-divider-strong rounded-lg overflow-hidden bg-paper shadow-[var(--shadow-lg)]",
                full && "z-20",
              )}
              style={full
                ? { inset: 0, transform: "none" }
                : architectureGraph
                  ? { left: screenX, top: screenY, width: screenW, height: screenH, transform: "none" }
                : { left: screenX, top: screenY, width: layoutW, transform: `scale(${screenW / layoutW})`, transformOrigin: "top left" }}
            >
              <div
                className={cn(
                  "flex items-center gap-1.5 px-2 py-1 border-b border-divider/50 select-none shrink-0",
                  !full && "cursor-grab active:cursor-grabbing",
                )}
                onPointerDown={(e) => { e.stopPropagation(); if (!full) startInteraction(att.id, "attachment", "move", e.clientX, e.clientY) }}
              >
                {nested ? <SquarePen className="h-3 w-3 text-ink-faint shrink-0" /> : <FileType className="h-3 w-3 text-ink-faint shrink-0" />}
                <span className="flex-1 min-w-0 truncate text-[10px] font-medium text-ink-muted">{name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); setFullscreenId(full ? null : att.id) }}
                  className="rounded p-0.5 text-ink-faint hover:text-ink transition-colors shrink-0"
                >
                  {full ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); removeAttachment(att.id) }}
                  className="rounded p-0.5 text-ink-faint hover:text-danger transition-colors shrink-0"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              <div
                data-canvas-attachment
                className={cn(full && "flex-1 min-h-0")}
                style={{ height: full ? undefined : architectureGraph ? screenH - 31 : layoutW * aspect }}
                onPointerDown={(e) => e.stopPropagation()}
              >
                {architectureGraph ? (
                  <CanvasArchitectureGraph path={att.path!} />
                ) : nested ? (
                  att.path ? (
                    depth < MAX_NEST_DEPTH ? (
                      <NestedCanvasFile
                        path={att.path}
                        slug={slug}
                        depth={depth + 1}
                        availablePdfs={availablePdfs}
                        availableImages={availableImages}
                      />
                    ) : (
                      <div className="flex h-full items-center justify-center px-3 text-center text-[10px] text-ink-faint">
                        {name} — open it directly to edit
                      </div>
                    )
                  ) : (
                    <CanvasEditor
                      doc={att.canvas ?? EMPTY_NESTED}
                      onChange={(d) => updateNestedCanvas(att.id, d)}
                      slug={slug}
                      depth={depth + 1}
                      availablePdfs={availablePdfs}
                      availableImages={availableImages}
                    />
                  )
                ) : (
                  <PdfViewer
                    url={fileViewerRawUrl(att.path!)}
                    onPageAspect={(r) => setAspect(att.id, r)}
                  />
                )}
              </div>
              {!full && (
                <div
                  className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize"
                  onPointerDown={(e) => { e.stopPropagation(); startInteraction(att.id, "attachment", "resize", e.clientX, e.clientY) }}
                >
                  <svg className="w-full h-full text-ink-faint" viewBox="0 0 16 16">
                    <path d="M14 2L2 14M14 6L6 14M14 10L10 14" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  </svg>
                </div>
              )}
            </div>
          )
        })}

        {/* Text notes — the CanvasText primitive. A draggable/resizable box with a real
            textarea inside, so clicking always focuses natively at the click position.
            Rasterized (word-wrapped, clipped) into export, unlike CanvasAttachment. */}
        {doc.texts.map((t) => {
          const screenX = t.x * scale + offset.x
          const screenY = t.y * scale + offset.y
          const screenW = t.width * scale
          const screenH = t.height * scale
          const band = 6
          return (
            <div key={t.id} className="absolute group/text" style={{ left: screenX, top: screenY, width: screenW, height: screenH }}>
              <textarea
                autoFocus={t.id === autoFocusId}
                value={t.text}
                onChange={(e) => {
                  if (!textEdited.current) { textEdited.current = true; if (t.text !== "") snapshot() }
                  onChange({ ...doc, texts: doc.texts.map((x) => x.id === t.id ? { ...x, text: e.target.value } : x) })
                }}
                onFocus={() => { textEdited.current = false; setAutoFocusId((cur) => cur === t.id ? null : cur) }}
                onBlur={() => dropIfEmptyText(t.id)}
                onKeyDown={(e) => {
                  e.stopPropagation()
                  if (e.key === "Escape") { e.preventDefault(); (e.target as HTMLTextAreaElement).blur() }
                }}
                onPointerDown={(e) => e.stopPropagation()}
                className="absolute inset-0 resize-none bg-transparent outline-none border border-transparent focus:border-dashed focus:border-divider-strong rounded px-1"
                style={{
                  fontFamily: "var(--font-handwriting), cursive",
                  fontSize: t.size * scale,
                  lineHeight: 1.2,
                  color: inkColor(t.color, isDark),
                }}
              />
              {/* top drag-band (move) — thin sliver above the textarea, dim until hover */}
              <div
                className="absolute inset-x-0 top-0 opacity-0 group-hover/text:opacity-60 hover:!opacity-100 transition-opacity bg-[var(--ink-faint)] rounded-t"
                style={{ height: band, cursor: "grab" }}
                onPointerDown={(e) => { e.stopPropagation(); startInteraction(t.id, "text", "move", e.clientX, e.clientY) }}
              />
              {/* resize handle (bottom-right) — broadens/elongates the box */}
              <div
                className="absolute bottom-0 right-0 w-3 h-3 opacity-0 group-hover/text:opacity-100 transition-opacity cursor-nwse-resize"
                onPointerDown={(e) => { e.stopPropagation(); startInteraction(t.id, "text", "resize", e.clientX, e.clientY) }}
              >
                <svg className="w-full h-full text-ink-faint" viewBox="0 0 16 16">
                  <path d="M14 2L2 14M14 6L6 14M14 10L10 14" stroke="currentColor" strokeWidth="1.5" fill="none" />
                </svg>
              </div>
            </div>
          )
        })}

        {/* Long-press context menu */}
        {contextMenu && (
          <div
            className="absolute z-20 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] py-1 min-w-[100px]"
            style={{ left: contextMenu.screenX - (containerRef.current?.getBoundingClientRect().left ?? 0), top: contextMenu.screenY - (containerRef.current?.getBoundingClientRect().top ?? 0) }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => {
                const { canvasX, canvasY } = contextMenu
                setContextMenu(null)
                // discard the tiny hold-dot stroke
                setIsDrawing(false)
                currentPointsRef.current = []
                redrawLive()
                cancelStraighten()
                pasteImageAtPoint(canvasX, canvasY)
              }}
              disabled={!slug}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors disabled:opacity-30"
            >
              <ImageIcon className="h-3 w-3 shrink-0 text-ink-faint" />
              Paste image
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// A graph attachment owns only its canvas-local frame; graph content continues to
// live in the canonical YAML file. It is deliberately rendered at real layout size:
// CSS scaling would desynchronise React Flow's handles from the pointer.
function CanvasArchitectureGraph({ path }: { path: string }) {
  const name = path.split("/").pop() ?? path
  const [graph, setGraph] = useState<ArchitectureGraph | null>(null)
  const write = useCallback((content: string) => writeArchitectureGraph(name, content), [name])
  const { queue, markSaved } = useGraphAutosave({ key: name, write })

  useEffect(() => {
    let alive = true
    setGraph(null)
    readArchitectureGraph(name).then((document) => {
      if (!alive) return
      const loaded = parseArchitectureGraph(document.content)
      markSaved(serializeArchitectureGraph(loaded))
      setGraph(loaded)
    }).catch(() => { if (alive) setGraph(null) })
    return () => { alive = false }
  }, [name, markSaved])

  const onChange = useCallback((next: ArchitectureGraph) => {
    setGraph(next)
    queue(serializeArchitectureGraph(next))
  }, [queue])

  if (!graph) return <div className="flex h-full items-center justify-center text-[10px] text-ink-faint">Loading Architecture Graph…</div>
  return <ArchitectureGraphSurface graph={graph} onChange={onChange} className="h-full" />
}

// A canvas window bound to an existing project `.canvas` file: loads it on open,
// writes back a second after you stop drawing, flushes on close. It owns the file
// outright — nothing but the path is stored in the canvas holding the window.
// ponytail: single-writer assumption, same as DocumentEditor. No dirty flag, no
// conflict detection; add both together if two surfaces ever edit one canvas at once.
function NestedCanvasFile({ path, slug, depth, availablePdfs, availableImages }: {
  path: string
  slug?: string
  depth: number
  availablePdfs?: { path: string; name: string }[]
  availableImages?: { path: string; name: string }[]
}) {
  const [doc, setDoc] = useState<CanvasDocument | null>(null)
  const savedRef = useRef<string | null>(null)
  const liveRef = useRef<CanvasDocument | null>(null)
  liveRef.current = doc

  useEffect(() => {
    let alive = true
    setDoc(null)
    savedRef.current = null
    loadFileViewerText(path).then((text) => {
      if (!alive) return
      const loaded = text.trim() ? parseCanvasDoc(text) : emptyCanvasDoc()
      savedRef.current = serializeCanvasDoc(loaded)
      setDoc(loaded)
    }).catch(() => { if (alive) setDoc(emptyCanvasDoc()) })
    return () => { alive = false }
  }, [path])

  const save = useCallback(() => {
    const live = liveRef.current
    // savedRef is null until the load lands — writing before that would persist an
    // empty document over the real one
    if (!live || !slug || savedRef.current === null) return
    const serialized = serializeCanvasDoc(live)
    if (serialized === savedRef.current) return
    savedRef.current = serialized
    writeDocument(slug, path.split("/").pop()!, serialized).catch(() => {})
  }, [slug, path])

  useEffect(() => {
    if (!doc) return
    const t = setTimeout(save, NESTED_SAVE_MS)
    return () => clearTimeout(t)
  }, [doc, save])

  // Closing the window unmounts us; the debounce cleanup alone would drop whatever was
  // drawn in the last second.
  useEffect(() => () => save(), [save])

  if (!doc) return <div className="flex h-full items-center justify-center text-[10px] text-ink-faint">Loading…</div>
  return (
    <CanvasEditor
      doc={doc}
      onChange={setDoc}
      slug={slug}
      depth={depth}
      docPath={path}
      availablePdfs={availablePdfs}
      availableImages={availableImages}
    />
  )
}
