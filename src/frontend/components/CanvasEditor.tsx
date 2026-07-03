"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { getStroke } from "perfect-freehand"
import { type StrokeData, type EmbedRect, getSvgPathFromStroke, renderPageToPng } from "@/lib/drawing"
import { fileViewerRawUrl, writeBinaryDocument } from "@/lib/api"
import { activeThemeName } from "@/lib/palette"
import { cn } from "@/lib/utils"
import { Plus, Undo2, Redo2, Trash2, FileType, ImageIcon, X, Camera, CircleDashed } from "lucide-react"
import { PdfViewer } from "./PdfViewer"

const A4_W = 794
const A4_H = 1123
const A4_ASPECT = A4_H / A4_W // height / width — page frames are locked to this
const PAGE_GAP = 40
const DEFAULT_PDF_WIDTH = 500 // attachment default width (canvas units)
const DEFAULT_IMAGE_WIDTH = 400 // image-frame default width (canvas units)
const MIN_FRAME_WIDTH = 120
const MIN_ATTACH_WIDTH = 200
const FALLBACK_ASPECT = 1.3 // height/width used until the real aspect is measured
const MIN_SCALE = 0.15
const MAX_SCALE = 3
const THIN_WIDTH = 2
const MEDIUM_WIDTH = 5
const THICK_WIDTH = 10
const PEN_HOLD_MS = 400 // press-and-hold the pen button to activate selection mode
const PEN_DOUBLE_CLICK_MS = 350 // double-click the pen button to activate screenshot mode

const STROKE_OPTIONS = {
  smoothing: 0.5,
  streamline: 0.5,
  simulatePressure: true,
  last: true,
} as const

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
// CanvasAttachment = a live, scrollable file reference (PDF; later text/LaTeX).
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
  kind: "pdf"
  path: string
  x: number
  y: number
  width: number
}

export interface CanvasDocument {
  version: 1
  viewport?: { scale: number; centerX: number; centerY: number }
  frames: CanvasFrame[]
  strokes: StrokeData[]
  attachments: CanvasAttachment[]
}

export function emptyCanvasDoc(): CanvasDocument {
  return { version: 1, frames: [], strokes: [], attachments: [] }
}

// Legacy on-disk shape (pages / pdfEmbeds / imageEmbeds) — migrated on load.
type LegacyEmbed = { id: string; path: string; x: number; y: number; width: number }
type LegacyDoc = {
  version: 1
  viewport?: CanvasDocument["viewport"]
  strokes?: StrokeData[]
  frames?: CanvasFrame[]
  attachments?: CanvasAttachment[]
  pages?: { id: string; x: number; y: number }[]
  pdfEmbeds?: LegacyEmbed[]
  imageEmbeds?: LegacyEmbed[]
}

function migrate(d: LegacyDoc): CanvasDocument {
  if (Array.isArray(d.frames)) {
    return { version: 1, viewport: d.viewport, frames: d.frames, strokes: d.strokes ?? [], attachments: d.attachments ?? [] }
  }
  const frames: CanvasFrame[] = []
  for (const p of d.pages ?? []) frames.push({ id: p.id, kind: "page", x: p.x, y: p.y, width: A4_W })
  for (const im of d.imageEmbeds ?? []) frames.push({ id: im.id, kind: "image", x: im.x, y: im.y, width: im.width, path: im.path })
  const attachments: CanvasAttachment[] = (d.pdfEmbeds ?? []).map((e) => ({ id: e.id, kind: "pdf", x: e.x, y: e.y, width: e.width, path: e.path }))
  return { version: 1, viewport: d.viewport, frames, strokes: d.strokes ?? [], attachments }
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
}

// convert between center (canvas-space point at view center) and offset (SVG translate)
function centerToOffset(cx: number, cy: number, s: number, w: number, h: number) {
  return { x: w / 2 - cx * s, y: h / 2 - cy * s }
}
function offsetToCenter(ox: number, oy: number, s: number, w: number, h: number) {
  return { cx: (w / 2 - ox) / s, cy: (h / 2 - oy) / s }
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

export function CanvasEditor({ doc, onChange, availablePdfs, availableImages, slug, onImageAdded }: CanvasEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const initScale = doc.viewport?.scale ?? 0.5
  // offset is derived once the container mounts; fall back to a reasonable default
  const [scale, setScale] = useState(initScale)
  const [offset, setOffset] = useState({ x: 40, y: 40 })
  const scaleRef = useRef(scale)
  const offsetRef = useRef(offset)
  scaleRef.current = scale
  offsetRef.current = offset

  // on mount (and resize), recompute offset from the stored center so the same
  // canvas point stays centered regardless of container dimensions
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
  const [currentPoints, setCurrentPoints] = useState<number[][]>([])
  const [isErasing, setIsErasing] = useState(false)
  const [strokeWidth, setStrokeWidth] = useState<StrokeSize>("thin")
  const [color, setColor] = useState("#000000")
  const [recentColors, setRecentColors] = useState<string[]>([])
  const [colorOpen, setColorOpen] = useState(false)
  const colorRef = useRef<HTMLDivElement>(null)

  const [redoStack, setRedoStack] = useState<StrokeData[]>([])

  const baseWidth = SIZE_MAP[strokeWidth]

  // --- Select & move strokes. Draw a freehand lasso (any stroke fully enclosed by
  // the loop is picked) — not a rectangle, so any shaped area works. Activated either
  // via the toolbar icon or by press-and-holding the pen's side button (which used to
  // pan the canvas — panning is still available via space+drag / touch). ---
  type RectDrag = { x0: number; y0: number; x1: number; y1: number }
  const [selectionMode, setSelectionMode] = useState(false)
  const [lassoPoints, setLassoPoints] = useState<[number, number][] | null>(null) // in-progress lasso being drawn
  const lassoActive = useRef(false)
  const [selectionLasso, setSelectionLasso] = useState<[number, number][] | null>(null) // frozen shape backing the current selection
  const [selectedStrokes, setSelectedStrokes] = useState<Set<number>>(new Set())
  const selDragRef = useRef<{ startX: number; startY: number; originals: Map<number, number[][]>; origLasso: [number, number][] } | null>(null)

  const clearSelection = useCallback(() => {
    setSelectedStrokes((prev) => (prev.size === 0 ? prev : new Set()))
    setSelectionLasso(null)
  }, [])

  // Pen side-button gesture: hold => activate selection mode, double-click => activate screenshot mode
  const penHoldTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const penHoldFired = useRef(false)
  const penClickPending = useRef(false)
  const penClickTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  // --- Screenshot: drag a rect, release copies the rasterized region to clipboard ---
  const [screenshotMode, setScreenshotMode] = useState(false)
  const [shotRect, setShotRect] = useState<RectDrag | null>(null)
  const shotStart = useRef<{ x: number; y: number } | null>(null)

  const [isDark, setIsDark] = useState(() => activeThemeName() === "dark")
  useEffect(() => {
    const obs = new MutationObserver(() => setIsDark(activeThemeName() === "dark"))
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => obs.disconnect()
  }, [])

  // ponytail: swap black↔white for display only, stored color stays unchanged
  const displayColor = useCallback((c: string) => {
    if (!isDark) return c
    return c === "#000000" ? "#ffffff" : c
  }, [isDark])

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

  const persistViewport = useCallback((s: number, o: { x: number; y: number }) => {
    const el = containerRef.current
    if (!el) return
    const { cx, cy } = offsetToCenter(o.x, o.y, s, el.clientWidth, el.clientHeight)
    onChange({ ...doc, viewport: { scale: s, centerX: cx, centerY: cy } })
  }, [doc, onChange])

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

  // Test every stroke against the finished lasso loop and freeze the ones fully enclosed
  const finalizeLasso = useCallback((pts: [number, number][] | null) => {
    if (pts && pts.length >= 3) {
      const picked = new Set<number>()
      doc.strokes.forEach((s, i) => {
        if (s.points.length > 0 && s.points.every((p) => pointInPolygon(p[0]!, p[1]!, pts))) picked.add(i)
      })
      setSelectedStrokes(picked)
      setSelectionLasso(picked.size > 0 ? pts : null)
    }
  }, [doc.strokes])

  // --- Two-finger touch: pinch-zoom + pan ---
  const touchRef = useRef<{ dist: number; cx: number; cy: number; scale: number; ox: number; oy: number } | null>(null)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const getTouchMid = (a: Touch, b: Touch) => ({ x: (a.clientX + b.clientX) / 2, y: (a.clientY + b.clientY) / 2 })
    const getTouchDist = (a: Touch, b: Touch) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)

    const onStart = (e: TouchEvent) => {
      if (e.touches.length !== 2) return
      e.preventDefault()
      const [a, b] = [e.touches[0]!, e.touches[1]!]
      const rect = el.getBoundingClientRect()
      const mid = getTouchMid(a, b)
      touchRef.current = {
        dist: getTouchDist(a, b),
        cx: mid.x - rect.left,
        cy: mid.y - rect.top,
        scale: scaleRef.current,
        ox: offsetRef.current.x,
        oy: offsetRef.current.y,
      }
    }
    const onMove = (e: TouchEvent) => {
      if (e.touches.length !== 2 || !touchRef.current) return
      e.preventDefault()
      const [a, b] = [e.touches[0]!, e.touches[1]!]
      const rect = el.getBoundingClientRect()
      const mid = getTouchMid(a, b)
      const t = touchRef.current
      const newDist = getTouchDist(a, b)
      const ratio = newDist / t.dist
      const next = Math.min(MAX_SCALE, Math.max(MIN_SCALE, t.scale * ratio))
      const scaleRatio = next / t.scale
      const mx = mid.x - rect.left
      const my = mid.y - rect.top
      const panDx = mx - t.cx
      const panDy = my - t.cy
      const newOx = t.cx - (t.cx - t.ox) * scaleRatio + panDx
      const newOy = t.cy - (t.cy - t.oy) * scaleRatio + panDy
      scaleRef.current = next
      offsetRef.current = { x: newOx, y: newOy }
      setScale(next)
      setOffset({ x: newOx, y: newOy })
    }
    const onEnd = () => { touchRef.current = null }

    el.addEventListener("touchstart", onStart, { passive: false })
    el.addEventListener("touchmove", onMove, { passive: false })
    el.addEventListener("touchend", onEnd)
    el.addEventListener("touchcancel", onEnd)
    return () => {
      el.removeEventListener("touchstart", onStart)
      el.removeEventListener("touchmove", onMove)
      el.removeEventListener("touchend", onEnd)
      el.removeEventListener("touchcancel", onEnd)
    }
  }, [])

  // --- Aspect cache (image frames + pdf attachments), keyed by element id ---
  const [aspects, setAspects] = useState<Record<string, number>>({})
  const setAspect = useCallback((id: string, r: number) => {
    setAspects((prev) => (Math.abs((prev[id] ?? 0) - r) < 0.001 ? prev : { ...prev, [id]: r }))
  }, [])
  const aspectFor = useCallback((f: CanvasFrame) => f.kind === "page" ? A4_ASPECT : (aspects[f.id] ?? FALLBACK_ASPECT), [aspects])

  // --- Long-press context menu ---
  const [contextMenu, setContextMenu] = useState<{ screenX: number; screenY: number; canvasX: number; canvasY: number } | null>(null)
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const longPressPos = useRef<{ clientX: number; clientY: number } | null>(null)

  const startLongPress = useCallback((clientX: number, clientY: number) => {
    longPressPos.current = { clientX, clientY }
    longPressTimer.current = setTimeout(() => {
      const [cx, cy] = screenToCanvas(clientX, clientY)
      setIsDrawing(false)
      setCurrentPoints([])
      setContextMenu({ screenX: clientX, screenY: clientY, canvasX: cx, canvasY: cy })
    }, 500)
  }, [screenToCanvas])

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
      const pngBytes = await renderPageToPng(doc.strokes, x, y, w, h, images)
      const blob = new Blob([pngBytes.buffer as ArrayBuffer], { type: "image/png" })
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
    } catch { /* clipboard write can fail without permission — silently drop */ }
  }, [doc.frames, doc.strokes, aspectFor])

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
    // drawing/erasing directly on the canvas counts as "outside" the selection
    clearSelection()
    if (e.button === 5) { setIsErasing(true); return }
    if (e.button !== 0) return
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    setIsDrawing(true)
    const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
    setCurrentPoints([[cx, cy, e.pressure > 0 ? e.pressure : 0.5]])
    startLongPress(e.clientX, e.clientY)
  }, [screenToCanvas, startLongPress, screenshotMode, selectionMode, clearSelection])

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
        for (const pt of stroke.points) {
          const dx = pt[0]! - ex
          const dy = pt[1]! - ey
          if (Math.sqrt(dx * dx + dy * dy) < threshold) return false
        }
        return true
      })
      if (updated.length !== doc.strokes.length) {
        onChange({ ...doc, strokes: updated })
      }
      return
    }
    if (!isDrawing) return
    const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
    setCurrentPoints((prev) => [...prev, [cx, cy, e.pressure > 0 ? e.pressure : 0.5]])
  }, [isDrawing, isErasing, screenToCanvas, doc, onChange, cancelLongPress])

  const handleSvgPointerUp = useCallback((e?: React.PointerEvent<SVGSVGElement>) => {
    cancelLongPress()
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
    if (isErasing) { setIsErasing(false); return }
    if (!isDrawing) return
    setIsDrawing(false)
    if (currentPoints.length < 2) { setCurrentPoints([]); return }
    onChange({ ...doc, strokes: [...doc.strokes, { points: currentPoints, color, width: baseWidth }] })
    setRedoStack([])
    setCurrentPoints([])
  }, [isDrawing, isErasing, currentPoints, color, baseWidth, doc, onChange, cancelLongPress, captureScreenshot, finalizeLasso])

  // --- Undo / Redo ---
  const undo = useCallback(() => {
    if (doc.strokes.length === 0) return
    setRedoStack((prev) => [...prev, doc.strokes[doc.strokes.length - 1]!])
    onChange({ ...doc, strokes: doc.strokes.slice(0, -1) })
  }, [doc, onChange])

  const redo = useCallback(() => {
    if (redoStack.length === 0) return
    const stroke = redoStack[redoStack.length - 1]!
    setRedoStack((prev) => prev.slice(0, -1))
    onChange({ ...doc, strokes: [...doc.strokes, stroke] })
  }, [doc, onChange, redoStack])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault()
        undo()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "y") {
        e.preventDefault()
        redo()
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [undo, redo])

  const clearAll = useCallback(() => {
    setRedoStack([])
    onChange({ ...doc, strokes: [] })
  }, [doc, onChange])

  // Escape cancels an in-progress screenshot/selection-drawing or drops the current stroke selection
  useEffect(() => {
    if (!screenshotMode && !selectionMode && selectedStrokes.size === 0) return
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return
      shotStart.current = null
      setShotRect(null)
      setScreenshotMode(false)
      lassoActive.current = false
      setLassoPoints(null)
      setSelectionMode(false)
      clearSelection()
    }
    document.addEventListener("keydown", esc)
    return () => document.removeEventListener("keydown", esc)
  }, [screenshotMode, selectionMode, selectedStrokes, clearSelection])

  const currentOutline = currentPoints.length >= 2
    ? getSvgPathFromStroke(getStroke(currentPoints, { ...STROKE_OPTIONS, size: baseWidth, last: false }))
    : ""

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

  // latest doc/onChange for the long-lived window drag listeners
  const liveRef = useRef({ doc, onChange })
  liveRef.current = { doc, onChange }

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
    onChange({ ...doc, frames: [...doc.frames, { id: `p-${Date.now()}`, kind: "page", x: newX, y: last?.y ?? 0, width: A4_W }] })
  }, [doc, onChange])

  const addImageFrameAt = useCallback((path: string, x: number, y: number) => {
    onChange({ ...doc, frames: [...doc.frames, { id: `img-${Date.now()}`, kind: "image", x, y, width: DEFAULT_IMAGE_WIDTH, path }] })
  }, [doc, onChange])

  const addImageFrame = useCallback((path: string) => {
    const { cx, cy } = pointAtCenter(DEFAULT_IMAGE_WIDTH, FALLBACK_ASPECT)
    addImageFrameAt(path, cx, cy)
    setAddMenuOpen(false)
  }, [pointAtCenter, addImageFrameAt])

  const removeFrame = useCallback((id: string) => {
    onChange({ ...doc, frames: doc.frames.filter((f) => f.id !== id) })
    setAspects(({ [id]: _gone, ...rest }) => rest)
  }, [doc, onChange])

  // --- Attachments (pdf) ---
  const addAttachment = useCallback((path: string) => {
    if (doc.attachments.some((a) => a.path === path)) { setAddMenuOpen(false); return }
    const { cx, cy } = pointAtCenter(DEFAULT_PDF_WIDTH, FALLBACK_ASPECT)
    onChange({ ...doc, attachments: [...doc.attachments, { id: `pdf-${Date.now()}`, kind: "pdf", path, x: cx, y: cy, width: DEFAULT_PDF_WIDTH }] })
    setAddMenuOpen(false)
  }, [doc, onChange, pointAtCenter])

  const removeAttachment = useCallback((id: string) => {
    onChange({ ...doc, attachments: doc.attachments.filter((a) => a.id !== id) })
    setAspects(({ [id]: _gone, ...rest }) => rest)
  }, [doc, onChange])

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
      if (!slug) return
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
  }, [slug, pointAtCenter, addImageFrameAt, onImageAdded])

  // --- Shared move / resize for frames + attachments ---
  const dragRef = useRef<{
    id: string; target: "frame" | "attachment"; mode: "move" | "resize"
    startX: number; startY: number; origX: number; origY: number; origW: number
  } | null>(null)

  const startInteraction = useCallback((id: string, target: "frame" | "attachment", mode: "move" | "resize", clientX: number, clientY: number) => {
    const list = target === "frame" ? liveRef.current.doc.frames : liveRef.current.doc.attachments
    const item = list.find((e) => e.id === id)
    if (!item) return
    dragRef.current = { id, target, mode, startX: clientX, startY: clientY, origX: item.x, origY: item.y, origW: item.width }
  }, [])

  // Drag the lasso selection to translate every selected stroke together
  const startSelectionMove = useCallback((clientX: number, clientY: number) => {
    const originals = new Map<number, number[][]>()
    for (const i of selectedStrokes) {
      const s = liveRef.current.doc.strokes[i]
      if (s) originals.set(i, s.points.map((p) => [...p]))
    }
    if (originals.size === 0 || !selectionLasso) return
    selDragRef.current = { startX: clientX, startY: clientY, originals, origLasso: selectionLasso.map((p) => [...p] as [number, number]) }
  }, [selectedStrokes, selectionLasso])

  const deleteSelection = useCallback(() => {
    if (selectedStrokes.size === 0) return
    const { doc: cur, onChange: change } = liveRef.current
    change({ ...cur, strokes: cur.strokes.filter((_, i) => !selectedStrokes.has(i)) })
    clearSelection()
  }, [selectedStrokes, clearSelection])

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const s = selDragRef.current
      if (s) {
        const dx = (e.clientX - s.startX) / scaleRef.current
        const dy = (e.clientY - s.startY) / scaleRef.current
        const { doc: cur, onChange: change } = liveRef.current
        change({ ...cur, strokes: cur.strokes.map((stroke, i) => {
          const orig = s.originals.get(i)
          return orig ? { ...stroke, points: orig.map((p) => [p[0]! + dx, p[1]! + dy, p[2] ?? 0.5]) } : stroke
        }) })
        // the lasso boundary (and the trash icon anchored to it) rides along with the strokes
        setSelectionLasso(s.origLasso.map(([x, y]) => [x + dx, y + dy]))
        return
      }
      const d = dragRef.current
      if (!d) return
      const { doc: cur, onChange: change } = liveRef.current
      const dx = (e.clientX - d.startX) / scaleRef.current
      const dy = (e.clientY - d.startY) / scaleRef.current
      const minW = d.target === "frame" ? MIN_FRAME_WIDTH : MIN_ATTACH_WIDTH
      if (d.target === "frame") {
        change({ ...cur, frames: cur.frames.map((f) => f.id !== d.id ? f
          : d.mode === "resize" ? { ...f, width: Math.max(minW, d.origW + dx) }
          : { ...f, x: d.origX + dx, y: d.origY + dy }) })
      } else {
        change({ ...cur, attachments: cur.attachments.map((a) => a.id !== d.id ? a
          : d.mode === "resize" ? { ...a, width: Math.max(minW, d.origW + dx) }
          : { ...a, x: d.origX + dx, y: d.origY + dy }) })
      }
    }
    const onUp = () => { dragRef.current = null; selDragRef.current = null }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
    }
  }, [])

  // Bounding box of the frozen selection lasso, in canvas units — positions the trash icon
  let selectionTop: { x: number; y: number } | null = null
  if (selectionLasso) {
    for (const [x, y] of selectionLasso) {
      if (!selectionTop || y < selectionTop.y) selectionTop = { x, y }
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-divider/50 shrink-0">
        <div className="relative" ref={addMenuRef}>
          <button
            onClick={() => setAddMenuOpen((o) => !o)}
            className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium text-ink-muted hover:text-ink hover:bg-hover transition-colors"
          >
            <Plus className="h-3 w-3" />
          </button>
          {addMenuOpen && (
            <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] py-1 min-w-[160px]">
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
            </div>
          )}
        </div>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={undo}
          disabled={doc.strokes.length === 0}
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
          onClick={clearAll}
          disabled={doc.strokes.length === 0}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
        >
          <Trash2 className="h-3.5 w-3.5" />
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
            style={{ color: displayColor(color) }}
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
                    style={{ backgroundColor: displayColor(c.value) }}
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
                        style={{ backgroundColor: displayColor(c) }}
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
          onClick={() => { setSelectionMode((v) => !v); setScreenshotMode(false) }}
          className={cn(
            "rounded p-1 transition-colors",
            selectionMode ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover",
          )}
        >
          <CircleDashed className="h-3.5 w-3.5" />
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={() => { setScreenshotMode((v) => !v); setSelectionMode(false) }}
          className={cn(
            "rounded p-1 transition-colors",
            screenshotMode ? "text-ink bg-hover" : "text-ink-subtle hover:text-ink hover:bg-hover",
          )}
        >
          <Camera className="h-3.5 w-3.5" />
        </button>
        <div className="flex-1" />
        <span className="text-[10px] text-ink-faint tabular-nums">{Math.round(scale * 100)}%</span>
        {isErasing && <span className="text-[10px] text-ink-faint ml-1">(eraser)</span>}
        {screenshotMode && <span className="text-[10px] text-ink-faint ml-1">(drag to capture)</span>}
        {selectionMode && <span className="text-[10px] text-ink-faint ml-1">(draw a lasso to select)</span>}
        {selectedStrokes.size > 0 && <span className="text-[10px] text-ink-faint ml-1">({selectedStrokes.size} selected — drag or Esc)</span>}
      </div>

      {/* Canvas area */}
      <div
        ref={containerRef}
        className={cn("flex-1 min-h-0 overflow-hidden relative", isPanning ? "cursor-grabbing" : "cursor-crosshair")}
        style={{ touchAction: "none" }}
        onPointerDown={handleContainerPointerDown}
        onPointerMove={handleContainerPointerMove}
        onPointerUp={handleContainerPointerUp}
      >
        {/* Dot grid background */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
          <defs>
            <pattern id="canvas-dots" x={offset.x % (20 * scale)} y={offset.y % (20 * scale)} width={20 * scale} height={20 * scale} patternUnits="userSpaceOnUse">
              <circle cx={1} cy={1} r={0.8} fill="var(--ink-faint)" opacity={0.3} />
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
            {doc.strokes.map((stroke, i) => {
              const outline = getStroke(stroke.points, { ...STROKE_OPTIONS, size: stroke.width })
              const d = getSvgPathFromStroke(outline)
              return d ? <path key={i} d={d} fill={displayColor(stroke.color)} /> : null
            })}
            {/* In-progress stroke */}
            {currentOutline && <path d={currentOutline} fill={displayColor(color)} />}

            {/* Stroke selection — highlighted outlines + a draggable lasso fill (moves them together) */}
            {selectionLasso && (
              <>
                {[...selectedStrokes].map((i) => {
                  const s = doc.strokes[i]
                  if (!s) return null
                  const d = getSvgPathFromStroke(getStroke(s.points, { ...STROKE_OPTIONS, size: s.width }))
                  return d ? <path key={`sel-${i}`} d={d} fill="none" stroke="var(--ink)" strokeWidth={1.5 / scale} strokeDasharray={`${3 / scale} ${3 / scale}`} /> : null
                })}
                <path
                  d={`M ${selectionLasso.map(([x, y]) => `${x},${y}`).join(" L ")} Z`}
                  fill="var(--ink-faint)" fillOpacity={0.08}
                  stroke="var(--ink-muted)" strokeWidth={1 / scale} strokeDasharray={`${4 / scale} ${4 / scale}`}
                  style={{ cursor: "grab", pointerEvents: "all" }}
                  onPointerDown={(e) => { e.stopPropagation(); startSelectionMove(e.clientX, e.clientY) }}
                />
              </>
            )}

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
          </g>
        </svg>

        {/* Selection delete — hover-responsive trash icon at the topmost point of the lasso */}
        {selectionTop && (
          <button
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => { e.stopPropagation(); deleteSelection() }}
            className="absolute z-10 -translate-x-1/2 -translate-y-full rounded-full p-1 bg-paper border border-divider-strong text-ink-faint hover:text-danger shadow-[var(--shadow-lg)] transition-colors"
            style={{
              left: selectionTop.x * scale + offset.x,
              top: selectionTop.y * scale + offset.y - 6,
            }}
          >
            <Trash2 className="h-3 w-3" />
          </button>
        )}

        {/* Attachments (pdf) — the CanvasAttachment primitive. Live, scrollable,
            positioned HTML over the SVG, never baked into export. */}
        {doc.attachments.map((att) => {
          const screenX = att.x * scale + offset.x
          const screenY = att.y * scale + offset.y
          const screenW = att.width * scale
          const aspect = aspects[att.id] ?? FALLBACK_ASPECT
          const name = att.path.split("/").pop() ?? "PDF"
          return (
            <div
              key={att.id}
              className="absolute flex flex-col border border-divider-strong rounded-lg overflow-hidden bg-paper shadow-[var(--shadow-lg)]"
              style={{ left: screenX, top: screenY, width: screenW }}
            >
              <div
                className="flex items-center gap-1.5 px-2 py-1 border-b border-divider/50 cursor-grab active:cursor-grabbing select-none shrink-0"
                onPointerDown={(e) => { e.stopPropagation(); startInteraction(att.id, "attachment", "move", e.clientX, e.clientY) }}
              >
                <FileType className="h-3 w-3 text-ink-faint shrink-0" />
                <span className="flex-1 min-w-0 truncate text-[10px] font-medium text-ink-muted">{name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); removeAttachment(att.id) }}
                  className="rounded p-0.5 text-ink-faint hover:text-danger transition-colors shrink-0"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              <div style={{ height: screenW * aspect }} onPointerDown={(e) => e.stopPropagation()}>
                <PdfViewer
                  url={fileViewerRawUrl(att.path)}
                  onPageAspect={(r) => setAspect(att.id, r)}
                />
              </div>
              <div
                className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize"
                onPointerDown={(e) => { e.stopPropagation(); startInteraction(att.id, "attachment", "resize", e.clientX, e.clientY) }}
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
                setCurrentPoints([])
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
