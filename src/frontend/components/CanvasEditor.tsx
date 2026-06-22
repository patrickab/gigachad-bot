"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { getStroke } from "perfect-freehand"
import { type StrokeData, getSvgPathFromStroke } from "@/lib/drawing"
import { fileViewerRawUrl } from "@/lib/api"
import { activeThemeName } from "@/lib/palette"
import { cn } from "@/lib/utils"
import { Plus, Undo2, Redo2, Trash2, FileType, X } from "lucide-react"
import { PdfViewer } from "./PdfViewer"

const A4_W = 794
const A4_H = 1123
const PAGE_GAP = 40
const DEFAULT_EMBED_WIDTH = 500 // canvas units; height follows the PDF's own aspect
const MIN_EMBED_WIDTH = 200
const FALLBACK_EMBED_ASPECT = 1.3 // height/width used until the real page aspect is measured
const MIN_SCALE = 0.15
const MAX_SCALE = 3
const THIN_WIDTH = 2
const MEDIUM_WIDTH = 5
const THICK_WIDTH = 10

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

export interface CanvasPage {
  id: string
  x: number
  y: number
}

export interface PdfEmbed {
  id: string
  path: string
  x: number
  y: number
  width: number
}

export interface ImageEmbed {
  id: string
  path: string
  x: number
  y: number
  width: number
}

export interface CanvasDocument {
  version: 1
  viewport?: { scale: number; centerX: number; centerY: number }
  pages: CanvasPage[]
  strokes: StrokeData[]
  pdfEmbeds?: PdfEmbed[]
  imageEmbeds?: ImageEmbed[]
}

export function emptyCanvasDoc(): CanvasDocument {
  return {
    version: 1,
    pages: [],
    strokes: [],
  }
}

export function parseCanvasDoc(text: string): CanvasDocument {
  try {
    const parsed = JSON.parse(text)
    if (parsed.version === 1 && Array.isArray(parsed.pages)) return parsed
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
}

// convert between center (canvas-space point at view center) and offset (SVG translate)
function centerToOffset(cx: number, cy: number, s: number, w: number, h: number) {
  return { x: w / 2 - cx * s, y: h / 2 - cy * s }
}
function offsetToCenter(ox: number, oy: number, s: number, w: number, h: number) {
  return { cx: (w / 2 - ox) / s, cy: (h / 2 - oy) / s }
}

export function CanvasEditor({ doc, onChange, availablePdfs }: CanvasEditorProps) {
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
    if (e.pointerType === "touch") return
    if (e.button === 1 || (e.button === 0 && spaceDown.current)) {
      e.preventDefault()
      setIsPanning(true)
      panStart.current = { x: e.clientX, y: e.clientY, ox: offsetRef.current.x, oy: offsetRef.current.y }
      ;(e.target as Element).setPointerCapture(e.pointerId)
    }
  }, [])

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
    if (isPanning) setIsPanning(false)
  }, [isPanning])

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

  // --- Drawing (on SVG) ---
  const handleSvgPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (spaceDown.current) return
    if (e.button === 5) { setIsErasing(true); return }
    if (e.button !== 0) return
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    setIsDrawing(true)
    const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
    setCurrentPoints([[cx, cy, e.pressure > 0 ? e.pressure : 0.5]])
  }, [screenToCanvas])

  const handleSvgPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
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
  }, [isDrawing, isErasing, screenToCanvas, doc, onChange])

  const handleSvgPointerUp = useCallback((e?: React.PointerEvent<SVGSVGElement>) => {
    if (e?.pointerType === "touch") return
    if (isErasing) { setIsErasing(false); return }
    if (!isDrawing) return
    setIsDrawing(false)
    if (currentPoints.length < 2) { setCurrentPoints([]); return }
    onChange({ ...doc, strokes: [...doc.strokes, { points: currentPoints, color, width: baseWidth }] })
    setRedoStack([])
    setCurrentPoints([])
  }, [isDrawing, isErasing, currentPoints, color, baseWidth, doc, onChange])

  // --- Undo / Redo ---
  const undo = useCallback(() => {
    if (doc.strokes.length === 0) return
    setRedoStack((prev) => [...prev, doc.strokes[doc.strokes.length - 1]!])
    onChange({ ...doc, strokes: doc.strokes.slice(0, -1) })
  }, [doc, onChange])

  const redo = useCallback(() => {
    setRedoStack((prev) => {
      if (prev.length === 0) return prev
      const stroke = prev[prev.length - 1]!
      onChange({ ...doc, strokes: [...doc.strokes, stroke] })
      return prev.slice(0, -1)
    })
  }, [doc, onChange])

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

  // --- Add / remove page ---
  const addPage = useCallback(() => {
    const lastPage = doc.pages[doc.pages.length - 1]
    const newX = lastPage ? lastPage.x + A4_W + PAGE_GAP : 0
    const newId = `p${Date.now()}`
    onChange({ ...doc, pages: [...doc.pages, { id: newId, x: newX, y: lastPage?.y ?? 0 }] })
  }, [doc, onChange])

  const removePage = useCallback((id: string) => {
    onChange({ ...doc, pages: doc.pages.filter((p) => p.id !== id) })
  }, [doc, onChange])

  const clearAll = useCallback(() => {
    setRedoStack([])
    onChange({ ...doc, strokes: [] })
  }, [doc, onChange])

  const currentOutline = currentPoints.length >= 2
    ? getSvgPathFromStroke(getStroke(currentPoints, { ...STROKE_OPTIONS, size: baseWidth, last: false }))
    : ""

  const nextSize = useCallback(() => {
    setStrokeWidth((w) => w === "thin" ? "medium" : w === "medium" ? "thick" : "thin")
  }, [])

  // --- PDF embeds ---
  const [addMenuOpen, setAddMenuOpen] = useState(false)
  const addMenuRef = useRef<HTMLDivElement>(null)
  const [embedAspects, setEmbedAspects] = useState<Record<string, number>>({})
  const embedDrag = useRef<{
    id: string; startX: number; startY: number
    origX: number; origY: number; origW: number
    mode: "move" | "resize"
  } | null>(null)

  useEffect(() => {
    if (!addMenuOpen) return
    const handler = (e: MouseEvent) => {
      if (addMenuRef.current && !addMenuRef.current.contains(e.target as Node)) setAddMenuOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [addMenuOpen])

  // latest doc/onChange for the long-lived window drag listeners, so they don't
  // need to re-subscribe on every stroke
  const liveRef = useRef({ doc, onChange })
  liveRef.current = { doc, onChange }

  const addPdfEmbed = useCallback((path: string) => {
    const embeds = doc.pdfEmbeds ?? []
    if (embeds.some((e) => e.path === path)) { setAddMenuOpen(false); return }
    const el = containerRef.current
    const vw = el ? el.clientWidth : 800
    const vh = el ? el.clientHeight : 600
    // drop it centred in the current viewport (height unknown until measured)
    const cx = (-offsetRef.current.x + vw / 2) / scaleRef.current - DEFAULT_EMBED_WIDTH / 2
    const cy = (-offsetRef.current.y + vh / 2) / scaleRef.current - (DEFAULT_EMBED_WIDTH * FALLBACK_EMBED_ASPECT) / 2
    onChange({ ...doc, pdfEmbeds: [...embeds, { id: `pdf-${Date.now()}`, path, x: cx, y: cy, width: DEFAULT_EMBED_WIDTH }] })
    setAddMenuOpen(false)
  }, [doc, onChange])

  const removePdfEmbed = useCallback((id: string) => {
    onChange({ ...doc, pdfEmbeds: (doc.pdfEmbeds ?? []).filter((e) => e.id !== id) })
    setEmbedAspects(({ [id]: _removed, ...rest }) => rest)
  }, [doc, onChange])

  const startEmbedInteraction = useCallback((id: string, clientX: number, clientY: number, mode: "move" | "resize") => {
    const embed = (liveRef.current.doc.pdfEmbeds ?? []).find((e) => e.id === id)
    if (!embed) return
    embedDrag.current = { id, startX: clientX, startY: clientY, origX: embed.x, origY: embed.y, origW: embed.width, mode }
  }, [])

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const d = embedDrag.current
      if (!d) return
      const { doc: cur, onChange: change } = liveRef.current
      const dx = (e.clientX - d.startX) / scaleRef.current
      const dy = (e.clientY - d.startY) / scaleRef.current
      const embeds = (cur.pdfEmbeds ?? []).map((em) => {
        if (em.id !== d.id) return em
        // only width is stored; height always follows the PDF's real aspect ratio
        if (d.mode === "resize") return { ...em, width: Math.max(MIN_EMBED_WIDTH, d.origW + dx) }
        return { ...em, x: d.origX + dx, y: d.origY + dy }
      })
      change({ ...cur, pdfEmbeds: embeds })
    }
    const onUp = () => { embedDrag.current = null }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
    }
  }, [])

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
              {(availablePdfs ?? []).length > 0 && (
                <>
                  <div className="mx-2 my-1 border-t border-divider/50" />
                  {availablePdfs!.map((pdf) => (
                    <button
                      key={pdf.path}
                      onClick={() => addPdfEmbed(pdf.path)}
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
        <div className="flex-1" />
        <span className="text-[10px] text-ink-faint tabular-nums">{Math.round(scale * 100)}%</span>
        {isErasing && <span className="text-[10px] text-ink-faint ml-1">(eraser)</span>}
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

        {/* Transformed canvas layer */}
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
            {/* Pages */}
            {doc.pages.map((page) => {
              const btnSize = 24 / scale
              return (
                <g key={page.id}>
                  <rect
                    x={page.x}
                    y={page.y}
                    width={A4_W}
                    height={A4_H}
                    fill="var(--surface-elevated)"
                    stroke="var(--divider-strong)"
                    strokeWidth={1 / scale}
                  />
                  <text
                    x={page.x + A4_W / 2}
                    y={page.y + A4_H + 16 / scale}
                    textAnchor="middle"
                    fontSize={10 / scale}
                    fill="var(--ink-faint)"
                    style={{ userSelect: "none", pointerEvents: "none" }}
                  >
                    A4 · rasterize on export
                  </text>
                  {/* Delete icon — visible only when pointer is near it */}
                  <foreignObject
                    x={page.x + A4_W - btnSize / 2}
                    y={page.y - btnSize / 2}
                    width={btnSize}
                    height={btnSize}
                    className="group/del"
                  >
                    <button
                      onClick={(e) => { e.stopPropagation(); removePage(page.id) }}
                      className="flex items-center justify-center w-full h-full rounded opacity-0 group-hover/del:opacity-100 transition-opacity bg-danger/10 text-danger hover:bg-danger/20"
                    >
                      <Trash2 style={{ width: btnSize * scale * 0.55, height: btnSize * scale * 0.55 }} />
                    </button>
                  </foreignObject>
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
          </g>
        </svg>

        {/* PDF embeds — positioned HTML over SVG, not part of export */}
        {(doc.pdfEmbeds ?? []).map((embed) => {
          const screenX = embed.x * scale + offset.x
          const screenY = embed.y * scale + offset.y
          const screenW = embed.width * scale
          // body height derived synchronously from width × aspect so resizing
          // never stretches; the page aspect is reported by PdfViewer once loaded
          const aspect = embedAspects[embed.id] ?? FALLBACK_EMBED_ASPECT
          const name = embed.path.split("/").pop() ?? "PDF"
          return (
            <div
              key={embed.id}
              className="absolute flex flex-col border border-divider-strong rounded-lg overflow-hidden bg-paper shadow-[var(--shadow-lg)]"
              style={{ left: screenX, top: screenY, width: screenW }}
            >
              {/* Header — drag to move */}
              <div
                className="flex items-center gap-1.5 px-2 py-1 border-b border-divider/50 cursor-grab active:cursor-grabbing select-none shrink-0"
                onPointerDown={(e) => { e.stopPropagation(); startEmbedInteraction(embed.id, e.clientX, e.clientY, "move") }}
              >
                <FileType className="h-3 w-3 text-ink-faint shrink-0" />
                <span className="flex-1 min-w-0 truncate text-[10px] font-medium text-ink-muted">{name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); removePdfEmbed(embed.id) }}
                  className="rounded p-0.5 text-ink-faint hover:text-danger transition-colors shrink-0"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              {/* PDF content — fills this fixed-aspect box */}
              <div style={{ height: screenW * aspect }} onPointerDown={(e) => e.stopPropagation()}>
                <PdfViewer
                  url={fileViewerRawUrl(embed.path)}
                  onPageAspect={(r) => setEmbedAspects((prev) => prev[embed.id] === r ? prev : { ...prev, [embed.id]: r })}
                />
              </div>
              {/* Resize handle — bottom-right corner */}
              <div
                className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize"
                onPointerDown={(e) => { e.stopPropagation(); startEmbedInteraction(embed.id, e.clientX, e.clientY, "resize") }}
              >
                <svg className="w-full h-full text-ink-faint" viewBox="0 0 16 16">
                  <path d="M14 2L2 14M14 6L6 14M14 10L10 14" stroke="currentColor" strokeWidth="1.5" fill="none" />
                </svg>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
