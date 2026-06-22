"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { getStroke } from "perfect-freehand"
import { type StrokeData, getSvgPathFromStroke } from "@/lib/drawing"
import { activeThemeName } from "@/lib/palette"
import { cn } from "@/lib/utils"
import { Plus, Undo2, Trash2 } from "lucide-react"

const A4_W = 794
const A4_H = 1123
const PAGE_GAP = 40
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

export interface CanvasDocument {
  version: 1
  viewport?: { scale: number; offsetX: number; offsetY: number }
  pages: CanvasPage[]
  strokes: StrokeData[]
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
}

export function CanvasEditor({ doc, onChange }: CanvasEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const [scale, setScale] = useState(doc.viewport?.scale ?? 0.5)
  const [offset, setOffset] = useState({ x: doc.viewport?.offsetX ?? 40, y: doc.viewport?.offsetY ?? 40 })
  const scaleRef = useRef(scale)
  const offsetRef = useRef(offset)
  scaleRef.current = scale
  offsetRef.current = offset

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

  const pickColor = useCallback((c: string) => {
    setColor(c)
    setRecentColors((prev) => {
      const without = prev.filter((x) => x !== c)
      return [c, ...without].slice(0, 3)
    })
    setColorOpen(false)
  }, [])

  const persistViewport = useCallback((s: number, o: { x: number; y: number }) => {
    onChange({ ...doc, viewport: { scale: s, offsetX: o.x, offsetY: o.y } })
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
    setCurrentPoints([])
  }, [isDrawing, isErasing, currentPoints, color, baseWidth, doc, onChange])

  // --- Undo ---
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault()
        onChange({ ...doc, strokes: doc.strokes.slice(0, -1) })
      }
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [doc, onChange])

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
    onChange({ ...doc, strokes: [] })
  }, [doc, onChange])

  const currentOutline = currentPoints.length >= 2
    ? getSvgPathFromStroke(getStroke(currentPoints, { ...STROKE_OPTIONS, size: baseWidth, last: false }))
    : ""

  const nextSize = useCallback(() => {
    setStrokeWidth((w) => w === "thin" ? "medium" : w === "medium" ? "thick" : "thin")
  }, [])

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-divider/50 shrink-0">
        <button onClick={addPage} className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium text-ink-muted hover:text-ink hover:bg-hover transition-colors">
          <Plus className="h-3 w-3" /> Page
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        <button
          onClick={() => onChange({ ...doc, strokes: doc.strokes.slice(0, -1) })}
          disabled={doc.strokes.length === 0}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
          title="Undo (Ctrl+Z)"
        >
          <Undo2 className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={clearAll}
          disabled={doc.strokes.length === 0}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
          title="Clear all"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        {/* Size toggle with dot hint */}
        <button
          onClick={nextSize}
          className="flex items-center gap-1.5 rounded px-2 py-1 text-[10px] font-medium text-ink-subtle hover:text-ink transition-colors"
          title="Stroke width"
        >
          <svg width="8" height="8" viewBox="0 0 8 8" className="shrink-0">
            <circle cx="4" cy="4" r={strokeWidth === "thin" ? 1.5 : strokeWidth === "medium" ? 2.5 : 3.5} fill="currentColor" />
          </svg>
          {strokeWidth === "thin" ? "Thin" : strokeWidth === "medium" ? "Medium" : "Thick"}
        </button>
        <div className="w-px h-4 bg-divider/50 mx-0.5" />
        {/* Color popover */}
        <div className="relative" ref={colorRef}>
          <button
            onClick={() => setColorOpen((o) => !o)}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-[10px] font-medium hover:bg-hover transition-colors"
            style={{ color: displayColor(color) }}
          >
            Color
          </button>
          {colorOpen && (
            <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] p-2 min-w-[120px]">
              <div className="grid grid-cols-4 gap-1 mb-1.5">
                {PRESET_COLORS.map((c) => (
                  <button
                    key={c.value}
                    onClick={() => pickColor(c.value)}
                    title={c.name}
                    className={cn(
                      "w-6 h-6 rounded-full border-2 transition-all",
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
                  onChange={(e) => pickColor(e.target.value)}
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
                    x={page.x + A4_W - btnSize - 4 / scale}
                    y={page.y + 4 / scale}
                    width={btnSize + 16 / scale}
                    height={btnSize + 16 / scale}
                    className="group/del"
                  >
                    <div className="flex items-start justify-end w-full h-full p-[4px]">
                      <button
                        onClick={(e) => { e.stopPropagation(); removePage(page.id) }}
                        className="flex items-center justify-center rounded opacity-0 group-hover/del:opacity-100 transition-opacity bg-danger/10 text-danger hover:bg-danger/20"
                        style={{ width: btnSize * scale, height: btnSize * scale }}
                        title="Delete page"
                      >
                        <Trash2 style={{ width: btnSize * scale * 0.55, height: btnSize * scale * 0.55 }} />
                      </button>
                    </div>
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
      </div>
    </div>
  )
}
