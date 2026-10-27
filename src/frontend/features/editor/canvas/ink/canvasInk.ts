import { canvasEntityId } from "@/lib/domain/canvasDocument"
import { inkOutlinePath, penPressure, pointsBounds, type Bounds, type StrokeData } from "@/lib/drawing"

export const PRESET_COLORS = [
  { name: "Black", value: "#000000" },
  { name: "Red", value: "#dc2626" },
  { name: "Green", value: "#16a34a" },
  { name: "Blue", value: "#2563eb" },
] as const

/** Highlighter presets, the first is the default. */
export const HIGHLIGHT_COLORS = [
  { name: "Green", value: "#4ade80" },
  { name: "Yellow", value: "#facc15" },
  { name: "Pink", value: "#f472b6" },
  { name: "Blue", value: "#60a5fa" },
] as const

/** Selected ink tool. None selected: the stylus writes with the pen, the mouse keeps its default (pan, read). */
export type InkTool = "pen" | "highlighter" | "eraser"

export type StrokeSize = "thin" | "medium" | "thick"
const STROKE_WIDTHS: Record<StrokeSize, number> = { thin: 2, medium: 5, thick: 10 }
const HIGHLIGHT_WIDTHS: Record<StrokeSize, number> = { thin: 10, medium: 16, thick: 24 }

export function strokeWidth(size: StrokeSize, highlight: boolean): number {
  return highlight ? HIGHLIGHT_WIDTHS[size] : STROKE_WIDTHS[size]
}

/** Stored colour of the default pen, displayed in the theme's text colour. */
export const DEFAULT_PEN_COLOR = "#000000"

/** Eraser radius in screen px. */
export const ERASER_RADIUS_PX = 15

/** Display colour of stored ink. The default pen (stored black) follows the theme's text colour (--ink, as the selected sidebar file). */
export function inkColor(color: string): string {
  return color === DEFAULT_PEN_COLOR ? "var(--ink)" : color
}

/**
 * Preserve the latest position supplied by either coalesced input or pointerup. A
 * captured pen can cross a child or canvas boundary without ending its stroke.
 */
export function appendStrokePoint(points: number[][], point: number[]): void {
  const last = points[points.length - 1]
  if (last?.[0] === point[0] && last[1] === point[1]) return
  points.push(point)
}

/** Stroke sample at a pointer position: its point in stroke units plus the pen pressure. */
export function pointerSample(e: { clientX: number; clientY: number; pressure: number }, toPoint: (clientX: number, clientY: number) => readonly number[]): number[] {
  return [...toPoint(e.clientX, e.clientY), penPressure(e)]
}

/** Appends every coalesced sample of a pointer move: they carry the pen's full sample rate (~240Hz) instead of one point per frame. */
export function appendPointerSamples(points: number[][], e: PointerEvent, toPoint: (clientX: number, clientY: number) => readonly number[]): void {
  for (const sample of e.getCoalescedEvents?.() ?? [e]) appendStrokePoint(points, pointerSample(sample, toPoint))
}

/**
 * Closes a stroke's points and returns them, or null when too short to keep. pointercancel and
 * lostpointercapture carry no usable final position (often 0, 0, a diagonal to the origin), so
 * only pointerup appends one and the latest move sample is otherwise the end.
 */
export function finishStroke(points: number[][], e: { type: string; clientX: number; clientY: number; pressure: number }, toPoint: (clientX: number, clientY: number) => readonly number[]): number[][] | null {
  if (e.type === "pointerup") appendStrokePoint(points, pointerSample(e, toPoint))
  return points.length >= 2 ? points : null
}

/**
 * What a press does with ink, shared by the canvas and the PDF viewer.
 *
 * - Touch never inks, it pans and zooms
 * - The pen's eraser end always erases
 * - Otherwise only the primary button inks: the selected tool decides, a stylus writes even without one, the mouse needs one
 */
export function pointerInkRole(e: { pointerType: string; button: number; buttons: number }, tool: InkTool | null): "draw" | "erase" | null {
  if (e.pointerType === "touch") return null
  if (e.pointerType === "pen" && (e.button === 5 || (e.buttons & 32) !== 0)) return "erase"
  if (e.button !== 0) return null
  if (e.pointerType === "mouse" && !tool) return null
  return tool === "eraser" ? "erase" : "draw"
}

/** New stroke of the given ink. */
export function newStroke(points: number[][], { color, width, highlight }: { color: string; width: number; highlight: boolean }): StrokeData {
  return { id: canvasEntityId(), points, color, width, ...(highlight && { highlight: true }) }
}

/** Strokes split into the translucent highlight layer and the pen ink drawn over it. */
export function splitHighlights<Stroke extends StrokeData>(strokes: readonly Stroke[]): { highlights: Stroke[]; pen: Stroke[] } {
  return { highlights: strokes.filter((stroke) => stroke.highlight), pen: strokes.filter((stroke) => !stroke.highlight) }
}

// Committed strokes are immutable, so outline path and bounds are computed once per
// object, not per render or eraser sample. Weak keys free the cache when an edit drops a stroke.
const strokePathCache = new WeakMap<StrokeData, string>()
const strokeBoundsCache = new WeakMap<StrokeData, Bounds | null>()

/** Outline path of a committed stroke, cached per stroke object. */
export function strokePath(stroke: StrokeData): string {
  let d = strokePathCache.get(stroke)
  if (d === undefined) {
    d = inkOutlinePath(stroke.points, stroke.width, !!stroke.highlight)
    strokePathCache.set(stroke, d)
  }
  return d
}

// The eraser tests every stroke on every pointermove, so reject cheaply with four
// comparisons against a cached box instead of per-segment distances.
function strokeBounds(s: StrokeData): Bounds | null {
  let b = strokeBoundsCache.get(s)
  if (b === undefined) {
    b = pointsBounds(s.points)
    strokeBoundsCache.set(s, b)
  }
  return b
}

// Shortest distance from (px,py) to segment (ax,ay)-(bx,by). Straight strokes (e.g.
// straightened lines) can be as sparse as 2 points, so the eraser must hit-test the
// segment between points, not just the points themselves.
function distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax
  const dy = by - ay
  const lenSq = dx * dx + dy * dy
  const t = lenSq === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

/** Strokes that survive an eraser of radius `threshold` centred at (ex, ey), all in canvas units. */
export function strokesAfterErase<Stroke extends StrokeData>(strokes: Stroke[], ex: number, ey: number, threshold: number): Stroke[] {
  return strokes.filter((stroke) => {
    const bounds = strokeBounds(stroke)
    if (!bounds || ex < bounds.minX - threshold || ex > bounds.maxX + threshold || ey < bounds.minY - threshold || ey > bounds.maxY + threshold) return true
    const pts = stroke.points
    if (pts.length === 1) return Math.hypot(pts[0]![0]! - ex, pts[0]![1]! - ey) >= threshold
    for (let i = 1; i < pts.length; i++) {
      if (distToSegment(ex, ey, pts[i - 1]![0]!, pts[i - 1]![1]!, pts[i]![0]!, pts[i]![1]!) < threshold) return false
    }
    return true
  })
}
