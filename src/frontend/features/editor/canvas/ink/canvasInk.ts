import { inkOutlinePath, type StrokeData } from "@/lib/drawing"

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

// Committed strokes are immutable, so outline path and bounds are computed once per
// object, not per render or eraser sample. Weak keys free the cache when an edit drops a stroke.
const strokePathCache = new WeakMap<StrokeData, string>()
const strokeBoundsCache = new WeakMap<StrokeData, [number, number, number, number]>()

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
    const [minX, minY, maxX, maxY] = strokeBounds(stroke)
    if (ex < minX - threshold || ex > maxX + threshold || ey < minY - threshold || ey > maxY + threshold) return true
    const pts = stroke.points
    if (pts.length === 1) return Math.hypot(pts[0]![0]! - ex, pts[0]![1]! - ey) >= threshold
    for (let i = 1; i < pts.length; i++) {
      if (distToSegment(ex, ey, pts[i - 1]![0]!, pts[i - 1]![1]!, pts[i]![0]!, pts[i]![1]!) < threshold) return false
    }
    return true
  })
}
