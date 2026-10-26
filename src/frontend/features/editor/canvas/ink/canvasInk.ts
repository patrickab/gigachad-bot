import { strokeOutlinePath, type StrokeData } from "@/lib/drawing"

export const PRESET_COLORS = [
  { name: "Black", value: "#000000" },
  { name: "Red", value: "#dc2626" },
  { name: "Green", value: "#16a34a" },
  { name: "Blue", value: "#2563eb" },
] as const

export type StrokeSize = "thin" | "medium" | "thick"
export const STROKE_WIDTHS: Record<StrokeSize, number> = { thin: 2, medium: 5, thick: 10 }

/** Display colour of stored ink. The default pen (stored black) follows the theme's text colour (--ink, as the selected sidebar file). */
export function inkColor(color: string): string {
  return color === "#000000" ? "var(--ink)" : color
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
    d = strokeOutlinePath(stroke.points, stroke.width)
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
