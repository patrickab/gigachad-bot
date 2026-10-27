import { getStroke } from "perfect-freehand"

export interface StrokeData {
  /** Each point is [x, y, pressure?]. Pressure defaults to 0.5 when drawing. */
  id: string
  points: number[][]
  color: string
  width: number
  /** Highlighter stroke: constant width, translucent, under the pen ink. */
  highlight?: boolean
  /** PDF page the stroke is anchored to, in page units; set only on PDF-annotation documents. */
  page?: number
}

/** Ray-casting point-in-polygon test, used by the lasso selection tools. */
export function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]!
    const [xj, yj] = poly[j]!
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

export const STROKE_OPTIONS = {
  smoothing: 0.5,
  streamline: 0.5,
  simulatePressure: false,
  last: true,
} as const

// A marker keeps its width regardless of pressure.
const HIGHLIGHT_STROKE_OPTIONS = { ...STROKE_OPTIONS, thinning: 0 }

/** Closed quadratic path through midpoints of the outline points. Returns "" when fewer than 4 points. */
function outlinePath(points: number[][]): string {
  const len = points.length
  if (len < 4) return ""

  const avg = (a: number, b: number) => (a + b) / 2

  let a = points[0]!
  let b = points[1]!
  const c = points[2]!

  let result = `M${a[0]!.toFixed(2)},${a[1]!.toFixed(2)} Q${b[0]!.toFixed(2)},${b[1]!.toFixed(2)} ${avg(b[0], c[0]).toFixed(2)},${avg(b[1], c[1]).toFixed(2)} T`

  for (let i = 2, max = len - 1; i < max; i++) {
    a = points[i]!
    b = points[i + 1]!
    result += `${avg(a[0], b[0]).toFixed(2)},${avg(a[1], b[1]).toFixed(2)} `
  }

  return `${result}Z`
}

/** SVG path of a freehand stroke's outline. Returns "" when the stroke is too short to outline. */
export function strokeOutlinePath(points: number[][], size: number, options: Parameters<typeof getStroke>[1] = STROKE_OPTIONS): string {
  return outlinePath(getStroke(points, { ...options, size }))
}

/** SVG outline of a pen or highlighter stroke. */
export function inkOutlinePath(points: number[][], size: number, highlight: boolean): string {
  return strokeOutlinePath(points, size, highlight ? HIGHLIGHT_STROKE_OPTIONS : STROKE_OPTIONS)
}

/** Pen pressure of a pointer event, with a mid value for mice and pens that report none. */
export function penPressure(e: { pressure: number }): number {
  return e.pressure > 0 ? e.pressure : 0.5
}

export interface Bounds { minX: number; minY: number; maxX: number; maxY: number }

/** Bounding box of `[x, y, ...]` points, or null when there are none. */
export function pointsBounds(points: Iterable<readonly number[]>): Bounds | null {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [x, y] of points) {
    if (x! < minX) minX = x!
    if (x! > maxX) maxX = x!
    if (y! < minY) minY = y!
    if (y! > maxY) maxY = y!
  }
  return minX === Infinity ? null : { minX, minY, maxX, maxY }
}

