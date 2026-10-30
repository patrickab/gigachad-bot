import { inkPoint, pointInPolygon, pointsBounds, type StrokeData } from "@/lib/drawing"

// The lasso keeps its freehand shape — that outline is what you drag to move, and it
// deforms along with the ink when scaled. A single handle at the shape's bottom-right
// resizes diagonally, anchored at the top-left. Stroke `width` is never touched by a
// scale — squeezing a drawing must not thin the pen that drew it.
export interface SelBox { x: number; y: number; w: number; h: number }
export type Poly = [number, number][]

const LASSO_COVERAGE = 0.6 // fraction of a stroke's points that must fall inside the lasso to select it
const SELECTION_MIN = 24 // canvas units — selection box can be squeezed no smaller

export function polyBounds(poly: Poly): SelBox | null {
  const bounds = pointsBounds(poly)
  return bounds && { x: bounds.minX, y: bounds.minY, w: bounds.maxX - bounds.minX, h: bounds.maxY - bounds.minY }
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
  return { ...stroke, points: points.map((p) => { const [x, y, pressure] = remap(p, from, to); return inkPoint(x!, y!, pressure!) }) }
}

// Ids of the strokes a finished lasso loop mostly covers — full enclosure meant a stroke
// poking one pixel out was silently missed.
// ponytail: coverage is measured on the raw sample points, so a 2-point straightened
// line still effectively needs both ends inside; resample if that ever bites.
export function strokesInLasso(strokes: StrokeData[], lasso: Poly): Set<string> {
  const picked = new Set<string>()
  for (const s of strokes) {
    if (s.points.length === 0) continue
    const inside = s.points.reduce((n, p) => n + (pointInPolygon(p[0]!, p[1]!, lasso) ? 1 : 0), 0)
    if (inside / s.points.length > LASSO_COVERAGE) picked.add(s.id)
  }
  return picked
}
