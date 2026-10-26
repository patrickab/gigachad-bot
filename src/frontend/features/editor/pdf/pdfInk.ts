import { STROKE_WIDTHS, type StrokeSize } from "@/features/editor/canvas/ink/canvasInk"
import { A4_WIDTH } from "@/lib/domain/canvasDocument"
import { STROKE_OPTIONS, strokeOutlinePath, type StrokeData } from "@/lib/drawing"

/** Page width in stroke units, so canvas pen widths look the same on a page. */
export const PAGE_UNITS = A4_WIDTH

/** A canvas stroke anchored to a PDF page, in page units. Points may lie outside the page (margin notes). */
export type PdfStroke = StrokeData & { page: number; highlight?: boolean }

/** Top-bar tool. None selected: the stylus writes with the pen, mouse and touch read. */
export type PdfInkTool = "pen" | "highlighter" | "eraser"

export const HIGHLIGHT_COLORS = [
  { name: "Yellow", value: "#facc15" },
  { name: "Green", value: "#4ade80" },
  { name: "Pink", value: "#f472b6" },
  { name: "Blue", value: "#60a5fa" },
] as const

const HIGHLIGHT_WIDTHS: Record<StrokeSize, number> = { thin: 10, medium: 16, thick: 24 }
// A marker keeps its width regardless of pressure.
const HIGHLIGHT_OPTIONS = { ...STROKE_OPTIONS, thinning: 0 }

export function strokeWidth(size: StrokeSize, highlight: boolean): number {
  return highlight ? HIGHLIGHT_WIDTHS[size] : STROKE_WIDTHS[size]
}

export function strokeOutline(points: number[][], width: number, highlight: boolean): string {
  return highlight ? strokeOutlinePath(points, width, HIGHLIGHT_OPTIONS) : strokeOutlinePath(points, width)
}

// Committed strokes are immutable, so each outline is computed once per stroke object.
const outlineCache = new WeakMap<PdfStroke, string>()

export function cachedStrokeOutline(stroke: PdfStroke): string {
  let outline = outlineCache.get(stroke)
  if (outline === undefined) {
    outline = strokeOutline(stroke.points, stroke.width, !!stroke.highlight)
    outlineCache.set(stroke, outline)
  }
  return outline
}

export function strokesByPage(strokes: PdfStroke[]): Map<number, PdfStroke[]> {
  const pages = new Map<number, PdfStroke[]>()
  for (const stroke of strokes) {
    const list = pages.get(stroke.page)
    if (list) list.push(stroke)
    else pages.set(stroke.page, [stroke])
  }
  return pages
}

export interface PageBox {
  page: number
  rect: DOMRect
}

/** Page a stroke starting at the pointer belongs to.
 *
 * - `insideOnly` -> only a page under the pointer, else null
 * - Otherwise the page spanning the pointer's height, else the vertically nearest one (margins and gaps)
 */
export function pageAt(pages: PageBox[], clientX: number, clientY: number, insideOnly: boolean): PageBox | null {
  if (insideOnly) {
    return pages.find(({ rect }) => clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom) ?? null
  }
  let nearest: PageBox | null = null
  let nearestDistance = Infinity
  for (const box of pages) {
    const distance = Math.max(box.rect.top - clientY, 0, clientY - box.rect.bottom)
    if (distance < nearestDistance) {
      nearest = box
      nearestDistance = distance
    }
  }
  return nearest
}

/** Client point in page units of the page drawn at `rect`. */
export function toPagePoint(rect: DOMRect, clientX: number, clientY: number): [number, number] {
  const unitsPerPixel = PAGE_UNITS / rect.width
  return [(clientX - rect.left) * unitsPerPixel, (clientY - rect.top) * unitsPerPixel]
}
