import { A4_WIDTH } from "@/lib/domain/canvasDocument"
import type { StrokeData } from "@/lib/drawing"

/** Page width in stroke units, so canvas pen widths look the same on a page. */
export const PAGE_UNITS = A4_WIDTH

/** Strokes grouped by the PDF page they are anchored to, in first-seen order. Strokes carry page units, points may lie outside the page (margin notes). */
export function strokesByPage(strokes: StrokeData[]): Map<number | undefined, StrokeData[]> {
  const pages = new Map<number | undefined, StrokeData[]>()
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
