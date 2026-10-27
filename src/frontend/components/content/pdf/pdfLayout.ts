import type { DocumentProps } from "react-pdf"

export type PdfDocument = Parameters<NonNullable<DocumentProps["onLoadSuccess"]>>[0]
export type PdfPage = Awaited<ReturnType<PdfDocument["getPage"]>>

// Max render width in px, and vertical gap between pages in px.
export const RENDER_WIDTH = 900
export const PAGE_GAP = 12
// Gap left and right of a fitted page. The scroller reserves its scrollbar gutter on both edges, so the page stays centered
export const PAGE_SIDE_GAP = 8

/** Every page's height / width, `fallback` for pages that fail to load. */
export async function readPageAspects(pdf: PdfDocument, fallback: number): Promise<number[]> {
  return Promise.all(Array.from({ length: pdf.numPages }, async (_, index) => {
    try {
      const viewport = (await pdf.getPage(index + 1)).getViewport({ scale: 1 })
      return viewport.height / viewport.width
    } catch {
      return fallback
    }
  }))
}

/** Top offset of every page at `pageWidth`, relative to the first page. */
function pageTops(aspects: number[], pageWidth: number): number[] {
  const tops: number[] = []
  let top = 0
  for (const aspect of aspects) {
    tops.push(top)
    top += Math.round(pageWidth * aspect) + PAGE_GAP
  }
  return tops
}

/** `scrollTop` re-projected from pages `previousWidth` px wide to `nextWidth`: same page, same fraction within it (the gaps do not scale). */
export function scrollTopAfterWidthChange(scrollTop: number, aspects: number[], previousWidth: number, nextWidth: number): number {
  const previousTops = pageTops(aspects, previousWidth)
  const nextTops = pageTops(aspects, nextWidth)
  let page = 0
  while (page + 1 < previousTops.length && previousTops[page + 1] <= scrollTop) page++
  const fraction = (scrollTop - previousTops[page]) / (previousWidth * aspects[page])
  return nextTops[page] + fraction * nextWidth * aspects[page]
}
