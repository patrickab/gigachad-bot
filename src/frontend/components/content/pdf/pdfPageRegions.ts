import type { CSSProperties } from "react"
import { pdfjs } from "react-pdf"
import type { PdfPage } from "./pdfLayout"

// image bbox as fractions of the page (0..1), so it survives zoom/width changes
export interface ImageRect { left: number; top: number; width: number; height: number }

export interface PageRegions { images: ImageRect[]; links: ImageRect[] }

// Walk a page's operator list and return the bounding box of every raster image
// (XObject / inline image). Used to punch images out of the theme color layers (and
// the dark-mode invert) so figures keep their true colors. ponytail: skips image *masks* (often 1-bit text
// stencils); add OPS.paintImageMaskXObject if photos render as masks somewhere.
export async function collectImageRects(page: PdfPage): Promise<ImageRect[]> {
  const ops = await page.getOperatorList()
  const vp = page.getViewport({ scale: 1 })
  const OPS = pdfjs.OPS
  type M = [number, number, number, number, number, number]
  const mul = (a: M, b: M): M => [
    a[0] * b[0] + a[2] * b[1], a[1] * b[0] + a[3] * b[1],
    a[0] * b[2] + a[2] * b[3], a[1] * b[2] + a[3] * b[3],
    a[0] * b[4] + a[2] * b[5] + a[4], a[1] * b[4] + a[3] * b[5] + a[5],
  ]
  const apply = (m: M, x: number, y: number): [number, number] => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]

  const rects: ImageRect[] = []
  let ctm: M = [1, 0, 0, 1, 0, 0]
  const stack: M[] = []
  for (let i = 0; i < ops.fnArray.length; i++) {
    const fn = ops.fnArray[i]
    if (fn === OPS.save) stack.push(ctm)
    else if (fn === OPS.restore) ctm = stack.pop() ?? ctm
    else if (fn === OPS.transform) ctm = mul(ctm, ops.argsArray[i] as M)
    else if (fn === OPS.paintImageXObject || fn === OPS.paintInlineImageXObject) {
      const dev = mul(vp.transform as M, ctm) // image fills the unit square under ctm
      const pts = [apply(dev, 0, 0), apply(dev, 1, 0), apply(dev, 0, 1), apply(dev, 1, 1)]
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
      const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys)
      rects.push({
        left: minX / vp.width,
        top: minY / vp.height,
        width: (maxX - minX) / vp.width,
        height: (maxY - minY) / vp.height,
      })
    }
  }
  return rects
}

/** Bounding box of every link annotation, so link text keeps the PDF's own link color. */
export async function collectLinkRects(page: PdfPage): Promise<ImageRect[]> {
  const vp = page.getViewport({ scale: 1 })
  const annotations = await page.getAnnotations()
  return annotations
    .filter((annotation) => annotation.subtype === "Link")
    .map((annotation) => {
      const [x1, y1, x2, y2] = vp.convertToViewportRectangle(annotation.rect)
      return {
        left: Math.min(x1, x2) / vp.width,
        top: Math.min(y1, y2) / vp.height,
        width: Math.abs(x2 - x1) / vp.width,
        height: Math.abs(y2 - y1) / vp.height,
      }
    })
}

/** CSS mask for a page-wide color layer: `holes` -> layer everywhere except `rects`, else -> layer only on `rects`. */
export function rectMask(rects: ImageRect[], holes: boolean): CSSProperties | undefined {
  if (!rects.length) return holes ? undefined : { display: "none" }
  const shapes = rects.map((r) => `M${r.left} ${r.top}h${r.width}v${r.height}h${-r.width}Z`).join("")
  const path = holes ? `M0 0H1V1H0Z${shapes}` : shapes
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1' preserveAspectRatio='none'><path fill-rule='evenodd' d='${path}'/></svg>`
  const image = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`
  return { maskImage: image, WebkitMaskImage: image, maskSize: "100% 100%", WebkitMaskSize: "100% 100%" }
}
