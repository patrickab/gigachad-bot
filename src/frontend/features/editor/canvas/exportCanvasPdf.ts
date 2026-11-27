import { A4_ASPECT, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { imageFrameEmbeds } from "@/lib/domain/canvasRender"
import { writeBinaryDocument } from "@/lib/backend/api"
import { renderPageToPng } from "@/lib/canvasExport"

/** Store a canvas's pages as `<name>.pdf` in its project. No UI calls it at present: the user wants it kept for re-homing. */
export async function exportCanvasPdf(canvasDocument: CanvasDocument, slug: string, filename: string): Promise<void> {
  const pages = canvasDocument.frames.filter((f) => f.kind === "page")
  if (pages.length === 0) return
  const { PDFDocument } = await import("pdf-lib") // lazy: heavy, and only needed on export
  // A4 at 96dpi in px (1123x794) for the page crop, and in pt (595.28x841.89) for the PDF page
  const imgs = imageFrameEmbeds(canvasDocument.frames)
  const pdfDoc = await PDFDocument.create()
  for (const page of pages) {
    const pngBytes = await renderPageToPng(canvasDocument.strokes, page.x, page.y, page.width, page.width * A4_ASPECT, imgs, canvasDocument.texts)
    const img = await pdfDoc.embedPng(pngBytes)
    const pdfPage = pdfDoc.addPage([595.28, 841.89])
    pdfPage.drawImage(img, { x: 0, y: 0, width: 595.28, height: 841.89 })
  }
  const pdfBytes = await pdfDoc.save()
  const pdfName = filename.replace(/\.canvas$/, ".pdf")
  await writeBinaryDocument(slug, pdfName, new Blob([pdfBytes.buffer as ArrayBuffer], { type: "application/pdf" }))
}
