"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Download, X } from "lucide-react"
import { CanvasEditor } from "./CanvasEditor"
import { ResizableEditor } from "@/features/editor/ResizableEditor"
import type { DocumentEditorProps } from "@/features/editor/TextDocumentEditor"
import { A4_ASPECT, serializeCanvasDoc, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { storeDrawing, writeBinaryDocument } from "@/lib/backend/api"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { renderPageToPng, renderCanvasToJpeg, type EmbedRect } from "@/lib/drawing"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { baseName } from "@/lib/paths"

/** Load each image frame to measure its aspect ratio (falls back to 1 if it fails to load). */
async function canvasImageEmbeds(doc: CanvasDocument): Promise<EmbedRect[]> {
  return Promise.all(doc.frames.filter((f) => f.kind === "image" && f.path).map(async (f) => {
    const url = fileViewerRawUrl(f.path!)
    let aspect = 1
    try {
      const img = new Image()
      img.crossOrigin = "anonymous"
      await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = reject; img.src = url })
      if (img.naturalWidth > 0) aspect = img.naturalHeight / img.naturalWidth
    } catch { /* */ }
    return { url, x: f.x, y: f.y, width: f.width, aspect }
  }))
}

/** Database canvas: the collaboration hook owns loading and saving. */
export function CanvasDocumentEditor({ path, slug, onClose, onSaved, onLiveContent, overlay, onModeLabel, canvasToolbarSlots }: DocumentEditorProps) {
  const { document: canvasDocument, replace, status, retry } = useCollaborativeCanvas(path)
  const filename = baseName(path)
  const [exporting, setExporting] = useState(false)
  const liveContentRef = useRef(onLiveContent)
  liveContentRef.current = onLiveContent
  // Only local edits refresh the drawing JPEG, otherwise every open client would
  // re-render and upload it for each remote change.
  const drawnRef = useRef(false)

  const handleChange = useCallback((next: CanvasDocument) => {
    drawnRef.current = true
    replace(next)
  }, [replace])

  // The one serialization per quiet second feeds the live content the chat reads at send time.
  useEffect(() => {
    if (!canvasDocument) return
    const timer = setTimeout(() => {
      liveContentRef.current?.(path, serializeCanvasDoc(canvasDocument))
      const { strokes, texts, frames } = canvasDocument
      if (!drawnRef.current || (strokes.length === 0 && texts.length === 0 && !frames.some((f) => f.kind === "image"))) return
      drawnRef.current = false
      canvasImageEmbeds(canvasDocument)
        .then((imgs) => renderCanvasToJpeg(strokes, 20, imgs, texts))
        .then((blob) => storeDrawing(filename.replace(/\.canvas$/, ".jpg"), blob))
        .catch(() => {})
    }, 1000)
    return () => clearTimeout(timer)
  }, [path, filename, canvasDocument])

  useEffect(() => () => {
    liveContentRef.current?.(path, null)
  }, [path])

  useEffect(() => {
    if (overlay) onModeLabel?.(filename)
  }, [overlay, filename, onModeLabel])

  const handleExportPdf = useCallback(async () => {
    if (!canvasDocument || exporting) return
    const pages = canvasDocument.frames.filter((f) => f.kind === "page")
    if (pages.length === 0) return
    setExporting(true)
    try {
      const { PDFDocument } = await import("pdf-lib") // lazy: heavy, and only needed on export
      // A4 at 96dpi in px (1123x794) for the page crop, and in pt (595.28x841.89) for the PDF page
      const imgs = await canvasImageEmbeds(canvasDocument)
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
      onSaved?.()
    } catch { /* */ }
    setExporting(false)
  }, [canvasDocument, exporting, slug, filename, onSaved])

  useEffect(() => {
    if (!overlay) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        onClose()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [overlay, onClose])

  if (!canvasDocument) {
    if (status === "error") {
      return (
        <div className="flex flex-col items-center justify-center gap-2 py-6 text-xs text-ink-faint">
          <span>Failed to load this canvas. Its stored content was left untouched.</span>
          <button onClick={retry} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
            Retry
          </button>
        </div>
      )
    }
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
  }

  const editor = (
    <>
      {status === "error" && (
        <div role="status" className="flex items-center justify-between gap-2 px-3 py-1 text-[10px] text-ink-faint border-b border-divider/50">
          <span>Canvas sync interrupted. Changes will retry automatically.</span>
          <button onClick={retry} className="rounded px-1.5 py-0.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">Retry</button>
        </div>
      )}
      <CanvasEditor
        doc={canvasDocument}
        onChange={handleChange}
        slug={slug}
        docPath={path}
        onImageAdded={() => onSaved?.()}
        toolbarSlots={canvasToolbarSlots}
      />
    </>
  )
  if (overlay) return <div className="absolute inset-0 z-30 flex flex-col bg-paper">{editor}</div>
  return (
    <ResizableEditor>
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-divider/50 shrink-0">
        <span className="text-[11px] font-medium text-ink truncate">{filename}</span>
        <div className="flex items-center gap-1 shrink-0">
          <button
            aria-label="Export PDF"
            onClick={handleExportPdf}
            disabled={exporting || !canvasDocument.frames.some((f) => f.kind === "page")}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
          >
            {exporting
              ? <span className="h-3.5 w-3.5 block animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
              : <Download className="h-3.5 w-3.5" />}
          </button>
          <button aria-label="Close document" onClick={onClose} className="rounded p-1 text-ink-subtle hover:text-danger transition-colors">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      {editor}
    </ResizableEditor>
  )
}
