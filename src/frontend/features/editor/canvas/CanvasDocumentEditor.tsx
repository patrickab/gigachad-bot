"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Download, X } from "lucide-react"
import { CanvasEditor } from "./CanvasEditor"
import { ResizableEditor } from "@/features/editor/ResizableEditor"
import type { DocumentEditorProps } from "@/features/editor/TextDocumentEditor"
import { A4_ASPECT, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { imageFrameEmbeds } from "@/lib/domain/canvasRender"
import { storeDrawing, writeBinaryDocument } from "@/lib/backend/api"
import { renderPageToPng, renderCanvasToJpeg } from "@/lib/canvasExport"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { baseName } from "@/lib/paths"
import { useDisplayName } from "@/contexts/ProjectContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useModeLabel } from "@/hooks/useModeLabel"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"

/** Database canvas: the collaboration hook owns loading and saving. */
export function CanvasDocumentEditor({ path, slug, onClose, onSaved, overlay, onModeLabel }: DocumentEditorProps) {
  const { document: canvasDocument, replace, status, retry } = useCollaborativeCanvas(path)
  const filename = baseName(path)
  const shownName = useDisplayName()(slug, path) ?? filename
  const [exporting, setExporting] = useState(false)
  // Only local edits refresh the drawing JPEG, otherwise every open client would
  // re-render and upload it for each remote change.
  const drawnRef = useRef(false)

  const handleChange = useCallback((next: CanvasDocument) => {
    drawnRef.current = true
    replace(next)
  }, [replace])

  useEffect(() => {
    if (!canvasDocument) return
    const timer = setTimeout(() => {
      const { strokes, texts, frames } = canvasDocument
      if (!drawnRef.current || (strokes.length === 0 && texts.length === 0 && !frames.some((f) => f.kind === "image"))) return
      drawnRef.current = false
      renderCanvasToJpeg(strokes, 20, imageFrameEmbeds(frames), texts)
        .then((blob) => storeDrawing(filename.replace(/\.canvas$/, ".jpg"), blob))
        .catch(() => {})
    }, 1000)
    return () => clearTimeout(timer)
  }, [filename, canvasDocument])

  useModeLabel(shownName, !!overlay, onModeLabel)

  const handleExportPdf = useCallback(async () => {
    if (!canvasDocument || exporting) return
    const pages = canvasDocument.frames.filter((f) => f.kind === "page")
    if (pages.length === 0) return
    setExporting(true)
    try {
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
      onSaved?.()
    } catch { /* */ }
    setExporting(false)
  }, [canvasDocument, exporting, slug, filename, onSaved])

  useEscapeLayer(onClose, !!overlay)

  if (!canvasDocument) {
    return status === "error"
      ? <LoadFailure message="Failed to load this canvas. Its stored content was left untouched." onRetry={retry} />
      : <LoadingPlaceholder />
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
      />
    </>
  )
  if (overlay) return <div className="absolute inset-0 z-30 flex flex-col bg-paper">{editor}</div>
  return (
    <ResizableEditor>
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-divider/50 shrink-0">
        <span className="text-[11px] font-medium text-ink truncate">{shownName}</span>
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
