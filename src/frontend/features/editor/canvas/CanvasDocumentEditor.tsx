"use client"

import { useCallback, useEffect, useRef } from "react"
import { CanvasEditor } from "./CanvasEditor"
import type { DocumentEditorProps } from "@/features/editor/TextDocumentEditor"
import type { CanvasDocument } from "@/lib/domain/canvasDocument"
import { imageFrameEmbeds } from "@/lib/domain/canvasRender"
import { canvasEndpoint, storeDrawing } from "@/lib/backend/api"
import { renderCanvasToJpeg } from "@/lib/canvasExport"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { baseName } from "@/lib/paths"
import { useDisplayName } from "@/contexts/ProjectContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useModeLabel } from "@/hooks/useModeLabel"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"

/** Database canvas: the collaboration hook owns loading and saving. */
export function CanvasDocumentEditor({ path, slug, onClose, onModeLabel }: DocumentEditorProps) {
  const { document: canvasDocument, replace, status, retry } = useCollaborativeCanvas(canvasEndpoint(path))
  const filename = baseName(path)
  const shownName = useDisplayName()(slug, path) ?? filename
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

  useModeLabel(shownName, true, onModeLabel)

  useEscapeLayer(onClose)

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
      />
    </>
  )
  return <div className="absolute inset-0 z-30 flex flex-col bg-paper">{editor}</div>
}
