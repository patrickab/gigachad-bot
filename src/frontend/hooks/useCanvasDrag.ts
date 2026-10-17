"use client"

import { useCallback, useEffect, useRef } from "react"
import type { CanvasAttachment, CanvasDocument, CanvasFrame, CanvasText } from "@/lib/canvasDocument"

type DragTarget = "frames" | "attachments" | "texts"
type DragEntity = CanvasFrame | CanvasAttachment | CanvasText
type DragMode = "move" | "resize"

const MIN_WIDTH: Record<DragTarget, number> = { frames: 120, attachments: 200, texts: 80 }
const MIN_HEIGHT: Record<"attachments" | "texts", number> = { attachments: 180, texts: 40 }

type ActiveDrag = {
  id: string; target: DragTarget; mode: DragMode
  startX: number; startY: number; origX: number; origY: number; origW: number; origH: number
}

// A move shifts the origin. A resize grows the width; text notes and the fixed-height
// attachments (architecture, document) also grow their height — frames and PDFs keep their aspect.
function dragPatch(d: ActiveDrag, entity: DragEntity, dx: number, dy: number): Partial<DragEntity> {
  if (d.mode === "move") return { x: d.origX + dx, y: d.origY + dy }
  const width = Math.max(MIN_WIDTH[d.target], d.origW + dx)
  const growsHeight = d.target === "texts" || (d.target === "attachments" && "kind" in entity && (entity.kind === "architecture-graph" || entity.kind === "document"))
  return growsHeight ? { width, height: Math.max(MIN_HEIGHT[d.target as "attachments" | "texts"], d.origH + dy) } : { width }
}

interface CanvasDragOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  scaleRef: { readonly current: number }
  snapshotOnce: () => void
  applyChange: (next: CanvasDocument) => void
  endGesture: () => void
  /** Offered every window pointermove first; returns true when a stroke-selection drag consumed it. */
  moveSelectionDrag: (e: PointerEvent) => boolean
  endSelectionDrag: (e: PointerEvent) => void
}

/**
 * Shared move / resize for frames, attachments and text notes, plus the window-level pointer
 * listeners that also drive stroke-selection drags. One undo step per drag.
 */
export function useCanvasDrag({ liveRef, scaleRef, snapshotOnce, applyChange, endGesture, moveSelectionDrag, endSelectionDrag }: CanvasDragOptions) {
  const dragRef = useRef<ActiveDrag | null>(null)

  const startDrag = useCallback((target: DragTarget, id: string, mode: DragMode, clientX: number, clientY: number) => {
    const item = (liveRef.current.doc[target] as DragEntity[]).find((e) => e.id === id)
    if (!item) return
    dragRef.current = {
      id, target, mode, startX: clientX, startY: clientY, origX: item.x, origY: item.y,
      origW: item.width, origH: "height" in item ? (item.height ?? 0) : 0,
    }
  }, [liveRef])

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      if (moveSelectionDrag(e)) return
      const d = dragRef.current
      if (!d) return
      const dx = (e.clientX - d.startX) / scaleRef.current
      const dy = (e.clientY - d.startY) / scaleRef.current
      if (dx === 0 && dy === 0) return
      snapshotOnce()
      const cur = liveRef.current.doc
      applyChange({ ...cur, [d.target]: (cur[d.target] as DragEntity[]).map((entity) => entity.id === d.id ? { ...entity, ...dragPatch(d, entity, dx, dy) } : entity) })
    }
    const onUp = (e: PointerEvent) => {
      endSelectionDrag(e)
      dragRef.current = null
      endGesture()
    }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
    return () => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
    }
  }, [liveRef, scaleRef, snapshotOnce, applyChange, endGesture, moveSelectionDrag, endSelectionDrag])

  return startDrag
}
