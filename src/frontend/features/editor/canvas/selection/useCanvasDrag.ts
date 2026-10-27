"use client"

import { useCallback, useRef } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { attachmentTraits } from "@/lib/domain/canvasAttachments"
import type { CanvasDocument } from "@/lib/domain/canvasDocument"

type DragTarget = "frames" | "attachments" | "texts"
type DragMode = "move" | "resize"
type DragPatch = { x?: number; y?: number; width?: number; height?: number }

/** One pointer drag: `move` gets the travel since the press in canvas units, `end` the release. */
export interface DragSession {
  move: (dx: number, dy: number) => void
  end?: (e: PointerEvent) => void
}

const MIN_WIDTH: Record<DragTarget, number> = { frames: 120, attachments: 200, texts: 80 }
const MIN_HEIGHT: Record<"attachments" | "texts", number> = { attachments: 180, texts: 40 }

const withPatch = <T extends { id: string }>(items: T[], id: string, patch: DragPatch): T[] => items.map((item) => item.id === id ? { ...item, ...patch } : item)

/** The document with `patch` applied to entity `id` of `target`. */
function patchEntity(doc: CanvasDocument, target: DragTarget, id: string, patch: DragPatch): CanvasDocument {
  switch (target) {
    case "frames": return { ...doc, frames: withPatch(doc.frames, id, patch) }
    case "attachments": return { ...doc, attachments: withPatch(doc.attachments, id, patch) }
    case "texts": return { ...doc, texts: withPatch(doc.texts, id, patch) }
  }
}

interface CanvasDragOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  scaleRef: { readonly current: number }
  snapshotOnce: () => void
  applyChange: (next: CanvasDocument) => void
  endGesture: () => void
}

/**
 * Window-level pointer drags: one session at a time follows the pointer until release, which also closes the history gesture.
 * `startDrag` moves or resizes frames, attachments and text notes, one undo step per drag.
 */
export function useCanvasDrag({ liveRef, scaleRef, snapshotOnce, applyChange, endGesture }: CanvasDragOptions) {
  const sessionRef = useRef<{ session: DragSession; startX: number; startY: number } | null>(null)

  const beginDrag = useCallback((session: DragSession, clientX: number, clientY: number) => {
    sessionRef.current = { session, startX: clientX, startY: clientY }
  }, [])

  // A move shifts the origin. A resize grows the width; text notes and the fixed-height
  // attachments (architecture, document) also grow their height, frames and PDFs keep their aspect.
  const startDrag = useCallback((target: DragTarget, id: string, mode: DragMode, clientX: number, clientY: number) => {
    const doc = liveRef.current.doc
    const item = doc[target].find((entity) => entity.id === id)
    if (!item) return
    const origH = "height" in item ? (item.height ?? 0) : 0
    const growsHeight = target === "texts" || (target === "attachments" && doc.attachments.some((a) => a.id === id && attachmentTraits[a.kind].fixedHeight))
    const resize = (dx: number, dy: number): DragPatch => {
      const width = Math.max(MIN_WIDTH[target], item.width + dx)
      return growsHeight ? { width, height: Math.max(MIN_HEIGHT[target], origH + dy) } : { width }
    }
    beginDrag({
      move: (dx, dy) => {
        snapshotOnce()
        applyChange(patchEntity(liveRef.current.doc, target, id, mode === "move" ? { x: item.x + dx, y: item.y + dy } : resize(dx, dy)))
      },
    }, clientX, clientY)
  }, [liveRef, beginDrag, snapshotOnce, applyChange])

  useEventListener("window", "pointermove", (e) => {
    const active = sessionRef.current
    if (!active) return
    const dx = (e.clientX - active.startX) / scaleRef.current
    const dy = (e.clientY - active.startY) / scaleRef.current
    if (dx === 0 && dy === 0) return
    active.session.move(dx, dy)
  })
  useEventListener("window", "pointerup", (e) => {
    const active = sessionRef.current
    sessionRef.current = null
    active?.session.end?.(e)
    endGesture()
  })

  return { beginDrag, startDrag }
}
