"use client"

import { useCallback, useEffect, useState, type RefObject } from "react"
import { canvasEntityId, type CanvasDocument } from "@/lib/domain/canvasDocument"
import type { Point } from "@/lib/domain/edgeRouting"
import { polyBounds, resizeBox, scalePoly, scaleStroke, strokesInLasso, type Poly, type SelBox } from "@/features/editor/canvas/selection/canvasSelection"
import { dropTargetAt, registerDropTarget, remapAcrossCanvases, type StrokeDrop } from "@/features/editor/canvas/attachments/canvasTransfer"
import type { StrokeData } from "@/lib/drawing"
import type { DragSession } from "./useCanvasDrag"

const COPY_OFFSET_PX = 24 // screen px a duplicated selection is nudged by, so it reads as a new object

interface CanvasSelectionOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  containerRef: RefObject<HTMLDivElement | null>
  scaleRef: { readonly current: number }
  offsetRef: { readonly current: Point }
  commit: (next: CanvasDocument) => void
  beginDrag: (session: DragSession, clientX: number, clientY: number) => void
}

/**
 * Select, move, scale, duplicate and hand over strokes. A freehand lasso picks strokes by id; once
 * released it becomes the bounding box of what it caught, which is what you then drag, resize or copy.
 */
export function useCanvasSelection({ liveRef, containerRef, scaleRef, offsetRef, commit, beginDrag }: CanvasSelectionOptions) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [lasso, setLasso] = useState<Poly | null>(null) // frozen shape backing the current selection
  // The strokes under an in-flight drag. While this is set the committed layer hides the
  // selected strokes and only this small list re-renders per frame; the document itself is
  // not touched until the gesture ends.
  const [preview, setPreview] = useState<StrokeData[] | null>(null)

  const clear = useCallback(() => {
    setSelectedIds((prev) => (prev.size === 0 ? prev : new Set()))
    setLasso(null)
  }, [])

  // Receive a selection dropped from another canvas: source canvas units → client px →
  // our canvas units, so the ink keeps its screen position and screen size.
  const dropStrokes = useCallback<StrokeDrop>((incoming, from) => {
    const el = containerRef.current
    if (!el) return
    const to = { rect: el.getBoundingClientRect(), scale: scaleRef.current, offset: offsetRef.current }
    const cur = liveRef.current.doc
    commit({ ...cur, strokes: [...cur.strokes, ...incoming.map((s) => remapAcrossCanvases(s, from, to))] })
  }, [containerRef, scaleRef, offsetRef, liveRef, commit])

  useEffect(() => {
    const el = containerRef.current
    return el ? registerDropTarget(el, dropStrokes) : undefined
  }, [containerRef, dropStrokes])

  /** Select the strokes a finished lasso loop covers; a loop too short to enclose anything leaves the selection alone. */
  const select = useCallback((points: Poly) => {
    if (points.length < 3) return
    const picked = strokesInLasso(liveRef.current.doc.strokes, points)
    setSelectedIds(picked)
    setLasso(picked.size > 0 ? points : null)
  }, [liveRef])

  // Originals are snapshotted at gesture start so every frame maps from the same
  // source — repeated relative deltas would drift.
  const startDrag = useCallback((handle: "move" | "resize", clientX: number, clientY: number) => {
    const originals = new Map<string, number[][]>()
    for (const s of liveRef.current.doc.strokes) {
      if (selectedIds.has(s.id)) originals.set(s.id, s.points.map((p) => [...p]))
    }
    const origBox = lasso && polyBounds(lasso)
    if (originals.size === 0 || !lasso || !origBox) return
    const origLasso = lasso.map((p) => [...p] as [number, number])
    let box: SelBox | null = null
    beginDrag({
      move: (dx, dy) => {
        // A move translates the shape's box by the delta; the handle grows it. Either way
        // the strokes AND the lasso outline are remapped from the original box into the new
        // one, so the outline keeps hugging the ink. `width` is left alone — scaling ink
        // must not change how thick the pen was.
        const next = handle === "move" ? { ...origBox, x: origBox.x + dx, y: origBox.y + dy } : resizeBox(origBox, dx, dy)
        box = next
        // Preview only: writing the document here would rebuild every stroke array and
        // re-serialize the whole canvas on every pointermove. The commit happens on release.
        setPreview(liveRef.current.doc.strokes.flatMap((stroke) => {
          const orig = originals.get(stroke.id)
          return orig ? [scaleStroke(stroke, orig, origBox, next)] : []
        }))
        // the outline (and the buttons anchored to it) rides along with the strokes
        setLasso(scalePoly(origLasso, origBox, next))
      },
      // Commit the drag, handing the ink to another canvas if it was released over one.
      end: (e) => {
        if (box) {
          const final = box
          const cur = liveRef.current.doc
          const moved = cur.strokes.map((stroke) => {
            const orig = originals.get(stroke.id)
            return orig ? scaleStroke(stroke, orig, origBox, final) : stroke
          })
          // Released over another canvas: hand the ink over and delete it here. Two
          // documents change, so it costs one undo step on each side.
          const el = containerRef.current
          const drop = handle === "move" ? dropTargetAt(e.clientX, e.clientY, el) : null
          if (drop && el) {
            commit({ ...cur, strokes: moved.filter((stroke) => !originals.has(stroke.id)) })
            drop(moved.filter((stroke) => originals.has(stroke.id)), { rect: el.getBoundingClientRect(), scale: scaleRef.current, offset: offsetRef.current })
            clear()
          } else {
            commit({ ...cur, strokes: moved })
          }
        }
        setPreview(null)
      },
    }, clientX, clientY)
  }, [liveRef, containerRef, scaleRef, offsetRef, selectedIds, lasso, beginDrag, commit, clear])

  const deleteSelected = useCallback(() => {
    if (selectedIds.size === 0) return
    const cur = liveRef.current.doc
    commit({ ...cur, strokes: cur.strokes.filter((s) => !selectedIds.has(s.id)) })
    clear()
  }, [liveRef, selectedIds, clear, commit])

  // Duplicate the selection, offset so the copy is visibly its own object, and hand the
  // selection over to it — the originals drop out, the copies are what you keep dragging.
  const copySelected = useCallback(() => {
    const cur = liveRef.current.doc
    if (selectedIds.size === 0 || !lasso) return
    const off = COPY_OFFSET_PX / scaleRef.current
    const copies = cur.strokes
      .filter((s) => selectedIds.has(s.id))
      .map((s) => ({ ...s, id: canvasEntityId(), points: s.points.map((p) => [p[0]! + off, p[1]! + off, p[2] ?? 0.5]) }))
    if (copies.length === 0) return
    commit({ ...cur, strokes: [...cur.strokes, ...copies] })
    setSelectedIds(new Set(copies.map((s) => s.id)))
    setLasso(lasso.map(([x, y]) => [x + off, y + off]))
  }, [liveRef, selectedIds, lasso, scaleRef, commit])

  return { selectedIds, lasso, bounds: lasso ? polyBounds(lasso) : null, preview, clear, select, startDrag, deleteSelected, copySelected }
}
