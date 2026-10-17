"use client"

import { useCallback, useEffect, useRef, useState, type RefObject } from "react"
import { canvasEntityId, type CanvasDocument } from "@/lib/canvasDocument"
import { polyBounds, resizeBox, scalePoly, scaleStroke, strokesInLasso, type Poly, type SelBox } from "@/lib/canvasSelection"
import { dropTargetAt, registerDropTarget, remapAcrossCanvases, type StrokeDrop } from "@/lib/canvasTransfer"
import type { StrokeData } from "@/lib/drawing"

const COPY_OFFSET_PX = 24 // screen px a duplicated selection is nudged by, so it reads as a new object

type SelectionDrag = {
  handle: "move" | "resize"
  startX: number
  startY: number
  originals: Map<string, number[][]>
  origLasso: Poly
  origBox: SelBox
  box: SelBox | null
}

interface CanvasSelectionOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  containerRef: RefObject<HTMLDivElement | null>
  scaleRef: { readonly current: number }
  offsetRef: { readonly current: { x: number; y: number } }
  commit: (next: CanvasDocument) => void
}

/**
 * Select, move, scale, duplicate and hand over strokes. A freehand lasso picks strokes by id; once
 * released it becomes the bounding box of what it caught, which is what you then drag, resize or copy.
 */
export function useCanvasSelection({ liveRef, containerRef, scaleRef, offsetRef, commit }: CanvasSelectionOptions) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [lasso, setLasso] = useState<Poly | null>(null) // frozen shape backing the current selection
  // The strokes under an in-flight drag. While this is set the committed layer hides the
  // selected strokes and only this small list re-renders per frame; the document itself is
  // not touched until the gesture ends.
  const [preview, setPreview] = useState<StrokeData[] | null>(null)
  const dragRef = useRef<SelectionDrag | null>(null)

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
    dragRef.current = { handle, startX: clientX, startY: clientY, originals, origLasso: lasso.map((p) => [...p] as [number, number]), origBox, box: null }
  }, [liveRef, selectedIds, lasso])

  /** Window pointermove: returns whether a selection drag is in flight, so the caller can skip entity drags. */
  const moveDrag = useCallback((e: PointerEvent): boolean => {
    const s = dragRef.current
    if (!s) return false
    const dx = (e.clientX - s.startX) / scaleRef.current
    const dy = (e.clientY - s.startY) / scaleRef.current
    if (dx === 0 && dy === 0) return true
    // A move translates the shape's box by the delta; the handle grows it. Either way
    // the strokes AND the lasso outline are remapped from the original box into the new
    // one, so the outline keeps hugging the ink. `width` is left alone — scaling ink
    // must not change how thick the pen was.
    const box = s.handle === "move"
      ? { ...s.origBox, x: s.origBox.x + dx, y: s.origBox.y + dy }
      : resizeBox(s.origBox, dx, dy)
    s.box = box
    // Preview only: writing the document here would rebuild every stroke array and
    // re-serialize the whole canvas on every pointermove. The commit happens on release.
    setPreview(liveRef.current.doc.strokes.flatMap((stroke) => {
      const orig = s.originals.get(stroke.id)
      return orig ? [scaleStroke(stroke, orig, s.origBox, box)] : []
    }))
    // the outline (and the buttons anchored to it) rides along with the strokes
    setLasso(scalePoly(s.origLasso, s.origBox, box))
    return true
  }, [liveRef, scaleRef])

  /** Window pointerup: commit the drag, handing the ink to another canvas if it was released over one. */
  const endDrag = useCallback((e: PointerEvent) => {
    const s = dragRef.current
    if (s?.box) {
      const cur = liveRef.current.doc
      const box = s.box
      const moved = cur.strokes.map((stroke) => {
        const orig = s.originals.get(stroke.id)
        return orig ? scaleStroke(stroke, orig, s.origBox, box) : stroke
      })
      // Released over another canvas: hand the ink over and delete it here. Two
      // documents change, so it costs one undo step on each side.
      const el = containerRef.current
      const drop = s.handle === "move" ? dropTargetAt(e.clientX, e.clientY, el) : null
      if (drop && el) {
        commit({ ...cur, strokes: moved.filter((stroke) => !s.originals.has(stroke.id)) })
        drop(moved.filter((stroke) => s.originals.has(stroke.id)), { rect: el.getBoundingClientRect(), scale: scaleRef.current, offset: offsetRef.current })
        clear()
      } else {
        commit({ ...cur, strokes: moved })
      }
    }
    setPreview(null)
    dragRef.current = null
  }, [liveRef, containerRef, scaleRef, offsetRef, commit, clear])

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

  return { selectedIds, lasso, bounds: lasso ? polyBounds(lasso) : null, preview, clear, select, startDrag, moveDrag, endDrag, deleteSelected, copySelected }
}
