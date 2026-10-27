"use client"

import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react"
import type { CanvasDocument } from "@/lib/domain/canvasDocument"
import { inkOutlinePath } from "@/lib/drawing"
import type { InkSettings } from "@/features/editor/canvas/CanvasToolbar"
import { ERASER_RADIUS_PX, appendPointerSamples, finishStroke, newStroke, pointerInkRole, pointerSample, strokesAfterErase } from "@/features/editor/canvas/ink/canvasInk"
import type { Band, RubberBand } from "@/features/editor/canvas/selection/useRubberBand"

const STRAIGHTEN_HOLD_MS = 500 // hold the pen still mid-stroke to snap it into a straight line
const STRAIGHTEN_MOVE_TOLERANCE = 5 // px of client movement that resets the "holding still" timer
const LONG_PRESS_MS = 500 // touch/mouse press-and-hold opens the context menu
const LONG_PRESS_TOLERANCE = 5 // px of client movement that cancels a long press

/** The ink gesture in flight. Pointer capture can still deliver a late event from another device, so only the pointer that began a stroke may extend or finish it. */
type Gesture = { kind: "draw"; pointerId: number } | { kind: "erase" } | null

type ClientPoint = { clientX: number; clientY: number }

/** Where the long-press menu opens, in container px, and the canvas point it acts on. */
export type CanvasContextMenuAt = { left: number; top: number; canvasX: number; canvasY: number }

function movedBeyond(from: ClientPoint, to: ClientPoint, tolerance: number): boolean {
  const dx = to.clientX - from.clientX
  const dy = to.clientY - from.clientY
  return dx * dx + dy * dy > tolerance * tolerance
}

interface InkSurfaceOptions {
  containerRef: RefObject<HTMLDivElement | null>
  screenToCanvas: (clientX: number, clientY: number) => [number, number]
  scaleRef: { readonly current: number }
  liveRef: { readonly current: { doc: CanvasDocument } }
  applyChange: (next: CanvasDocument) => void
  commit: (next: CanvasDocument) => void
  snapshotOnce: () => void
  endGesture: () => void
  ink: InkSettings
  rubberBand: RubberBand
  /** Space held: the press pans instead of inking. */
  spaceDown: { readonly current: boolean }
  clearSelection: () => void
  /** A released band, with the canvas point it ended at. */
  onBandFinished: (band: Band, end: [number, number]) => void
}

/**
 * Pointer handling of the canvas's drawing layer: pen strokes, the eraser and rubber bands.
 *
 * - The in-progress stroke lives in a ref and renders imperatively through `livePathRef`, so a pointermove never re-renders the editor
 * - Holding the pen still mid-stroke snaps the stroke into a straight line from its start
 * - A long press opens the context menu; its timers are cleared on unmount
 */
export function useInkSurface({
  containerRef, screenToCanvas, scaleRef, liveRef, applyChange, commit, snapshotOnce, endGesture,
  ink, rubberBand, spaceDown, clearSelection, onBandFinished,
}: InkSurfaceOptions) {
  const { tool: inkTool, width: inkWidth, highlighting, color: inkStrokeColor } = ink
  const { tool, isBanding, startBand, extendBand, finishBand } = rubberBand

  const [gesture, setGesture] = useState<Gesture>(null)
  // The ref is the synchronous truth for the pointer handlers; the state drives rendering.
  const gestureRef = useRef<Gesture>(null)
  const writeGesture = useCallback((next: Gesture) => {
    gestureRef.current = next
    setGesture(next)
  }, [])

  const pointsRef = useRef<number[][]>([])
  const livePathRef = useRef<SVGPathElement>(null)
  const redrawLive = useCallback(() => {
    const pts = pointsRef.current
    livePathRef.current?.setAttribute("d", pts.length >= 2 ? inkOutlinePath(pts, inkWidth, highlighting) : "")
  }, [inkWidth, highlighting])

  // Straighten-on-hold: pausing mid-stroke for STRAIGHTEN_HOLD_MS snaps the in-progress
  // freehand stroke to a straight line from its start to the hold point.
  const straightenTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const straightenAnchor = useRef<ClientPoint | null>(null)
  const straightened = useRef(false)

  const armStraighten = useCallback((clientX: number, clientY: number) => {
    straightenAnchor.current = { clientX, clientY }
    clearTimeout(straightenTimer.current)
    straightenTimer.current = setTimeout(() => {
      const pts = pointsRef.current
      if (pts.length < 2) return
      straightened.current = true
      pointsRef.current = [pts[0]!, pts[pts.length - 1]!]
      redrawLive()
    }, STRAIGHTEN_HOLD_MS)
  }, [redrawLive])

  const cancelStraighten = useCallback(() => {
    clearTimeout(straightenTimer.current)
    straightenAnchor.current = null
    straightened.current = false
  }, [])

  // Drop the in-progress stroke, e.g. the tiny hold-dot a long press leaves behind.
  const abortStroke = useCallback(() => {
    writeGesture(null)
    pointsRef.current = []
    redrawLive()
    cancelStraighten()
  }, [writeGesture, redrawLive, cancelStraighten])

  const [contextMenu, setContextMenu] = useState<CanvasContextMenuAt | null>(null)
  const closeContextMenu = useCallback(() => setContextMenu(null), [])
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const longPressPos = useRef<ClientPoint | null>(null)

  // Fires the context menu after LONG_PRESS_MS without a cancelling move or release.
  const startLongPress = useCallback((clientX: number, clientY: number) => {
    longPressPos.current = { clientX, clientY }
    longPressTimer.current = setTimeout(() => {
      const [cx, cy] = screenToCanvas(clientX, clientY)
      abortStroke()
      const containerRect = containerRef.current?.getBoundingClientRect()
      setContextMenu({ left: clientX - (containerRect?.left ?? 0), top: clientY - (containerRect?.top ?? 0), canvasX: cx, canvasY: cy })
    }, LONG_PRESS_MS)
  }, [screenToCanvas, abortStroke, containerRef])

  const cancelLongPress = useCallback(() => {
    clearTimeout(longPressTimer.current)
    longPressPos.current = null
  }, [])

  useEffect(() => () => {
    clearTimeout(straightenTimer.current)
    clearTimeout(longPressTimer.current)
  }, [])

  const onPointerDown = useCallback((e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (spaceDown.current) return
    if (tool !== "draw") {
      if (e.button !== 0) return
      e.stopPropagation()
      e.currentTarget.setPointerCapture(e.pointerId)
      startBand(tool, screenToCanvas(e.clientX, e.clientY))
      return
    }
    // drawing/erasing directly on the canvas counts as "outside" the selection
    clearSelection()
    // The mouse inks only with a selected tool, otherwise its drag falls through to the container and pans.
    const role = pointerInkRole(e, inkTool)
    if (!role) return
    // The pen's eraser end (button 5) erases without claiming the press.
    if (e.button === 0) e.stopPropagation()
    if (role === "erase") { writeGesture({ kind: "erase" }); return }
    e.currentTarget.setPointerCapture(e.pointerId)
    writeGesture({ kind: "draw", pointerId: e.pointerId })
    pointsRef.current = [pointerSample(e, screenToCanvas)]
    redrawLive()
    // the long-press context menu (incl. "Paste image") is a touch affordance — a pen or
    // mouse stroke that pauses mid-draw must never be mistaken for a long-press
    if (e.pointerType !== "pen" && e.pointerType !== "mouse") startLongPress(e.clientX, e.clientY)
    armStraighten(e.clientX, e.clientY)
  }, [spaceDown, tool, inkTool, screenToCanvas, startBand, clearSelection, writeGesture, redrawLive, startLongPress, armStraighten])

  const onPointerMove = useCallback((e: ReactPointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (isBanding()) {
      extendBand(screenToCanvas(e.clientX, e.clientY))
      return
    }
    if (longPressPos.current && movedBeyond(longPressPos.current, e, LONG_PRESS_TOLERANCE)) cancelLongPress()
    const current = gestureRef.current
    switch (current?.kind) {
      case "erase": {
        const [ex, ey] = screenToCanvas(e.clientX, e.clientY)
        const cur = liveRef.current.doc
        const updated = strokesAfterErase(cur.strokes, ex, ey, ERASER_RADIUS_PX / scaleRef.current)
        if (updated.length !== cur.strokes.length) {
          snapshotOnce()
          applyChange({ ...cur, strokes: updated })
        }
        return
      }
      case "draw": {
        if (e.pointerId !== current.pointerId) return
        if (straightened.current) {
          pointsRef.current = [pointsRef.current[0]!, pointerSample(e, screenToCanvas)]
          redrawLive()
          return
        }
        if (straightenAnchor.current && movedBeyond(straightenAnchor.current, e, STRAIGHTEN_MOVE_TOLERANCE)) armStraighten(e.clientX, e.clientY)
        appendPointerSamples(pointsRef.current, e.nativeEvent, screenToCanvas)
        redrawLive()
      }
    }
  }, [isBanding, extendBand, screenToCanvas, cancelLongPress, liveRef, scaleRef, snapshotOnce, applyChange, redrawLive, armStraighten])

  // Also bound to pointercancel and lostpointercapture.
  const onPointerUp = useCallback((e: ReactPointerEvent<SVGSVGElement>) => {
    cancelLongPress()
    cancelStraighten()
    if (e.pointerType === "touch") return
    const band = finishBand()
    if (band) {
      onBandFinished(band, screenToCanvas(e.clientX, e.clientY))
      return
    }
    const current = gestureRef.current
    switch (current?.kind) {
      case "erase":
        writeGesture(null)
        endGesture()
        return
      case "draw": {
        if (e.pointerId !== current.pointerId) return
        writeGesture(null)
        const pts = finishStroke(pointsRef.current, e, screenToCanvas)
        pointsRef.current = []
        redrawLive()
        if (!pts) return
        const cur = liveRef.current.doc
        commit({ ...cur, strokes: [...cur.strokes, newStroke(pts, { color: inkStrokeColor, width: inkWidth, highlight: highlighting })] })
      }
    }
  }, [cancelLongPress, cancelStraighten, finishBand, onBandFinished, screenToCanvas, writeGesture, endGesture, redrawLive, liveRef, commit, inkStrokeColor, inkWidth, highlighting])

  return {
    svgHandlers: { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, onLostPointerCapture: onPointerUp },
    livePathRef,
    isDrawing: gesture?.kind === "draw",
    isErasing: gesture?.kind === "erase",
    contextMenu,
    closeContextMenu,
  }
}
