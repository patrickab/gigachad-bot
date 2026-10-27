"use client"

import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { isToolButton, type RubberBand } from "@/features/editor/canvas/selection/useRubberBand"

interface CanvasPointerOptions {
  /** Gates the window key listeners to the frontmost tab. */
  active: boolean
  screenToCanvas: (clientX: number, clientY: number) => [number, number]
  /** Last canvas point under the pointer, where typing starts a text note. */
  pointerPointRef: { current: [number, number] | null }
  rubberBand: RubberBand
  clearSelection: () => void
  startPan: (e: ReactPointerEvent) => void
  movePan: (e: ReactPointerEvent) => void
  endPan: () => void
}

/** Pointer handling of the canvas container: focus, pan, the pen side button and the typing position. */
export function useCanvasPointer({ active, screenToCanvas, pointerPointRef, rubberBand, clearSelection, startPan, movePan, endPan }: CanvasPointerOptions) {
  const { tool, toolButtonDown, toolButtonUp } = rubberBand
  const spaceDown = useRef(false)
  useEventListener("window", "keydown", (e) => { if (e.code === "Space" && !e.repeat) spaceDown.current = true }, { active })
  useEventListener("window", "keyup", (e) => { if (e.code === "Space") spaceDown.current = false }, { active })

  // Hide the cursor for pen input without updating state on every move.
  const [pointerIsPen, setPointerIsPen] = useState(false)
  const lastPointerTypeRef = useRef<string>("mouse")

  const onPointerDown = useCallback((e: ReactPointerEvent) => {
    if (e.pointerType === "touch") return
    ;(e.currentTarget as HTMLElement).focus({ preventScroll: true })
    pointerPointRef.current = screenToCanvas(e.clientX, e.clientY)
    if (tool !== "draw") return
    // any pointerdown that bubbles this far didn't hit the selection lasso or its
    // handles (they stopPropagation) — so it's "outside" and drops the selection
    clearSelection()
    if (isToolButton(e)) {
      // pen side button: hold = activate selection mode & start the lasso right on this
      // same press (no lift-and-retouch needed), double-click = activate screenshot mode
      e.preventDefault()
      ;(e.target as Element).setPointerCapture(e.pointerId)
      toolButtonDown(e.clientX, e.clientY)
      return
    }
    if ((e.button === 0 && (spaceDown.current || e.pointerType === "mouse")) || (e.button === 2 && e.pointerType === "mouse")) startPan(e)
  }, [tool, pointerPointRef, screenToCanvas, clearSelection, toolButtonDown, startPan])

  const onPointerMove = useCallback((e: ReactPointerEvent) => {
    if (e.pointerType === "touch") return
    pointerPointRef.current = screenToCanvas(e.clientX, e.clientY)
    if (e.pointerType !== lastPointerTypeRef.current) {
      lastPointerTypeRef.current = e.pointerType
      setPointerIsPen(e.pointerType === "pen")
    }
    movePan(e)
  }, [pointerPointRef, screenToCanvas, movePan])

  const onPointerUp = useCallback((e: ReactPointerEvent) => {
    if (e.pointerType === "touch") return
    if (isToolButton(e)) { toolButtonUp(); return }
    endPan()
  }, [toolButtonUp, endPan])

  return { spaceDown, pointerIsPen, containerHandlers: { onPointerDown, onPointerMove, onPointerUp } }
}
