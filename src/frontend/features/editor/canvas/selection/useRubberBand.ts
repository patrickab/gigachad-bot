"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type { Poly } from "@/features/editor/canvas/selection/canvasSelection"
import type { CanvasTool } from "@/features/editor/canvas/CanvasToolbar"

const PEN_HOLD_MS = 400 // press-and-hold the pen button to activate selection mode
const PEN_DOUBLE_CLICK_MS = 350 // double-click the pen button to activate screenshot mode

/** A rubber-band drag in canvas units: the lasso traces every point, the screenshot rectangle uses its first and last. */
export type Band = { tool: Exclude<CanvasTool, "draw">; points: Poly }

/** The middle mouse button or a pen's side button: both drive the hold and double-click tool gestures. */
export function isToolButton(e: { button: number; pointerType: string }): boolean {
  return e.button === 1 || (e.pointerType === "pen" && e.button === 2)
}

/** The pointer tool and its band, as the editor and the ink surface use it. */
export interface RubberBand {
  tool: CanvasTool
  band: Band | null
  /** Whether a band is being dragged, read synchronously by pointer handlers. */
  isBanding: () => boolean
  startBand: (tool: Band["tool"], point: [number, number]) => void
  extendBand: (point: [number, number]) => void
  /** Drops any band and returns to drawing. */
  cancelBand: () => void
  /** The band being released, if any; finishing it returns to drawing. */
  finishBand: () => Band | null
  toggleTool: (tool: Band["tool"]) => void
  /** Tool-button press and release at a client point: hold selects, a double-click picks screenshot. */
  toolButtonDown: (clientX: number, clientY: number) => void
  toolButtonUp: () => void
}

/**
 * The pointer tool and its rubber band: a lasso (select) or a rectangle (screenshot) dragged on the canvas.
 * The pen side button picks the tool: hold activates selection and starts the lasso on the same press, a double-click activates screenshot.
 * Its timers are cleared on unmount.
 */
export function useRubberBand(screenToCanvas: (clientX: number, clientY: number) => [number, number]): RubberBand {
  const [tool, setTool] = useState<CanvasTool>("draw")
  // The ref is the synchronous truth for the pointer handlers; the state drives rendering.
  const [band, setBand] = useState<Band | null>(null)
  const bandRef = useRef<Band | null>(null)
  const writeBand = useCallback((next: Band | null) => {
    bandRef.current = next
    setBand(next)
  }, [])

  const isBanding = useCallback(() => bandRef.current !== null, [])
  const startBand = useCallback((bandTool: Band["tool"], point: [number, number]) => writeBand({ tool: bandTool, points: [point] }), [writeBand])
  const extendBand = useCallback((point: [number, number]) => {
    const b = bandRef.current
    if (b) writeBand({ tool: b.tool, points: b.tool === "select" ? [...b.points, point] : [b.points[0]!, point] })
  }, [writeBand])

  const cancelBand = useCallback(() => {
    writeBand(null)
    setTool("draw")
  }, [writeBand])

  const finishBand = useCallback((): Band | null => {
    const finished = bandRef.current
    if (finished) cancelBand()
    return finished
  }, [cancelBand])

  const toggleTool = useCallback((next: Band["tool"]) => setTool((current) => current === next ? "draw" : next), [])

  const penHoldTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const penHoldFired = useRef(false)
  const penClickPending = useRef(false)
  const penClickTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => {
    clearTimeout(penHoldTimer.current)
    clearTimeout(penClickTimer.current)
  }, [])

  const toolButtonDown = useCallback((clientX: number, clientY: number) => {
    penHoldFired.current = false
    clearTimeout(penHoldTimer.current)
    penHoldTimer.current = setTimeout(() => {
      penHoldFired.current = true
      setTool("select")
      startBand("select", screenToCanvas(clientX, clientY))
    }, PEN_HOLD_MS)
  }, [screenToCanvas, startBand])

  const toolButtonUp = useCallback(() => {
    clearTimeout(penHoldTimer.current)
    if (penHoldFired.current) { penHoldFired.current = false; return }
    if (penClickPending.current) {
      penClickPending.current = false
      clearTimeout(penClickTimer.current)
      setTool("screenshot")
    } else {
      penClickPending.current = true
      penClickTimer.current = setTimeout(() => { penClickPending.current = false }, PEN_DOUBLE_CLICK_MS)
    }
  }, [])

  return { tool, band, isBanding, startBand, extendBand, cancelBand, finishBand, toggleTool, toolButtonDown, toolButtonUp }
}
