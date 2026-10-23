"use client"

import { ReactNode, useCallback, useEffect, useRef } from "react"
import { motion, useMotionValue, useTransform, animate } from "framer-motion"
import { cn } from "@/lib/utils"

const MIN_SIDEBAR_WIDTH = 200
export const MAX_SIDEBAR_WIDTH = 600

/** Spring-animated width of a right-anchored panel, plus a drag handler for its left edge. */
function useResizableWidth(width: number, onWidthChange: (w: number) => void, maxWidth: number) {
  /** Ends the running drag's listeners and body styles, null while not dragging. */
  const stopDrag = useRef<(() => void) | null>(null)
  const motionWidth = useMotionValue(width)

  // Spring to external width changes, but never fight an active drag
  useEffect(() => {
    if (stopDrag.current) return
    const controls = animate(motionWidth, width, { type: "spring", stiffness: 400, damping: 40, mass: 0.6 })
    return controls.stop
  }, [width, motionWidth])

  const widthStyle = useTransform(motionWidth, (v) => `${v}px`)
  // A drag cut short by unmounting (e.g. closing the tab) must not leave the app unselectable
  useEffect(() => () => stopDrag.current?.(), [])

  // Pointer capture keeps the drag on the handle while the pointer crosses an iframe (omp frame, previews),
  // which would otherwise swallow the moves and the release.
  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    const handle = e.currentTarget
    handle.setPointerCapture(e.pointerId)
    const startX = e.clientX
    const startWidth = motionWidth.get()
    const clamp = (px: number) => Math.min(maxWidth, Math.max(MIN_SIDEBAR_WIDTH, px))
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"

    // Right-anchored, so dragging left widens the panel
    const onPointerMove = (ev: PointerEvent) => motionWidth.set(clamp(startWidth + (startX - ev.clientX)))
    const stop = () => {
      stopDrag.current = null
      handle.removeEventListener("pointermove", onPointerMove)
      handle.removeEventListener("lostpointercapture", onCaptureEnd)
      document.body.style.cursor = ""
      document.body.style.userSelect = ""
    }
    // Capture ends on release and on cancel alike
    const onCaptureEnd = () => {
      stop()
      const finalWidth = clamp(motionWidth.get())
      motionWidth.set(finalWidth)
      onWidthChange(finalWidth)
    }
    stopDrag.current = stop
    handle.addEventListener("pointermove", onPointerMove)
    handle.addEventListener("lostpointercapture", onCaptureEnd)
  }, [motionWidth, onWidthChange, maxWidth])

  return { widthStyle, handlePointerDown }
}

interface SidePanelFrameProps {
  width: number
  onWidthChange: (w: number) => void
  maxWidth?: number
  className?: string
  children: ReactNode
}

/** Right-anchored panel with a draggable left edge; callers own the width and the surface styling. */
export function SidePanelFrame({ width, onWidthChange, maxWidth = MAX_SIDEBAR_WIDTH, className, children }: SidePanelFrameProps) {
  const { widthStyle, handlePointerDown } = useResizableWidth(width, onWidthChange, maxWidth)
  return (
    <motion.aside
      initial={false}
      style={{ width: widthStyle }}
      className={cn("shrink-0 flex flex-col overflow-hidden relative", className)}
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-surface-elevated active:bg-surface-elevated focus-visible:bg-surface-elevated transition-colors z-10"
        onPointerDown={handlePointerDown}
      />
      {children}
    </motion.aside>
  )
}
