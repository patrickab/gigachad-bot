"use client"

import { ReactNode, useCallback, useEffect, useRef } from "react"
import { motion, useMotionValue, useTransform, animate } from "framer-motion"
import { cn } from "@/lib/utils"

const MIN_SIDEBAR_WIDTH = 200
export const MAX_SIDEBAR_WIDTH = 600

/** Spring-animated width of a right-anchored panel, plus a drag handler for its left edge. */
function useResizableWidth(width: number, onWidthChange: (w: number) => void, maxWidth: number) {
  const dragging = useRef(false)
  const motionWidth = useMotionValue(width)

  // Spring to external width changes, but never fight an active drag
  useEffect(() => {
    if (dragging.current) return
    const controls = animate(motionWidth, width, { type: "spring", stiffness: 400, damping: 40, mass: 0.6 })
    return controls.stop
  }, [width, motionWidth])

  const widthStyle = useTransform(motionWidth, (v) => `${v}px`)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    const startX = e.clientX
    const startWidth = motionWidth.get()
    const clamp = (px: number) => Math.min(maxWidth, Math.max(MIN_SIDEBAR_WIDTH, px))

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return
      // Right-anchored, so dragging left widens the panel
      motionWidth.set(clamp(startWidth + (startX - ev.clientX)))
    }
    const onMouseUp = () => {
      if (!dragging.current) return
      dragging.current = false
      const finalWidth = clamp(motionWidth.get())
      motionWidth.set(finalWidth)
      onWidthChange(finalWidth)
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }
    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [motionWidth, onWidthChange, maxWidth])

  return { widthStyle, handleMouseDown }
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
  const { widthStyle, handleMouseDown } = useResizableWidth(width, onWidthChange, maxWidth)
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
        onMouseDown={handleMouseDown}
      />
      {children}
    </motion.aside>
  )
}
