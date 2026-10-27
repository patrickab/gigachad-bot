"use client"

import { cn } from "@/lib/utils"

/** Bottom-right corner grip of an HTML canvas element; dragging it resizes. */
export function CanvasResizeHandle({ onDragStart, className }: {
  onDragStart: (mode: "resize", clientX: number, clientY: number) => void
  className?: string
}) {
  return (
    <div
      className={cn("absolute bottom-0 right-0 cursor-nwse-resize", className)}
      onPointerDown={(e) => { e.stopPropagation(); onDragStart("resize", e.clientX, e.clientY) }}
    >
      <svg className="w-full h-full text-ink-faint" viewBox="0 0 16 16">
        <path d="M14 2L2 14M14 6L6 14M14 10L10 14" stroke="currentColor" strokeWidth="1.5" fill="none" />
      </svg>
    </div>
  )
}
