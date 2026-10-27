"use client"

import { useRef, useState } from "react"

/** Fixed-height container with a bottom drag handle (min 200px). */
export function ResizableEditor({ children }: { children: React.ReactNode }) {
  const [height, setHeight] = useState(400)
  const drag = useRef<{ startY: number; startHeight: number } | null>(null)

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault()
    drag.current = { startY: e.clientY, startHeight: height }
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }

  const onPointerMove = (e: React.PointerEvent) => {
    if (drag.current) setHeight(Math.max(200, drag.current.startHeight + (e.clientY - drag.current.startY)))
  }

  const onPointerUp = () => { drag.current = null }

  return (
    <div className="flex flex-col" style={{ height }}>
      {children}
      <div
        className="h-1.5 cursor-row-resize shrink-0 flex items-center justify-center hover:bg-surface/60 transition-colors"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <div className="w-8 h-0.5 rounded-full bg-divider-strong" />
      </div>
    </div>
  )
}
