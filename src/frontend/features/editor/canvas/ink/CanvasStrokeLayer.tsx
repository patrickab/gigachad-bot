"use client"

import { memo } from "react"
import { inkColor, strokePath } from "@/features/editor/canvas/ink/canvasInk"
import type { StrokeData } from "@/lib/drawing"

// The committed ink is by far the biggest thing on the canvas, and it does not change
// when you pan, zoom, type in a note or drag a frame — memoizing it keeps those off an
// O(strokes) reconciliation. `hidden` lets a drag preview take a few strokes over
// without touching the layer, so a selection drag re-renders k strokes, not all of them.
// Highlights share one translucent group under the pen ink, so overlaps don't stack darker.
export const CanvasStrokeLayer = memo(function CanvasStrokeLayer({ strokes, hidden }: { strokes: StrokeData[]; hidden: Set<string> | null }) {
  const paths = (highlight: boolean) => strokes.map((stroke) => {
    if (!!stroke.highlight !== highlight || hidden?.has(stroke.id)) return null
    const d = strokePath(stroke)
    return d ? <path key={stroke.id} d={d} fill={inkColor(stroke.color)} /> : null
  })
  return (
    <>
      <g className="ink-highlight">{paths(true)}</g>
      {paths(false)}
    </>
  )
})
