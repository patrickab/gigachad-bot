"use client"

import { useId } from "react"

const GRID_SPACING = 28 // canvas units between dots at the finest grid level
const GRID_MIN_SCREEN_SPACING = 18
const DOT_RADIUS = 0.8

/** Keep a quiet, world-anchored grid with eased fades between density levels. */
export function CanvasDotGrid({ offset, scale }: { offset: { x: number; y: number }; scale: number }) {
  const patternId = useId()
  let spacing = GRID_SPACING * scale
  while (spacing < GRID_MIN_SCREEN_SPACING) spacing *= 2
  const fade = Math.min(1, spacing / GRID_MIN_SCREEN_SPACING - 1)
  const fadingOpacity = fade * fade * (3 - 2 * fade)
  const opacity = 0.85 + 0.1 * Math.min(1, scale)
  const cell = 2 * spacing
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
      <defs>
        <pattern id={patternId} x={offset.x % cell} y={offset.y % cell} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <g className="fill-ink-muted dark:fill-ink-faint" opacity={opacity}>
            <circle cx={DOT_RADIUS} cy={DOT_RADIUS} r={DOT_RADIUS} />
            <g opacity={fadingOpacity}>
              <circle cx={spacing + DOT_RADIUS} cy={DOT_RADIUS} r={DOT_RADIUS} />
              <circle cx={DOT_RADIUS} cy={spacing + DOT_RADIUS} r={DOT_RADIUS} />
              <circle cx={spacing + DOT_RADIUS} cy={spacing + DOT_RADIUS} r={DOT_RADIUS} />
            </g>
          </g>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${patternId})`} />
    </svg>
  )
}
