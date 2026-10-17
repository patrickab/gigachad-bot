"use client"

import { useId } from "react"

const GRID_SPACING = 30 // canvas units between dots at the finest grid level
const GRID_MIN_SCREEN_SPACING = 12 // px — below this the grid doubles its spacing

/** Dot grid with a constant on-screen density.
 *
 * - Zooming out doubles the spacing whenever dots would sit closer than `GRID_MIN_SCREEN_SPACING`
 * - Dots dropped by the next level fade out first, so levels blend instead of popping
 * - Dots keep a fixed screen size, the surviving ones stay put in canvas space
 */
export function CanvasDotGrid({ offset, scale }: { offset: { x: number; y: number }; scale: number }) {
  const patternId = useId()
  let spacing = GRID_SPACING * scale
  while (spacing < GRID_MIN_SCREEN_SPACING) spacing *= 2
  const fadingOpacity = Math.min(1, spacing / GRID_MIN_SCREEN_SPACING - 1)
  const cell = 2 * spacing
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
      <defs>
        <pattern id={patternId} x={offset.x % cell} y={offset.y % cell} width={cell} height={cell} patternUnits="userSpaceOnUse">
          <g fill="var(--ink-faint)" opacity={0.7}>
            <circle cx={1.2} cy={1.2} r={1.2} />
            <g opacity={fadingOpacity}>
              <circle cx={spacing + 1.2} cy={1.2} r={1.2} />
              <circle cx={1.2} cy={spacing + 1.2} r={1.2} />
              <circle cx={spacing + 1.2} cy={spacing + 1.2} r={1.2} />
            </g>
          </g>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${patternId})`} />
    </svg>
  )
}
