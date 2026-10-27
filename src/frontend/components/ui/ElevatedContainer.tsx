"use client"

import React, { createContext, useContext } from "react"
import { cn } from "@/lib/utils"

// One role token per nesting level, so level 1 matches every other --surface panel (sidebar,
// composer). Deeper nesting clamps to the top rung.
const ELEVATION_LADDER = ["var(--paper)", "var(--surface)", "var(--surface-elevated)"]

const ElevationContext = createContext(0)

/** Reset the nesting level to zero for the outermost `ElevatedContainer`s below. */
export function ElevationProvider({ children }: { children: React.ReactNode }) {
  return <ElevationContext.Provider value={0}>{children}</ElevationContext.Provider>
}

interface ElevatedContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Lifts the surface with a deeper shadow on hover (for cards). */
  hoverLift?: boolean
  ref?: React.Ref<HTMLDivElement>
}

/** A surface whose background climbs one `ELEVATION_LADDER` rung per nesting level. */
export function ElevatedContainer({ hoverLift = false, className, style, children, ref, ...props }: ElevatedContainerProps) {
  const level = useContext(ElevationContext)

  return (
    <ElevationContext.Provider value={level + 1}>
      <div
        ref={ref}
        style={{ backgroundColor: ELEVATION_LADDER[Math.min(level, ELEVATION_LADDER.length - 1)], ...style }}
        className={cn(
          "relative transition-all duration-200 text-ink",
          level === 1 && "shadow-[var(--shadow-sm),var(--inner-highlight)]",
          level >= 2 && "shadow-[var(--shadow-md),var(--inner-highlight)]",
          hoverLift && "hover:shadow-[var(--shadow-xl),var(--inner-highlight-strong)] hover:z-10",
          className
        )}
        {...props}
      >
        {children}
      </div>
    </ElevationContext.Provider>
  )
}
