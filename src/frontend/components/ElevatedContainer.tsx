"use client"

import React, { createContext, useContext, useState } from "react"
import { cn } from "@/lib/utils"

// One role token per nesting level, so level 1 matches every other --surface panel (sidebar,
// composer). Deeper nesting clamps to the top rung.
const ELEVATION_LADDER = ["var(--paper)", "var(--surface)", "var(--surface-elevated)"]

const ElevationContext = createContext<number | null>(null)

/** Set the nesting level of the outermost `ElevatedContainer`s below. */
export function ElevationProvider({ children, startLevel = 0 }: { children: React.ReactNode; startLevel?: number }) {
  return <ElevationContext.Provider value={startLevel}>{children}</ElevationContext.Provider>
}

function ladderColor(level: number): string {
  return ELEVATION_LADDER[Math.min(level, ELEVATION_LADDER.length - 1)]
}

export interface ElevatedContainerProps extends React.HTMLAttributes<HTMLElement> {
  children?: React.ReactNode
  as?: React.ElementType     // Any HTML tag or component (aside, section, motion.div, etc.)
  asButton?: boolean         // For backwards compatibility
  castShadow?: boolean       // Whether to show a default flat shadow
  hoverLift?: boolean        // Opt-in for dynamic hover lifting & downward shadow-casting (for cards)
  disabled?: boolean
  type?: "button" | "submit" | "reset"
  onContextMenu?: React.MouseEventHandler<any>
}

/** A surface whose background climbs one `ELEVATION_LADDER` rung per nesting level. */
export const ElevatedContainer = React.forwardRef<any, ElevatedContainerProps>(
  (
    {
      children,
      className,
      as,
      asButton = false,
      castShadow = true,
      hoverLift = false,      // Off by default to keep flat interfaces (like the chat feed) flat
      style,
      ...props
    },
    ref
  ) => {
    const parentLevel = useContext(ElevationContext)
    const [hovered, setHovered] = useState(false)
    const [focused, setFocused] = useState(false)

    const currentLevel = parentLevel ?? 0

    // Hover or keyboard focus shifts 16% towards the next rung
    const isActive = (hovered || focused) && asButton
    const backgroundColor = isActive
      ? `color-mix(in srgb, ${ladderColor(currentLevel)}, ${ladderColor(currentLevel + 1)} 16%)`
      : ladderColor(currentLevel)

    const Component = as ?? (asButton ? "button" : "div") as any

    return (
      <ElevationContext.Provider value={currentLevel + 1}>
        <Component
          ref={ref}
          style={{
            backgroundColor,
            ...style,
          }}
          onMouseEnter={(e: any) => {
            setHovered(true)
            props.onMouseEnter?.(e)
          }}
          onMouseLeave={(e: any) => {
            setHovered(false)
            props.onMouseLeave?.(e)
          }}
          onFocus={(e: any) => {
            setFocused(true)
            props.onFocus?.(e)
          }}
          onBlur={(e: any) => {
            setFocused(false)
            props.onBlur?.(e)
          }}
          className={cn(
            "relative transition-all duration-200 text-ink",
            castShadow && currentLevel === 1 && "shadow-[var(--shadow-sm),var(--inner-highlight)]",
            castShadow && currentLevel >= 2 && "shadow-[var(--shadow-md),var(--inner-highlight)]",
            hoverLift && "hover:shadow-[var(--shadow-xl),var(--inner-highlight-strong)] hover:z-10",
            asButton && "cursor-pointer select-none text-left outline-none",
            className
          )}
          {...props}
        >
          {children}
        </Component>
      </ElevationContext.Provider>
    )
  }
)

ElevatedContainer.displayName = "ElevatedContainer"
