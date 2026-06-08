"use client"

import React, { createContext, useContext, useState } from "react"
import { cn } from "@/lib/utils"

// Global/local default styling configurations for consistent theme fallbacks
// Anchored to role tokens so the elevation ladder inverts cleanly between modes:
//   dark mode  : descends from --paper (warm-tinted near-black) to --surface-elevated
//   light mode : ascends  from --paper (yellowish broken white)   to --surface-elevated (near-white)
const DEFAULT_DARK = "var(--paper)"
const DEFAULT_BRIGHT = "var(--surface-elevated)"
const DEFAULT_NUM_LEVELS = 3

interface ElevationContextType {
  level: number
  numLevels: number
  darkColor: string
  brightColor: string
}

const ElevationContext = createContext<ElevationContextType | null>(null)

export interface ElevationProviderProps {
  children: React.ReactNode
  darkColor?: string
  brightColor?: string
  numLevels?: number
  startLevel?: number
}

/**
 * Configure global/local base colors and total nesting depth.
 * Defaults to 3 levels spanning from var(--paper) to var(--surface-elevated)
 */
export function ElevationProvider({
  children,
  darkColor = DEFAULT_DARK,
  brightColor = DEFAULT_BRIGHT,
  numLevels = DEFAULT_NUM_LEVELS,
  startLevel = 0,
}: ElevationProviderProps) {
  return (
    <ElevationContext.Provider value={{ level: startLevel, numLevels, darkColor, brightColor }}>
      {children}
    </ElevationContext.Provider>
  )
}

export interface ElevatedContainerProps extends React.HTMLAttributes<HTMLElement> {
  children?: React.ReactNode
  as?: React.ElementType     // Highly extensible: lets the developer render any HTML tag or custom React component (aside, section, motion.div, etc.)
  asButton?: boolean         // For backwards compatibility
  castShadow?: boolean       // Whether to show a default flat shadow
  hoverLift?: boolean        // Opt-in for dynamic hover lifting & downward shadow-casting (for cards)
  darkColor?: string         // Local stack override
  brightColor?: string       // Local stack override
  numLevels?: number         // Local stack override
  disabled?: boolean
  type?: "button" | "submit" | "reset"
  onContextMenu?: React.MouseEventHandler<any>
}

export const ElevatedContainer = React.forwardRef<any, ElevatedContainerProps>(
  (
    {
      children,
      className,
      as,
      asButton = false,
      castShadow = true,
      hoverLift = false,      // Disabled by default to protect flat interfaces (like the chat feed)
      darkColor,
      brightColor,
      numLevels,
      style,
      ...props
    },
    ref
  ) => {
    const parentContext = useContext(ElevationContext)
    const [hovered, setHovered] = useState(false)
    const [focused, setFocused] = useState(false) // Keyboard focus tracking for premium accessibility

    // Resolve Context properties with clean, predictable cascade fallbacks
    const finalDark = darkColor ?? parentContext?.darkColor ?? DEFAULT_DARK
    const finalBright = brightColor ?? parentContext?.brightColor ?? DEFAULT_BRIGHT
    const finalNumLevels = numLevels ?? parentContext?.numLevels ?? DEFAULT_NUM_LEVELS
    const currentLevel = parentContext ? parentContext.level : 0

    // Compute dynamic linear interpolation percentage:
    // This scales the level (e.g. 0, 1, 2) smoothly to a percentage (e.g. 0%, 50%, 100%)
    // spanning between the `darkColor` base (Level 0) and the `brightColor` ceiling (highest Level).
    const percentage = finalNumLevels > 1
      ? Math.min(100, Math.max(0, (currentLevel * 100) / (finalNumLevels - 1)))
      : 0

    // Active triggers (mouse hover or keyboard focus) shift the percentage 8% further towards brightColor for an responsive spotlight feel
    const isActive = (hovered || focused) && asButton
    const displayPercentage = isActive
      ? Math.min(100, percentage + 8)
      : percentage

    const backgroundColor = `color-mix(in srgb, ${finalDark}, ${finalBright} ${displayPercentage}%)`

    // Increment depth level for any nested child elements to enable automatic visual hierarchy
    const nextContext: ElevationContextType = {
      level: currentLevel + 1,
      numLevels: finalNumLevels,
      darkColor: finalDark,
      brightColor: finalBright,
    }

    // Extensible Component tag resolution
    const Component = as ?? (asButton ? "button" : "div") as any

    return (
      <ElevationContext.Provider value={nextContext}>
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
            castShadow && currentLevel === 1 && "shadow-[var(--shadow-sm)]",
            castShadow && currentLevel >= 2 && "shadow-[var(--shadow-md)]",
            hoverLift && "hover:shadow-[var(--shadow-xl)] hover:z-10",
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
