"use client"

import React, { createContext, useContext, useState } from "react"
import { cn } from "@/lib/utils"

// Anchored to role tokens so the elevation ladder inverts between modes:
//   dark mode  : descends from --paper (near-black) to --surface-elevated
//   light mode : ascends from --paper (broken white) to --surface-elevated (near-white)
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
  as?: React.ElementType     // Any HTML tag or component (aside, section, motion.div, etc.)
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

/** A surface whose background steps from `darkColor` toward `brightColor` with each nesting level. */
export const ElevatedContainer = React.forwardRef<any, ElevatedContainerProps>(
  (
    {
      children,
      className,
      as,
      asButton = false,
      castShadow = true,
      hoverLift = false,      // Off by default to keep flat interfaces (like the chat feed) flat
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
    const [focused, setFocused] = useState(false)

    // Cascade: own prop, then parent context, then default
    const finalDark = darkColor ?? parentContext?.darkColor ?? DEFAULT_DARK
    const finalBright = brightColor ?? parentContext?.brightColor ?? DEFAULT_BRIGHT
    const finalNumLevels = numLevels ?? parentContext?.numLevels ?? DEFAULT_NUM_LEVELS
    const currentLevel = parentContext ? parentContext.level : 0

    // Level 0 maps to `darkColor`, the highest level to `brightColor`
    const percentage = finalNumLevels > 1
      ? Math.min(100, Math.max(0, (currentLevel * 100) / (finalNumLevels - 1)))
      : 0

    // Hover or keyboard focus shifts 8% towards brightColor
    const isActive = (hovered || focused) && asButton
    const displayPercentage = isActive
      ? Math.min(100, percentage + 8)
      : percentage

    const backgroundColor = `color-mix(in srgb, ${finalDark}, ${finalBright} ${displayPercentage}%)`

    // Nested children sit one level higher
    const nextContext: ElevationContextType = {
      level: currentLevel + 1,
      numLevels: finalNumLevels,
      darkColor: finalDark,
      brightColor: finalBright,
    }

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
