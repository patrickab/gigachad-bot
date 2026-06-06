"use client"

import React, { createContext, useContext, useState } from "react"
import { cn } from "@/lib/utils"

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
 * Defaults to 3 levels spanning from var(--color-zinc-950) to var(--color-zinc-900)
 */
export function ElevationProvider({
  children,
  darkColor = "var(--color-zinc-950)",
  brightColor = "var(--color-zinc-900)",
  numLevels = 3,
  startLevel = 0,
}: ElevationProviderProps) {
  return (
    <ElevationContext.Provider value={{ level: startLevel, numLevels, darkColor, brightColor }}>
      {children}
    </ElevationContext.Provider>
  )
}

export interface ElevatedContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  children?: React.ReactNode
  asButton?: boolean
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

    // Resolve Context properties
    const finalDark = darkColor ?? parentContext?.darkColor ?? "var(--color-zinc-950)"
    const finalBright = brightColor ?? parentContext?.brightColor ?? "var(--color-zinc-900)"
    const finalNumLevels = numLevels ?? parentContext?.numLevels ?? 3
    const currentLevel = parentContext ? parentContext.level : 0

    // Compute dynamic linear interpolation
    const percentage = finalNumLevels > 1
      ? Math.min(100, Math.max(0, (currentLevel * 100) / (finalNumLevels - 1)))
      : 0

    const displayPercentage = hovered && asButton
      ? Math.min(100, percentage + 8)
      : percentage

    const backgroundColor = `color-mix(in srgb, ${finalDark}, ${finalBright} ${displayPercentage}%)`

    // Increment depth level for any nested elements
    const nextContext: ElevationContextType = {
      level: currentLevel + 1,
      numLevels: finalNumLevels,
      darkColor: finalDark,
      brightColor: finalBright,
    }

    const Component = (asButton ? "button" : "div") as any

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
          className={cn(
            "relative transition-all duration-200",
            castShadow && currentLevel === 1 && "shadow-[0_0_8px_rgba(0,0,0,0.25)]",
            castShadow && currentLevel >= 2 && "shadow-[0_0_12px_rgba(0,0,0,0.35)]",
            hoverLift && "hover:shadow-[0_0_18px_rgba(0,0,0,0.45)] hover:z-10",
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
