"use client"

import { useEffect, useState } from "react"
import { useThemeName } from "@/hooks/useThemeName"
import { afterColorReveal } from "@/lib/themeTransition"
import type { ThemeName } from "@/lib/palette"

/** Theme to send an embedded frame: flips only once the color switch's reveal runs (`afterColorReveal`). */
export function useFrameThemeName(): ThemeName {
  const theme = useThemeName()
  const [frameTheme, setFrameTheme] = useState(theme)
  useEffect(() => afterColorReveal(() => setFrameTheme(theme)), [theme])
  return frameTheme
}
