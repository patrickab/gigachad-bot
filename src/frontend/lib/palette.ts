/** Fallback colors for `readToken` where no CSS custom property resolves (canvas fills, exports). */

interface RoleTokens {
  ink: string
  paper: string
  surfaceElevated: string
}

const dark: RoleTokens = {
  ink:             "oklch(94% 0.010 75)",
  paper:           "oklch(14%   0.008 55)",
  surfaceElevated: "oklch(22.5% 0.010 55)",
}

const light: RoleTokens = {
  ink:             "oklch(28% 0.012 85)",
  paper:           "oklch(95%   0.020 85)",
  surfaceElevated: "oklch(99%   0.006 85)",
}

const THEMES = { dark, light } as const

export type ThemeName = keyof typeof THEMES
type ColorToken = keyof RoleTokens

const CSS_VAR: Record<ColorToken, string> = {
  ink:             "--ink",
  paper:           "--paper",
  surfaceElevated: "--surface-elevated",
}

export function activeThemeName(): ThemeName {
  if (typeof document === "undefined") return "dark"
  return document.documentElement.classList.contains("light") ? "light" : "dark"
}

/**
 * Read a role-token value live from the active CSS custom property,
 * falling back to the matching entry in the palette spec. Use this
 * for non-DOM consumers (canvas, SVG fills, exports) that need an
 * actual color string at render time.
 */
export function readToken(name: ColorToken): string {
  const fallback = THEMES[activeThemeName()][name]
  if (typeof document === "undefined") return fallback
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(CSS_VAR[name])
    .trim()
  return v || fallback
}
