/**
 * Role-token palette spec, the source of truth for the color system. `app/globals.css`
 * mirrors it in `:root` and `.light`. This TS copy is the fallback for non-DOM surfaces
 * (canvas fills, JPEG export) where `getComputedStyle` is unavailable.
 * Colors are OKLCH with a warm hue (about 85) and tiny chroma to stay monochromatic.
 *
 * To add a theme: add an entry to THEMES, add a matching `.name` block in globals.css
 * with the same custom properties, and toggle the class on <html>.
 */

export interface RoleTokens {
  ink: string
  inkMuted: string
  inkSubtle: string
  inkFaint: string
  paper: string
  surface: string
  surfaceElevated: string
  overlay: string
  danger: string
  backdrop: string
  shadowStrength: number
}

const dark: RoleTokens = {
  ink:             "oklch(95% 0.006 85)",
  inkMuted:        "oklch(72% 0.010 85)",
  inkSubtle:       "oklch(55% 0.010 85)",
  inkFaint:        "oklch(42% 0.008 85)",
  paper:           "oklch(8%  0.006 85)",
  surface:         "oklch(13% 0.008 85)",
  surfaceElevated: "oklch(18% 0.010 85)",
  overlay:         "oklch(6%  0.006 85)",
  danger:          "oklch(68% 0.20 25)",
  backdrop:        "oklch(0% 0 0 / 0.65)",
  shadowStrength:  0.35,
}

const light: RoleTokens = {
  ink:             "oklch(28% 0.012 85)",
  inkMuted:        "oklch(40% 0.012 85)",
  inkSubtle:       "oklch(50% 0.010 85)",
  inkFaint:        "oklch(60% 0.008 85)",
  paper:           "oklch(95%   0.020 85)",
  surface:         "oklch(97.5% 0.012 85)",
  surfaceElevated: "oklch(99%   0.006 85)",
  overlay:         "oklch(96%   0.015 85)",
  danger:          "oklch(52% 0.22 25)",
  backdrop:        "oklch(0% 0 0 / 0.55)",
  shadowStrength:  0.08,
}

const THEMES = { dark, light } as const

export type ThemeName = keyof typeof THEMES
export type ColorToken = Exclude<keyof RoleTokens, "shadowStrength">

const CSS_VAR: Record<ColorToken, string> = {
  ink:             "--ink",
  inkMuted:        "--ink-muted",
  inkSubtle:       "--ink-subtle",
  inkFaint:        "--ink-faint",
  paper:           "--paper",
  surface:         "--surface",
  surfaceElevated: "--surface-elevated",
  overlay:         "--overlay",
  danger:          "--danger",
  backdrop:        "--backdrop",
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
