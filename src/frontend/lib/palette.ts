/**
 * Role-token palette spec.
 *
 * Single source of truth for the app's color system. The CSS in
 * `app/globals.css` mirrors these values inside `:root` and `.light`
 * (and any future theme selectors). The TS object is consumed by
 * non-DOM surfaces (DrawingCanvas fill colors, JPEG export, etc.)
 * as fallback values when `getComputedStyle` is not available.
 *
 * Every color is OKLCH with a warm hue (≈ 85°) and tiny chroma
 * (0.005–0.020), so the UI stays strictly monochromatic.
 *
 * To add a new theme:
 *   1. Add an entry to THEMES below with the same shape as `dark`/`light`.
 *   2. Add a matching `.yourtheme { ... }` block in globals.css
 *      using the same CSS custom-property names.
 *   3. Toggle the class on <html> (see app/layout.tsx and ThemeToggle.tsx).
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

export const dark: RoleTokens = {
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

export const light: RoleTokens = {
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

export const THEMES = { dark, light } as const

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

export function themeFor(name: ThemeName): RoleTokens {
  return THEMES[name]
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
