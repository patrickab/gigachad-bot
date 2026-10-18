/**
 * Role-token palette spec, the source of truth for the color system. `app/globals.css`
 * mirrors it in `:root` and `.light`. This TS copy is the fallback for non-DOM surfaces
 * (canvas fills, JPEG export) where `getComputedStyle` is unavailable. The ompweb fork's
 * `custom/theme/palette.css` copies the `globals.css` values for the embedded omp tabs.
 * Colors are OKLCH with a warm hue (dark about 55-75, light about 85) and tiny chroma.
 *
 * To add a theme: add an entry to THEMES, add a matching `.name` block in globals.css
 * with the same custom properties, and toggle the class on <html>.
 */

interface RoleTokens {
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
}

const dark: RoleTokens = {
  ink:             "oklch(94% 0.010 75)",
  inkMuted:        "oklch(75% 0.013 70)",
  inkSubtle:       "oklch(59% 0.012 65)",
  inkFaint:        "oklch(48% 0.010 60)",
  paper:           "oklch(14%   0.008 55)",
  surface:         "oklch(18%   0.009 55)",
  surfaceElevated: "oklch(22.5% 0.010 55)",
  overlay:         "oklch(11.5% 0.007 55)",
  danger:          "oklch(68% 0.20 25)",
  backdrop:        "oklch(0% 0 0 / 0.65)",
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
}

const THEMES = { dark, light } as const

export type ThemeName = keyof typeof THEMES
type ColorToken = keyof RoleTokens

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
