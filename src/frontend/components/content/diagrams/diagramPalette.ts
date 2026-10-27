// Mirrors the globals.css role tokens (sRGB of the oklch palette). Hardcoded because mermaid
// can't parse oklch() and the browser preserves it through getComputedStyle/canvas. If the
// palette warmth is retuned in globals.css, regenerate these to match.
export const DIAGRAM_DARK = {
  ink: "rgb(239, 234, 228)", inkMuted: "rgb(179, 173, 165)", inkSubtle: "rgb(130, 124, 118)",
  inkFaint: "rgb(98, 92, 88)", paper: "rgb(12, 8, 6)", surface: "rgb(21, 17, 14)",
  surfaceElevated: "rgb(32, 27, 23)",
}
export const DIAGRAM_LIGHT = {
  ink: "rgb(80, 77, 70)", inkMuted: "rgb(108, 104, 97)", inkSubtle: "rgb(137, 134, 127)",
  inkFaint: "rgb(167, 164, 159)", paper: "rgb(245, 238, 224)", surface: "rgb(250, 246, 238)",
  surfaceElevated: "rgb(254, 252, 247)",
}
