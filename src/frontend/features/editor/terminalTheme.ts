import { readToken } from "@/lib/palette"

export const TERMINAL_FONT_SIZE = 13
// Local Nerd Fonts first, so icons from the user's config render as in a terminal. "Mono" keeps icons one cell wide.
export const NERD_FONTS = "'JetBrainsMono Nerd Font Mono', 'Symbols Nerd Font Mono'"

type Rgb = [red: number, green: number, blue: number]

/** `color` as RGB, null when fully transparent. Canvas resolves any CSS color, xterm only takes opaque ones. */
function toRgb(color: string): Rgb | null {
  const context = window.document.createElement("canvas").getContext("2d", { willReadFrequently: true })
  if (!context) return null
  context.fillStyle = color
  context.fillRect(0, 0, 1, 1)
  const [red, green, blue, alpha] = context.getImageData(0, 0, 1, 1).data
  return alpha === 0 ? null : [red, green, blue]
}

const rgbCss = ([red, green, blue]: Rgb) => `rgb(${red}, ${green}, ${blue})`

/** Share of ink in the cursor block over the background: clearly visible, softer than full ink. */
const CURSOR_INK_SHARE = 0.3

const mix = (base: Rgb, over: Rgb, share: number): Rgb => base.map((channel, index) => Math.round(channel + (over[index] - channel) * share)) as Rgb

/** Terminal colors of the app theme: the background `element` sits on, and the ink on it. `isLight` is what Neovim gets as 'background'. */
export function terminalTheme(element: HTMLElement) {
  let background: Rgb | null = null
  for (let node: HTMLElement | null = element; node && !background; node = node.parentElement) background = toRgb(getComputedStyle(node).backgroundColor)
  background ??= toRgb(readToken("paper")) ?? [0, 0, 0]
  const ink = toRgb(readToken("ink")) ?? [255, 255, 255]
  const [red, green, blue] = background
  return {
    // The character under the cursor block keeps the ink
    theme: { background: rgbCss(background), foreground: rgbCss(ink), cursor: rgbCss(mix(background, ink, CURSOR_INK_SHARE)), cursorAccent: rgbCss(ink) },
    isLight: 0.299 * red + 0.587 * green + 0.114 * blue > 128,
  }
}
