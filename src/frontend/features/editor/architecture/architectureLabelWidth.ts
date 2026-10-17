/** The narrowest card: resize floor and minimum drawn size; a card's labels raise it (`nodeMinWidth`). */
export const NODE_MIN_WIDTH = 80
// Horizontal room the title bar needs beyond its heading and kind label: its
// padding, the gap between the two, and a little slack for the caret while editing.
const TITLE_WIDTH_SLACK = 34

// One hidden title bar measures every card's heading and kind label in the real
// (hand-drawn) font: the surface's font, set here too since the spans live on <body>.
// Cached per text; the cache is dropped once that font has loaded, since earlier
// measurements used the fallback.
const labelWidths = new Map<string, number>()
let labelMeasure: { title: HTMLSpanElement, kind: HTMLSpanElement } | null = null
/** Settles when fonts load, then clears widths measured in the fallback font. Callers then re-measure. */
export const labelFontsReady: Promise<void> = typeof document === "undefined" || !document.fonts
  ? Promise.resolve()
  : document.fonts.ready.then(() => { labelWidths.clear() })
/** The narrowest a card can be and still show its whole heading and kind label. */
export function nodeMinWidth(title: string, kind: string): number {
  if (typeof document === "undefined") return NODE_MIN_WIDTH
  const key = `${title}\u0000${kind}`
  let width = labelWidths.get(key)
  if (width === undefined) {
    if (!labelMeasure) {
      const span = (className: string) => {
        const element = document.createElement("span")
        element.className = className
        element.setAttribute("aria-hidden", "true")
        element.style.cssText = "position:absolute;left:-9999px;top:0;visibility:hidden;white-space:nowrap;font-family:var(--font-handwriting),cursive"
        document.body.appendChild(element)
        return element
      }
      labelMeasure = { title: span("architecture-diagram-title"), kind: span("architecture-diagram-node-kind") }
    }
    labelMeasure.title.textContent = title
    labelMeasure.kind.textContent = kind
    width = Math.max(NODE_MIN_WIDTH, Math.ceil(labelMeasure.title.offsetWidth + labelMeasure.kind.offsetWidth) + TITLE_WIDTH_SLACK)
    labelWidths.set(key, width)
  }
  return width
}
