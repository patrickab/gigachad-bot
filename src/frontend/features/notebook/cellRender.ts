import type { CSSProperties } from "react"
import { isRecord } from "@/lib/utils"

/** A style of a render: colors as `#rrggbb`, `underline` straight (`line`) or curly (`curl`). */
export interface CellStyle {
  fg?: string
  bg?: string
  sp?: string
  bold?: true
  italic?: true
  strike?: true
  underline?: "line" | "curl"
}

/** A piece of a line: its text and the index of its style in `CellRendering.styles`. */
export type CellRun = [text: string, style: number]

/**
 * How Neovim draws a code cell (`cell_render` from the backend, see `lib/neovim_render.py`): its lines as runs of text, the number column and signs leading
 * each line and the diagnostics following it. `base` is the color of text without a style when Neovim's colorscheme sets one. `rows` holds the cell line
 * (0 is the first) every drawn line shows: a line that a language server adds above a line (`virt_lines`) belongs to the line below it. `height` is the
 * screen rows the cell takes in Neovim's window, `wrap` (null when the window does not wrap lines) the column a wrapped line goes on at.
 */
export interface CellRendering {
  styles: CellStyle[]
  lines: CellRun[][]
  base: Pick<CellStyle, "fg" | "bg">
  rows: number[]
  height: number
  wrap: number | null
}

const COLOR = /^#[0-9a-f]{6}$/i
const FLAGS = ["bold", "italic", "strike"] as const

function readStyle(value: unknown): CellStyle {
  const style: CellStyle = {}
  if (!isRecord(value)) return style
  for (const name of ["fg", "bg", "sp"] as const) {
    const color = value[name]
    if (typeof color === "string" && COLOR.test(color)) style[name] = color
  }
  for (const flag of FLAGS) if (value[flag] === true) style[flag] = true
  if (value.underline === "line" || value.underline === "curl") style.underline = value.underline
  return style
}

function readRun(value: unknown, styles: number): CellRun[] {
  if (!Array.isArray(value) || value.length !== 2) return []
  const [text, style] = value as unknown[]
  return typeof text === "string" && typeof style === "number" && Number.isInteger(style) && style >= 0 && style < styles ? [[text, style]] : []
}

/**
 * The render in a `cell_render` frame, null when it is none (a frame without lines takes a render back). A frame comes from text of files and of language
 * servers: only what fits the shape is kept (colors that are `#rrggbb`, flags that are `true`, runs whose style exists), so nothing but plain text and checked
 * colors ever reaches the page.
 */
export function readCellRendering(value: unknown): CellRendering | null {
  if (!isRecord(value) || !Array.isArray(value.styles) || !Array.isArray(value.lines)) return null
  const styles = value.styles.map(readStyle)
  const lines = value.lines.map((line) => (Array.isArray(line) ? line.flatMap((run) => readRun(run, styles.length)) : []))
  const { fg, bg } = readStyle(value.base)
  const given = Array.isArray(value.rows) ? value.rows : []
  const rows = lines.map((_line, index) => {
    const row: unknown = given[index]
    return typeof row === "number" && Number.isInteger(row) && row >= 0 ? row : index
  })
  const height = typeof value.height === "number" && Number.isInteger(value.height) && value.height > 0 ? value.height : lines.length
  const wrap = typeof value.wrap === "number" && Number.isInteger(value.wrap) && value.wrap >= 0 ? value.wrap : null
  return { styles, lines, base: { ...(fg && { fg }), ...(bg && { bg }) }, rows, height, wrap }
}

/** The CSS of a run in `style`, drawn as xterm draws Neovim's terminal next to it: every underline straight and in the text's color (the terminal gets no curly or colored underlines). */
export function runStyle(style: CellStyle | undefined): CSSProperties | undefined {
  if (!style) return undefined
  const decorations = [style.underline && "underline", style.strike && "line-through"].filter(Boolean)
  const css: CSSProperties = {}
  if (style.fg) css.color = style.fg
  if (style.bg) css.backgroundColor = style.bg
  if (style.bold) css.fontWeight = "bold"
  if (style.italic) css.fontStyle = "italic"
  if (decorations.length) css.textDecorationLine = decorations.join(" ")
  return css
}
