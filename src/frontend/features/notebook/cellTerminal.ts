import type { ReactNode } from "react"

/** Where Neovim's cursor lands when its window is pinned to a cell: its first or last line, wherever it is when the cell is the one it was in, or a line of the cell (0 is the first, a click on its render). */
export type CellCursor = "first" | "last" | "keep" | { line: number }

/** How the terminal draws text: what a render of a cell needs to take the same room. `cellWidth` and `cellHeight` are the px of one terminal cell, `columns` how many cells a row has (Neovim cuts its lines there). */
export interface CellTextMetrics {
  fontFamily: string
  fontSize: number
  cellWidth: number
  cellHeight: number
  columns: number
}

/** The cells Neovim edits: code cells in one buffer (one script for the language server), markdown cells in a second one. */
export type CellKind = "code" | "markdown"

/** Where Neovim's cursor left a cell past its edge: `cell_exit`, the neighbor cell in the notebook is of the other kind. */
export type CellExitDirection = "up" | "down"

/** One cell of the notebook as Neovim's buffer of its `kind` holds it. */
export interface CellText {
  id: string
  text: string
  kind: CellKind
}

/** What Neovim reports about the notebook's buffers, in the order it happened. `id` is the cell's id in the notebook. */
export interface CellTerminalListener {
  /** The cell `id` holds `text` after an edit. */
  edit(id: string, text: string): void
  /** The cursor moved into the cell `id` (`j`/`k`, `G`, a search, an LSP jump), always one of the same kind: the notebook shows it as the active one. */
  focus(id: string): void
  /** Shift+Enter (`advance`) or Ctrl+Enter in the cell `id`. */
  run(id: string, text: string, advance: boolean): void
  /** Esc in normal mode in the markdown cell `id`: done editing it. */
  leave(id: string): void
  /** The cursor left the cell `id` past its edge (`j`/`k`) toward a neighbor cell of the other kind. */
  exit(id: string, direction: CellExitDirection): void
  /** `:w`: the notebook is saved, there is no file. */
  save(): void
  /** How Neovim draws the code cell `id` (`cell_render`, unchecked: `readCellRendering` takes what is valid of it). */
  render(id: string, rendering: unknown): void
  /** The text of the cell `id`, the one Neovim's window is pinned to, takes `rows` screen rows there (wrapped lines, folds, virtual lines). */
  height(id: string, rows: number): void
}

/**
 * The Neovim terminal of a notebook, handed in by whoever owns the Neovim session (`features/editor`: the notebook feature does not import it).
 * Neovim holds the code cells of the notebook as one buffer (one script for the LSP), the markdown cells as a second one, and its window shows the focused cell only.
 * Every method returns false while the session cannot take the frame.
 */
export interface CellTerminal {
  /** `starting` until the session runs, `stopped` once it died (`restart` brings it back). */
  state: "starting" | "running" | "stopped"
  /** Identifies the running session, another object after a restart: what Neovim holds belongs to one session. Null unless `running`. */
  session: object | null
  /** The code and markdown cells in notebook order, the whole truth: Neovim replaces what differs without echoing it back. `filename` only picks the filetype of the code buffer. */
  setCells(filename: string, cells: readonly CellText[]): boolean
  /** The window shows the cell `id`, in the buffer of its kind. */
  focusCell(id: string, cursor: CellCursor): boolean
  /** Keys go to the terminal. */
  focus(): void
  restart(): void
  /** How the terminal draws text, null until it was on screen. */
  metrics(): CellTextMetrics | null
  /** The one listener of the session's notifications, replaced by the next call. The function stays the same while the terminal changes. */
  subscribe(listener: CellTerminalListener): () => void
  /** The terminal as a box that fills its parent. */
  render(className: string): ReactNode
}
