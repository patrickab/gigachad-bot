import type { ReactNode } from "react"

/** Where Neovim's cursor lands when its window is pinned to a cell: its first or last line, or wherever it is when the cell is the one it was in. */
export type CellCursor = "first" | "last" | "keep"

/** One code cell of the notebook as Neovim's buffer holds it. */
export interface CellText {
  id: string
  text: string
}

/** What Neovim reports about the notebook buffer, in the order it happened. `id` is the cell's id in the notebook. */
export interface CellTerminalListener {
  /** The cell `id` holds `text` after an edit. */
  edit(id: string, text: string): void
  /** The cursor moved into the cell `id` (`j`/`k`, `G`, a search, an LSP jump): the notebook shows it as the active one. */
  focus(id: string): void
  /** Shift+Enter in the cell `id`. */
  run(id: string, text: string): void
  /** `:w`: the notebook is saved, there is no file. */
  save(): void
}

/**
 * The Neovim terminal of a notebook, handed in by whoever owns the Neovim session (`features/editor`: the notebook feature does not import it).
 * Neovim holds the code cells of the notebook as one buffer (one script for the LSP) and its window shows the focused cell only.
 * Every method returns false while the session cannot take the frame.
 */
export interface CellTerminal {
  /** `starting` until the session runs, `stopped` once it died (`restart` brings it back). */
  state: "starting" | "running" | "stopped"
  /** Identifies the running session, another object after a restart: what Neovim holds belongs to one session. Null unless `running`. */
  session: object | null
  /** The code cells in notebook order, the whole truth: Neovim replaces what differs without echoing it back. `filename` only picks the filetype. */
  setCells(filename: string, cells: readonly CellText[]): boolean
  /** The window shows the cell `id`. */
  focusCell(id: string, cursor: CellCursor): boolean
  /** Keys go to the terminal. */
  focus(): void
  restart(): void
  /** Height of one terminal row in px, 0 until the terminal is on screen. */
  cellHeight(): number
  /** The one listener of the session's notifications, replaced by the next call. The function stays the same while the terminal changes. */
  subscribe(listener: CellTerminalListener): () => void
  /** The terminal as a box that fills its parent. */
  render(className: string): ReactNode
}
