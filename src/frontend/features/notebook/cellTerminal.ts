import type { ReactNode } from "react"

/** Where the cursor lands in a cell shown in the terminal. */
export type CellCursor = "first" | "last"

/** `next` leaves through the last line, `prev` through the first. */
export type CellDirection = "next" | "prev"

/** What Neovim reports about the cell buffers, in the order it happened. `key` names the cell as in `CellTerminal.open`. */
export interface CellTerminalListener {
  /** The buffer of `key` holds `text` after an edit. */
  edit(key: string, text: string): void
  /** Shift+Enter in the buffer of `key`. */
  run(key: string, text: string): void
  /** `j`/`k` or the arrows left the buffer of `key` through its last or first line. */
  exit(key: string, direction: CellDirection, text: string): void
  /** `:w` in the buffer of `key`: the notebook is saved, there is no file. */
  save(key: string, text: string): void
}

/**
 * A Neovim terminal that edits one notebook cell at a time, handed in by whoever owns the Neovim session (`features/editor`: the notebook feature does not import it).
 * Every method returns false while the session cannot take the frame.
 */
export interface CellTerminal {
  /** `starting` until the session runs, `stopped` once it died (`restart` brings it back). */
  state: "starting" | "running" | "stopped"
  /** Identifies the running session, another object after a restart: what is open in Neovim belongs to one session. Null unless `running`. */
  session: object | null
  /** Show the buffer of cell `key` holding exactly `text`. `filename` only picks the filetype. */
  open(key: string, filename: string, text: string, cursor: CellCursor): boolean
  /** The cell `key` changed elsewhere: its buffer takes `text` without echoing it back. */
  set(key: string, text: string): boolean
  /** The cell `key` is gone: wipe its buffer. */
  close(key: string): boolean
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
