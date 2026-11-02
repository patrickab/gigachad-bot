import type { CellTerminal, CellTerminalListener, CellText } from "./cellTerminal"
import { setCellSource, type SharedCellLike, type SharedNotebookLike } from "./notebookSync"

/** A Lumino signal as far as this file listens to one. */
interface SignalLike<Args> {
  connect(slot: (sender: unknown, args: Args) => void): unknown
  disconnect(slot: (sender: unknown, args: Args) => void): unknown
}

/** The `@jupyterlab/cells` cell widget members used here. */
export interface CellWidgetLike {
  readonly node: HTMLElement
  readonly model: { readonly id: string; readonly type: string }
}

/** The `@jupyterlab/notebook` `Notebook` members used here. */
export interface NotebookWidgetLike {
  activeCellIndex: number
  readonly activeCell: CellWidgetLike | null
  readonly activeCellChanged: SignalLike<CellWidgetLike | null>
  readonly stateChanged: SignalLike<{ name: string; newValue: unknown }>
  setMode(mode: "command" | "edit", options?: { focus?: boolean }): void
}

/** The shared notebook: its Yjs document's `update` is the one signal that sees every change of every cell. */
export type CellModelLike = Pick<SharedNotebookLike, "cells" | "transact" | "ydoc">

/** The `language_info` notebook metadata as far as this file reads it. */
export interface LanguageInfoLike {
  name?: string
  file_extension?: string
}

interface CellNeovimDeps {
  notebook: NotebookWidgetLike
  model: CellModelLike
  terminal: CellTerminal
  /** What the notebook says about its language (`python`, `.py`), null when it says nothing. */
  language: () => LanguageInfoLike | null
  runAndAdvance: () => void
  save: () => void
}

/** What `ActiveCellTerminal` shows: the id of the cell Neovim's window is pinned to (null when none), its widget, and how many lines it has. */
export interface CellNeovimSnapshot {
  id: string | null
  cell: CellWidgetLike | null
  lines: number
}

/** The class of the element that holds the terminal while it is in a cell. */
export const TERMINAL_HOST_CLASS = "gc-nvim-host"

/** Room the terminal box adds around its rows: its padding and border. */
const BOX_CHROME_PX = 10
/** Row height until the terminal is on screen and measures its own. */
const FALLBACK_ROW_PX = 17
/** Rows the terminal has besides the cell's lines: the status line, the command line, and one free row so that a new line (Enter) does not scroll the first one away before the terminal has grown. */
const SPARE_ROWS = 3

/** The height in px of the terminal box of a cell with `lines` lines, `rowHeight` the px of one terminal row (0 while unknown). */
export function cellTerminalHeight(lines: number, rowHeight: number): number {
  return Math.ceil((lines + SPARE_ROWS) * (rowHeight > 0 ? rowHeight : FALLBACK_ROW_PX)) + BOX_CHROME_PX
}

const lineCount = (text: string) => text.split("\n").length

/** The extension of the files of the notebook's language: what the kernel reports (`.py`), else `py` for Python and the language name for a notebook that has not run yet. */
export function languageExtension(info: LanguageInfoLike | null): string | null {
  const name = info?.name?.toLowerCase()
  return info?.file_extension?.replace(/^\./, "") || (name === "python" ? "py" : name) || null
}

/** The name `set_cells` gives for Neovim's buffer, only its extension matters: the filetype, and for a Python notebook the `.py` the language server analyzes. The backend reads it once, when it creates the buffer, and names the buffer after the notebook's file. */
const bufferName = (language: LanguageInfoLike | null) => `notebook.${languageExtension(language) ?? "txt"}`

const isCode = (cell: CellWidgetLike | null): cell is CellWidgetLike => cell?.model.type === "code"

const sameCells = (a: readonly CellText[], b: readonly CellText[]) => a.length === b.length && a.every((cell, index) => cell.id === b[index].id && cell.text === b[index].text)

/** A code cell of the notebook: `key` is what Neovim knows it by, `index` its position among all cells (the notebook widget's index too). */
interface CodeCell {
  key: string
  index: number
  cell: SharedCellLike
}

/**
 * Neovim on a notebook, both ways, with `terminal` as the session. Neovim holds all code cells as one buffer, its window shows the active one.
 *
 * - `setCells` hands Neovim the code cells whenever the notebook changes them (anything but Neovim's own edits): `sent` is what Neovim has, an update equal to it is not sent
 * - Neovim knows a cell by its `key`, the cell's id. A notebook file may repeat ids, so a later cell with a taken id gets `<id>#2` and so on: Neovim never gets a repeated id (it would refuse the cells), and everything it reports maps back to the cell by its key
 * - An edit in Neovim goes into its cell (`setCellSource`, outside the undo stack) and into `sent` first, so it is not sent back
 * - The active code cell gets the window (`focusCell`) however it became active: click, keyboard, run and advance, first load. A markdown or raw cell has no Neovim, its own editor shows
 * - Neovim moving its cursor into another cell activates that cell, Shift+Enter runs the cell and advances
 */
export class CellNeovim {
  /** The terminal calls these, `terminal.subscribe` hands them over. */
  readonly listener: CellTerminalListener = {
    edit: (id, text) => { this.edit(id, text) },
    focus: (id) => this.focusFromNeovim(id),
    run: (id, text) => this.run(id, text),
    save: () => this.deps.save(),
  }

  private terminal: CellTerminal
  private session: object | null
  private shown: string | null = null
  private sent: CellText[] | null = null
  private readonly subscribers = new Set<() => void>()
  private snapshot: CellNeovimSnapshot = { id: null, cell: null, lines: 1 }
  private started = false

  constructor(private readonly deps: CellNeovimDeps) {
    this.terminal = deps.terminal
    this.session = deps.terminal.session
  }

  start(): void {
    if (this.started) return
    this.started = true
    this.deps.notebook.activeCellChanged.connect(this.onActiveChanged)
    this.deps.notebook.stateChanged.connect(this.onNotebookState)
    this.deps.model.ydoc.on("update", this.onModelUpdate)
    this.reconcile()
  }

  dispose(): void {
    if (!this.started) return
    this.started = false
    this.deps.notebook.activeCellChanged.disconnect(this.onActiveChanged)
    this.deps.notebook.stateChanged.disconnect(this.onNotebookState)
    this.deps.model.ydoc.off("update", this.onModelUpdate)
    this.sent = null
    this.shown = null
  }

  /** The terminal changed state, or is another session: what the old session held is gone. */
  update(terminal: CellTerminal): void {
    this.terminal = terminal
    if (terminal.session !== this.session) {
      this.session = terminal.session
      this.sent = null
      this.shown = null
    }
    if (this.started) this.reconcile()
  }

  readonly subscribe = (listener: () => void) => {
    this.subscribers.add(listener)
    return () => { this.subscribers.delete(listener) }
  }

  readonly getSnapshot = () => this.snapshot

  /** The notebook's code cells in order with the key Neovim knows each by: the cell's id, `<id>#2`, `<id>#3` for a repeat of an id (and past an id that is taken already). */
  private codeCells(): CodeCell[] {
    const taken = new Set<string>()
    const cells: CodeCell[] = []
    this.deps.model.cells.forEach((cell, index) => {
      if (cell.cell_type !== "code") return
      let key = cell.id
      for (let repeat = 2; taken.has(key); repeat++) key = `${cell.id}#${repeat}`
      taken.add(key)
      cells.push({ key, index, cell })
    })
    return cells
  }

  private codeCell(key: string): CodeCell | undefined {
    return this.codeCells().find((candidate) => candidate.key === key)
  }

  private publish(id: string | null) {
    const cell = id === null ? null : this.deps.notebook.activeCell
    const source = id === null ? undefined : this.codeCell(id)?.cell.getSource()
    const lines = source === undefined ? 1 : lineCount(source)
    if (this.snapshot.id === id && this.snapshot.cell === cell && this.snapshot.lines === lines) return
    this.snapshot = { id, cell, lines }
    for (const subscriber of [...this.subscribers]) subscriber()
  }

  /**
   * Bring Neovim to the notebook's state, whatever changed: the code cells it holds, and the window on the active cell. Idempotent, every trigger calls it.
   */
  private reconcile() {
    const active = this.deps.notebook.activeCell
    if (!this.session) {
      this.shown = null
      this.publish(null)
      return
    }
    if (!this.pushCells()) {
      this.shown = null
      this.publish(null)
      return
    }
    const entry = this.codeCells().find((candidate) => candidate.index === this.deps.notebook.activeCellIndex)
    if (!isCode(active) || !entry) {
      this.shown = null
      this.publish(null)
      return
    }
    const id = entry.key
    if (this.shown !== id) {
      if (!this.terminal.focusCell(id, "first")) {
        this.shown = null
        this.publish(null)
        return
      }
      this.claimKeys()
    }
    this.shown = id
    this.publish(id)
  }

  /** The code cells go to Neovim unless it has them already. False when the session cannot take them. The name only matters to a buffer Neovim creates now: it keeps the one it has when the notebook's language changes. */
  private pushCells(): boolean {
    const cells = this.codeCells().map(({ key, cell }) => ({ id: key, text: cell.getSource() }))
    if (this.sent && sameCells(this.sent, cells)) return true
    if (!this.terminal.setCells(bufferName(this.deps.language()), cells)) return false
    this.sent = cells
    return true
  }

  /** The notebook must not give the cell's editor the keys, the terminal has them. A terminal still in another cell is left alone: focusing it there would make that cell the active one again, it focuses itself once it is in the active cell. */
  private claimKeys() {
    this.deps.notebook.setMode("command", { focus: false })
    queueMicrotask(() => { if (this.deps.notebook.activeCell?.node.querySelector(`.${TERMINAL_HOST_CLASS}`)) this.terminal.focus() })
  }

  private readonly onActiveChanged = () => {
    // A session that died comes back when a code cell becomes active, not on every notebook change
    if (this.terminal.state === "stopped" && isCode(this.deps.notebook.activeCell)) this.terminal.restart()
    this.reconcile()
  }

  /** Edit mode (a click in another cell's editor, running the last cell) focuses the cell's editor behind the terminal. */
  private readonly onNotebookState = (_sender: unknown, change: { name: string; newValue: unknown }) => {
    if (change.name === "mode" && change.newValue === "edit" && this.shown !== null) this.claimKeys()
  }

  private readonly onModelUpdate = () => {
    this.reconcile()
  }

  /** Neovim's text of the cell `key` goes into it, the cell's index comes back. A key that is none of this notebook's code cells is not ours: -1. */
  private edit(key: string, text: string): number {
    const entry = this.codeCell(key)
    if (!entry) return -1
    const held = this.sent?.find((cell) => cell.id === key)
    if (held) held.text = text
    if (entry.cell.getSource() !== text) this.deps.model.transact(() => setCellSource(entry.cell, text), false)
    return entry.index
  }

  private activate(index: number) {
    if (this.deps.notebook.activeCellIndex !== index) this.deps.notebook.activeCellIndex = index
  }

  private run(id: string, text: string) {
    const index = this.edit(id, text)
    if (index < 0) return
    this.activate(index)
    this.deps.runAndAdvance()
  }

  /** Neovim pinned its window to the cell `key` itself: the notebook follows, Neovim is not told. */
  private focusFromNeovim(key: string) {
    const entry = this.codeCell(key)
    if (!entry) return
    this.shown = key
    this.activate(entry.index)
    this.publish(key)
  }
}
