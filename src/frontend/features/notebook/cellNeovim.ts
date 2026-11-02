import type { CellCursor, CellDirection, CellTerminal, CellTerminalListener } from "./cellTerminal"
import { setCellSource, type SharedCellLike, type SharedNotebookLike } from "./notebookSync"

/** A Lumino signal as far as this file listens to one. */
interface SignalLike<Args> {
  connect(slot: (sender: unknown, args: Args) => void): unknown
  disconnect(slot: (sender: unknown, args: Args) => void): unknown
}

/** The `@jupyterlab/cells` cell widget members used here. */
export interface CellWidgetLike {
  readonly node: HTMLElement
  readonly model: {
    readonly id: string
    readonly type: string
    readonly sharedModel: Pick<SharedCellLike, "getSource" | "updateSource"> & { readonly changed: SignalLike<unknown> }
  }
  /** Markdown cells only: true while the rendered text shows instead of the editor. */
  rendered?: boolean
  readonly renderedChanged?: SignalLike<boolean>
}

/** The `@jupyterlab/notebook` `Notebook` members used here. */
export interface NotebookWidgetLike {
  activeCellIndex: number
  readonly activeCell: CellWidgetLike | null
  readonly widgets: readonly CellWidgetLike[]
  readonly activeCellChanged: SignalLike<CellWidgetLike | null>
  readonly stateChanged: SignalLike<{ name: string; newValue: unknown }>
  setMode(mode: "command" | "edit", options?: { focus?: boolean }): void
}

/** The shared notebook with the signal that reports the cell list changing. */
export type CellModelLike = Pick<SharedNotebookLike, "cells" | "transact"> & { readonly changed: SignalLike<unknown> }

/** The `language_info` notebook metadata as far as this file reads it. */
export interface LanguageInfoLike {
  name?: string
  file_extension?: string
}

interface CellNeovimDeps {
  tabId: string
  notebook: NotebookWidgetLike
  model: CellModelLike
  terminal: CellTerminal
  /** What the notebook says about its language (`python`, `.py`), null when it says nothing. */
  language: () => LanguageInfoLike | null
  runAndAdvance: () => void
  save: () => void
}

/** What `ActiveCellTerminal` shows: the cell whose buffer is on screen in Neovim (null when none) and how many lines it has. */
export interface CellNeovimSnapshot {
  key: string | null
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

/** The extension of the files of the notebook's language: what the kernel reports (`.py`), else the language name for a notebook that has not run yet. */
export function languageExtension(info: LanguageInfoLike | null): string | null {
  return info?.file_extension?.replace(/^\./, "") || info?.name?.toLowerCase() || null
}

/** The name a cell buffer gets, only its extension matters (the filetype). */
function filenameOf(cell: CellWidgetLike, language: LanguageInfoLike | null): string {
  if (cell.model.type === "markdown") return "cell.md"
  if (cell.model.type !== "code") return "cell.txt"
  return `cell.${languageExtension(language) ?? "txt"}`
}

const isEditable = (cell: CellWidgetLike) => cell.model.type !== "markdown" || cell.rendered === false

/**
 * Neovim on the active cell of a notebook, both ways, with `terminal` as the session.
 *
 * - The active cell's source is the buffer of Neovim's `key` (`<tab id>#<cell id>`): opened when the cell becomes active, changes made in the notebook go to the buffer (`set`), changes of the buffer into the cell (`setCellSource`, outside the undo stack)
 * - Echo stops by equality: `sent` holds the text last exchanged per cell, an edit equal to it is not sent back
 * - Shift+Enter runs the cell and advances, `j` and `k` past the last or first line activate the next or previous cell with the cursor on its first or last line
 * - A cell removed from the notebook has its buffer wiped. Markdown cells show Neovim only while their editor shows
 */
export class CellNeovim {
  /** The terminal calls these, `terminal.subscribe` hands them over. */
  readonly listener: CellTerminalListener = {
    edit: (key, text) => this.edit(key, text),
    run: (key, text) => this.run(key, text),
    exit: (key, direction, text) => this.exit(key, direction, text),
    save: (key, text) => {
      this.edit(key, text)
      this.deps.save()
    },
  }

  private terminal: CellTerminal
  private session: object | null
  private active: CellWidgetLike | null = null
  private shown: string | null = null
  private cursor: CellCursor = "first"
  private readonly opened = new Set<string>()
  private readonly sent = new Map<string, string>()
  private readonly subscribers = new Set<() => void>()
  private snapshot: CellNeovimSnapshot = { key: null, cell: null, lines: 1 }
  private detachActive: (() => void) | null = null
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
    this.deps.model.changed.connect(this.onModelChanged)
    this.attach(this.deps.notebook.activeCell)
    this.show(true)
  }

  dispose(): void {
    if (!this.started) return
    this.started = false
    this.deps.notebook.activeCellChanged.disconnect(this.onActiveChanged)
    this.deps.notebook.stateChanged.disconnect(this.onNotebookState)
    this.deps.model.changed.disconnect(this.onModelChanged)
    this.detachActive?.()
    this.detachActive = null
    for (const key of this.opened) this.terminal.close(key)
    this.opened.clear()
    this.sent.clear()
  }

  /** The terminal changed state, or is another session: what the old session had open is gone. */
  update(terminal: CellTerminal): void {
    this.terminal = terminal
    if (terminal.session !== this.session) {
      this.session = terminal.session
      this.opened.clear()
      this.sent.clear()
      this.shown = null
    }
    if (this.started) this.show(false)
  }

  readonly subscribe = (listener: () => void) => {
    this.subscribers.add(listener)
    return () => { this.subscribers.delete(listener) }
  }

  readonly getSnapshot = () => this.snapshot

  private publish(key: string | null) {
    const cell = key === null ? null : this.active
    const lines = cell ? lineCount(cell.model.sharedModel.getSource()) : 1
    if (this.snapshot.key === key && this.snapshot.cell === cell && this.snapshot.lines === lines) return
    this.snapshot = { key, cell, lines }
    for (const subscriber of [...this.subscribers]) subscriber()
  }

  private keyOf(cell: CellWidgetLike) { return `${this.deps.tabId}#${cell.model.id}` }

  private find(key: string): { cell: CellWidgetLike; index: number } | null {
    const index = this.deps.notebook.widgets.findIndex((cell) => this.keyOf(cell) === key)
    return index < 0 ? null : { cell: this.deps.notebook.widgets[index], index }
  }

  /** Follow `cell`, the active one: its source changes go to Neovim, a markdown cell's editor showing or not changes whether Neovim does. */
  private attach(cell: CellWidgetLike | null) {
    this.detachActive?.()
    this.detachActive = null
    this.active = cell
    if (!cell) return
    const onSource = () => this.onSource()
    const onRendered = () => this.show(false)
    cell.model.sharedModel.changed.connect(onSource)
    cell.renderedChanged?.connect(onRendered)
    this.detachActive = () => {
      cell.model.sharedModel.changed.disconnect(onSource)
      cell.renderedChanged?.disconnect(onRendered)
    }
  }

  private readonly onActiveChanged = (_sender: unknown, cell: CellWidgetLike | null) => {
    if (cell === this.active && cell && this.shown === this.keyOf(cell)) return
    this.attach(cell)
    if (this.terminal.state === "stopped") this.terminal.restart()
    this.show(true)
  }

  /** Put the active cell in Neovim: `force` reopens it even when it is on screen already. Hides Neovim when the cell has no editor or the session is not up. */
  private show(force: boolean) {
    const cell = this.active
    if (!cell || !isEditable(cell) || !this.session) {
      this.shown = null
      this.publish(null)
      return
    }
    const key = this.keyOf(cell)
    if (force || this.shown !== key) {
      const text = cell.model.sharedModel.getSource()
      if (!this.terminal.open(key, filenameOf(cell, this.deps.language()), text, this.cursor)) {
        this.shown = null
        this.publish(null)
        return
      }
      this.opened.add(key)
      this.sent.set(key, text)
      this.cursor = "first"
      this.claimKeys()
    }
    this.shown = key
    this.publish(key)
  }

  /** The notebook must not give the cell's editor the keys, the terminal has them. A terminal still in another cell is left alone: focusing it there would make that cell the active one again, it focuses itself once it is in the active cell. */
  private claimKeys() {
    this.deps.notebook.setMode("command", { focus: false })
    queueMicrotask(() => { if (this.active?.node.querySelector(`.${TERMINAL_HOST_CLASS}`)) this.terminal.focus() })
  }

  /** Edit mode (a click in another cell's editor, running the last cell) focuses the cell's editor behind the terminal. */
  private readonly onNotebookState = (_sender: unknown, change: { name: string; newValue: unknown }) => {
    if (change.name === "mode" && change.newValue === "edit" && this.shown !== null) this.claimKeys()
  }

  private onSource() {
    const cell = this.active
    if (!cell) return
    const key = this.keyOf(cell)
    const text = cell.model.sharedModel.getSource()
    if (this.shown === key && this.sent.get(key) !== text && this.terminal.set(key, text)) this.sent.set(key, text)
    this.publish(this.shown)
  }

  private readonly onModelChanged = () => {
    const ids = new Set(this.deps.model.cells.map((cell) => cell.id))
    for (const key of [...this.opened]) {
      if (ids.has(key.slice(key.indexOf("#") + 1))) continue
      this.terminal.close(key)
      this.opened.delete(key)
      this.sent.delete(key)
    }
  }

  /** Neovim's text of `key` goes into its cell. A buffer that is none of this notebook's is not ours. */
  private edit(key: string, text: string): { cell: CellWidgetLike; index: number } | null {
    const found = this.find(key)
    if (!found) return null
    this.sent.set(key, text)
    const shared = found.cell.model.sharedModel
    if (shared.getSource() !== text) this.deps.model.transact(() => setCellSource(shared, text), false)
    return found
  }

  private run(key: string, text: string) {
    const found = this.edit(key, text)
    if (!found) return
    if (this.deps.notebook.activeCellIndex !== found.index) this.deps.notebook.activeCellIndex = found.index
    this.deps.runAndAdvance()
  }

  private exit(key: string, direction: CellDirection, text: string) {
    const found = this.edit(key, text)
    if (!found) return
    const target = this.deps.notebook.widgets[found.index + (direction === "next" ? 1 : -1)]
    if (!target) return
    this.cursor = direction === "next" ? "first" : "last"
    // Leaving a markdown cell renders it, as running it does. Entering one edits it, a rendered cell has no editor to enter
    if (found.cell.model.type === "markdown") found.cell.rendered = true
    if (target.rendered) target.rendered = false
    this.deps.notebook.activeCellIndex = found.index + (direction === "next" ? 1 : -1)
  }
}
