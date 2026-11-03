import { EDITOR_PAGE } from "@/features/workbench/pages"
import { cn } from "@/lib/utils"
import { readCellRendering, type CellRendering } from "./cellRender"
import type { CellCursor, CellExitDirection, CellKind, CellTerminal, CellTerminalListener, CellText } from "./cellTerminal"
import { setCellSource, type SharedCellLike, type SharedNotebookLike } from "./notebookSync"

/** A Lumino signal as far as this file listens to one. */
interface SignalLike<Args> {
  connect(slot: (sender: unknown, args: Args) => void): unknown
  disconnect(slot: (sender: unknown, args: Args) => void): unknown
}

/** The `@jupyterlab/cells` cell widget members used here. A markdown cell (`MarkdownCell`) also shows its text `rendered` or its editor. */
export interface CellWidgetLike {
  readonly node: HTMLElement
  readonly model: { readonly type: string }
  rendered?: boolean
  readonly renderedChanged?: SignalLike<boolean>
}

/** The `@jupyterlab/notebook` `Notebook` members used here. */
export interface NotebookWidgetLike {
  activeCellIndex: number
  readonly activeCell: CellWidgetLike | null
  /** The cell widgets in notebook order. */
  readonly widgets: readonly CellWidgetLike[]
  /** The notebook's scrolling element: it is empty (no size) while a hidden tab holds it. */
  readonly node: HTMLElement
  readonly activeCellChanged: SignalLike<CellWidgetLike | null>
  readonly stateChanged: SignalLike<{ name: string; newValue: unknown }>
  setMode(mode: "command" | "edit", options?: { focus?: boolean }): void
  /** Scrolls the windowed notebook to the cell at `index`, `auto` moving the least. Unlike `scrollToCell` it leaves the selection and the focus alone. */
  scrollToItem(index: number, align?: "auto" | "smart" | "start" | "center" | "end"): Promise<void>
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
  /** Runs the active cell, which stays the active one. A markdown cell is rendered. */
  run: () => void
  /** Runs the active cell and moves to the next, which is a new cell at the end of the notebook. */
  runAndAdvance: () => void
  save: () => void
}

/** What `ActiveCellTerminal` shows: the id of the cell Neovim's window is pinned to (null when none), its widget, and how many screen rows its text takes in Neovim. */
export interface CellNeovimSnapshot {
  id: string | null
  cell: CellWidgetLike | null
  rows: number
  /** False when the terminal entered its cell by a click on the cell's render: the cell is on screen where it was clicked and nothing scrolls. */
  reveal?: boolean
}

/** A code cell as `CellSnapshots` shows it: the key Neovim knows it by and its widget. */
export interface CodeCellWidget {
  key: string
  cell: CellWidgetLike
}

/** The class of the element that holds the terminal while it is in a cell. */
export const TERMINAL_HOST_CLASS = "gc-nvim-host"
/**
 * The page the terminal draws on in a cell, which a cell's render (`CellSnapshots`) copies so that both look and measure the same: the Neovim panel of the
 * markdown editor's split view.
 */
export const TERMINAL_BOX_CLASS = cn("h-full", EDITOR_PAGE)

/** Room the terminal box adds around its rows: its top padding (`pt-3` of `EDITOR_PAGE`). */
const BOX_CHROME_PX = 12
/** Row height until the terminal is on screen and measures its own. */
const FALLBACK_ROW_PX = 17
/** Rows the terminal has besides the cell's text: the status line, the command line, and one free row so that a new line (Enter) does not scroll the first one away before the terminal has grown. */
const SPARE_ROWS = 3

/** The height in px of the terminal box of a cell whose text takes `rows` screen rows in Neovim, `rowHeight` the px of one terminal row (0 while unknown). */
export function cellTerminalHeight(rows: number, rowHeight: number): number {
  return Math.ceil((rows + SPARE_ROWS) * (rowHeight > 0 ? rowHeight : FALLBACK_ROW_PX)) + BOX_CHROME_PX
}

const lineCount = (text: string) => text.split("\n").length

/** The extension of the files of the notebook's language: what the kernel reports (`.py`), else `py` for Python and the language name for a notebook that has not run yet. */
export function languageExtension(info: LanguageInfoLike | null): string | null {
  const name = info?.name?.toLowerCase()
  return info?.file_extension?.replace(/^\./, "") || (name === "python" ? "py" : name) || null
}

/** The name `set_cells` gives for Neovim's code buffer, only its extension matters: the filetype, and for a Python notebook the `.py` the language server analyzes. The backend reads it once, when it creates the buffer, and names the buffer after the notebook's file. */
const bufferName = (language: LanguageInfoLike | null) => `notebook.${languageExtension(language) ?? "txt"}`

const isCode = (cell: CellWidgetLike | null): cell is CellWidgetLike => cell?.model.type === "code"

const sameCells = (a: readonly CellText[], b: readonly CellText[]) => a.length === b.length && a.every((cell, index) => cell.id === b[index].id && cell.text === b[index].text && cell.kind === b[index].kind)

/** The keys of the cells without an id, one for each cell as long as it lives. `#` is in no nbformat id, so `#<n>` is no other cell's id. */
const keysOfIdless = new WeakMap<SharedCellLike, string>()
let idlessCount = 0

/** The cell's id, else the key it got for having none. */
function idOrKey(cell: SharedCellLike): string {
  if (cell.id) return cell.id
  let key = keysOfIdless.get(cell)
  if (key === undefined) {
    key = `#${++idlessCount}`
    keysOfIdless.set(cell, key)
  }
  return key
}

/** A code or markdown cell of the notebook: `key` is what Neovim knows it by, `index` its position among all cells (the notebook widget's index too). */
interface NeovimCell {
  key: string
  index: number
  kind: CellKind
  cell: SharedCellLike
}

/** Where the next cell to get the window puts Neovim's cursor, and whether the terminal scrolls into view there. */
interface PendingEntry {
  key: string
  cursor: CellCursor
  reveal: boolean
}

/**
 * Neovim on a notebook, both ways, with `terminal` as the session. Neovim holds all code cells as one buffer, all markdown cells as another, its window shows the active cell.
 *
 * - `setCells` hands Neovim the code and markdown cells whenever the notebook changes them (anything but Neovim's own edits): `sent` is what Neovim has, an update equal to it is not sent
 * - Neovim knows a cell by its `key`, the cell's id. A cell without one (a notebook below nbformat 4.5) gets `#1`, `#2`... for as long as it lives. A notebook file may repeat ids, so a later cell with a taken id gets `<id>#2` and so on: Neovim never gets a repeated id (it would refuse the cells), and everything it reports maps back to the cell by its key
 * - An edit in Neovim goes into its cell (`setCellSource`, outside the undo stack) and into `sent` first, so it is not sent back
 * - The active code cell gets the window (`focusCell`) however it became active: click, keyboard, run and advance, first load. A raw cell has no Neovim, its own editor shows
 * - A markdown cell shows its rendered text. Editing it (double-click, Enter: the notebook unrenders it) puts the terminal in its place with the window on it, until Esc in normal mode, Ctrl or Shift+Enter, `j`/`k` out of it, or another cell becomes active: then it is rendered again
 * - The notebook lands on a code cell wherever Neovim can follow: a notebook that opens on a markdown cell (a title) activates its first code cell once, and Shift+Enter moves to the next code cell as `j` does, never to the markdown cells between
 * - Neovim moving its cursor into another cell activates that cell. At the edge of a cell whose neighbor in the notebook is of the other kind (`exit`), that neighbor becomes active: a code cell with the cursor on its first or last line, a markdown cell rendered and the keys on the notebook
 * - Shift+Enter runs the cell and advances, Ctrl+Enter runs it and stays (a markdown cell: rendered)
 * - A hidden browser tab (`display: none`) loses the notebook's scroll position and its keyboard focus: when the notebook is on screen again the active cell is scrolled back into view and the terminal gets the keys
 * - Neovim draws every code cell (`render`, kept by key until the session ends or Neovim takes it back): the cells without the terminal show that render (`CellSnapshots`), a click on one makes it the active cell with the cursor on the clicked line
 * - The terminal is as tall as the screen rows Neovim draws the active cell's text in (`height`: wrapped lines, folds, virtual lines), the cell's line count until Neovim told them, and a cell's render takes the rows it took in Neovim: entering a cell does not move what is below it
 */
export class CellNeovim {
  /** The terminal calls these, `terminal.subscribe` hands them over. */
  readonly listener: CellTerminalListener = {
    edit: (id, text) => { this.edit(id, text) },
    focus: (id) => this.focusFromNeovim(id),
    run: (id, text, advance) => this.run(id, text, advance),
    leave: (id) => this.leave(id),
    exit: (id, direction) => this.exit(id, direction),
    save: () => this.deps.save(),
    render: (id, rendering) => this.render(id, rendering),
    height: (id, rows) => this.height(id, rows),
  }

  private terminal: CellTerminal
  private session: object | null
  private shown: string | null = null
  private sent: CellText[] | null = null
  private readonly subscribers = new Set<() => void>()
  private snapshot: CellNeovimSnapshot = { id: null, cell: null, rows: 1 }
  private started = false
  private placed = false
  private visibility: ResizeObserver | null = null
  private visible: boolean | null = null
  /** The renders of the cells by key, a new map on every change. */
  private renders: ReadonlyMap<string, CellRendering> = new Map()
  private readonly renderSubscribers = new Set<() => void>()
  /** The screen rows Neovim last said each cell's text takes (`cell_height`), by key. */
  private heights = new Map<string, number>()
  /** Where the cursor goes in the cell that gets the window next (a click on its render, `k` out of the cell below), until the window is pinned there. */
  private pending: PendingEntry | null = null
  /** The key of the markdown cell edited in Neovim: unrendered, the terminal in place of its text. */
  private editing: string | null = null
  /** The active markdown cell, whose rendering is watched: the notebook unrenders it for editing (double-click, Enter). */
  private watched: CellWidgetLike | null = null

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
    // Without ResizeObserver (a test environment) there is no hidden tab either
    if (typeof ResizeObserver !== "undefined") {
      this.visibility = new ResizeObserver(this.onResize)
      this.visibility.observe(this.deps.notebook.node)
    }
    this.reconcile()
  }

  dispose(): void {
    if (!this.started) return
    this.started = false
    this.deps.notebook.activeCellChanged.disconnect(this.onActiveChanged)
    this.deps.notebook.stateChanged.disconnect(this.onNotebookState)
    this.deps.model.ydoc.off("update", this.onModelUpdate)
    this.watch(null)
    this.visibility?.disconnect()
    this.visibility = null
    this.visible = null
    this.sent = null
    this.editing = null
    this.heights.clear()
    this.setRenders(new Map())
  }

  /** The terminal changed state, or is another session: what the old session held is gone, its renders too. */
  update(terminal: CellTerminal): void {
    this.terminal = terminal
    if (terminal.session !== this.session) {
      this.session = terminal.session
      this.sent = null
      this.shown = null
      this.heights.clear()
      this.setRenders(new Map())
    }
    if (this.started) this.reconcile()
  }

  readonly subscribe = (listener: () => void) => {
    this.subscribers.add(listener)
    return () => { this.subscribers.delete(listener) }
  }

  readonly getSnapshot = () => this.snapshot

  readonly subscribeRenders = (listener: () => void) => {
    this.renderSubscribers.add(listener)
    return () => { this.renderSubscribers.delete(listener) }
  }

  /** The render of every code cell Neovim drew, by key: what the cells without the terminal show. Empty without a session. */
  readonly getRenders = () => this.renders

  /** The notebook's scrolling element, where the windowed notebook adds the cells it renders. */
  get notebookNode(): HTMLElement {
    return this.deps.notebook.node
  }

  /** The code cells in notebook order with their widgets. */
  codeCellWidgets(): CodeCellWidget[] {
    const widgets = this.deps.notebook.widgets
    return this.cells().flatMap(({ key, index, kind }) => (kind === "code" && widgets[index] ? [{ key, cell: widgets[index] }] : []))
  }

  /** A click on the render of the cell `key`: the cell becomes the active one with Neovim's cursor on its `line` (0 is the first). */
  activateAt(key: string, line: number): void {
    const entry = this.cell(key)
    if (!entry) return
    this.enter(entry, { key, cursor: { line }, reveal: false })
  }

  private setRenders(renders: ReadonlyMap<string, CellRendering>) {
    if (renders.size === 0 && this.renders.size === 0) return
    this.renders = renders
    for (const subscriber of [...this.renderSubscribers]) subscriber()
  }

  /** A frame with a render keeps it for the cell `key`. One without (the render failed, or the notebook grew too long for renders) drops it: the cell shows its own editor again. */
  private render(key: string, value: unknown) {
    const rendering = readCellRendering(value)
    if (rendering) {
      this.setRenders(new Map(this.renders).set(key, rendering))
    } else if (this.renders.has(key)) {
      const renders = new Map(this.renders)
      renders.delete(key)
      this.setRenders(renders)
    }
  }

  /** The notebook's code and markdown cells in order with the key Neovim knows each by: the cell's id (or its key for having none), `<id>#2`, `<id>#3` for a repeat of an id (and past an id that is taken already). */
  private cells(): NeovimCell[] {
    const taken = new Set<string>()
    const cells: NeovimCell[] = []
    this.deps.model.cells.forEach((cell, index) => {
      const kind = cell.cell_type
      if (kind !== "code" && kind !== "markdown") return
      const id = idOrKey(cell)
      let key = id
      for (let repeat = 2; taken.has(key); repeat++) key = `${id}#${repeat}`
      taken.add(key)
      cells.push({ key, index, kind, cell })
    })
    return cells
  }

  private cell(key: string): NeovimCell | undefined {
    return this.cells().find((candidate) => candidate.key === key)
  }

  private activeEntry(): NeovimCell | undefined {
    const index = this.deps.notebook.activeCellIndex
    return this.cells().find((candidate) => candidate.index === index)
  }

  /** The screen rows of the cell `key`: what Neovim said, else its render's, else its line count (`source`). */
  private rowsOf(key: string, source: string): number {
    return this.heights.get(key) ?? this.renders.get(key)?.height ?? lineCount(source)
  }

  private publish(id: string | null) {
    const cell = id === null ? null : this.deps.notebook.activeCell
    const source = id === null ? undefined : this.cell(id)?.cell.getSource()
    const rows = id === null || source === undefined ? 1 : this.rowsOf(id, source)
    if (this.snapshot.id === id && this.snapshot.cell === cell && this.snapshot.rows === rows) return
    const entered = id !== null && this.snapshot.id !== id
    // A cell entered by a click on its render is on screen where it was clicked: the terminal takes its place without scrolling
    const reveal = entered ? this.pending?.key !== id || this.pending.reveal : this.snapshot.reveal
    this.snapshot = { id, cell, rows, reveal }
    for (const subscriber of [...this.subscribers]) subscriber()
    // After the notebook has finished the activation that got us here
    if (entered && reveal) void Promise.resolve().then(() => this.reveal())
  }

  /** Neovim's text of the cell `key` takes `rows` screen rows: the terminal fits them when it is in that cell. */
  private height(key: string, rows: number) {
    if (!Number.isInteger(rows) || rows < 1 || this.heights.get(key) === rows) return
    this.heights.set(key, rows)
    if (this.snapshot.id === key) this.publish(key)
  }

  /**
   * A notebook opens with its first cell active, often a markdown title: Neovim has nothing to show there and the first code cell would stay CodeMirror until a click.
   * Once, when the notebook has its first active cell, that cell becomes the first code cell. Later markdown cells the user picks stay.
   */
  private placeFirst() {
    if (this.placed || !this.deps.notebook.activeCell) return
    this.placed = true
    if (isCode(this.deps.notebook.activeCell)) return
    const first = this.cells().find((cell) => cell.kind === "code")
    if (first) this.activate(first.index)
  }

  /**
   * Bring Neovim to the notebook's state, whatever changed: the cells it holds, the markdown cell it edits, and the window on the active cell. Idempotent, every trigger calls it.
   */
  private reconcile() {
    this.placeFirst()
    const active = this.deps.notebook.activeCell
    const entry = this.activeEntry()
    this.watch(entry?.kind === "markdown" && active?.model.type === "markdown" ? active : null)
    if (this.editing !== null && entry?.key !== this.editing) this.stopEditing()
    if (!this.session || !this.pushCells()) {
      this.shown = null
      this.publish(null)
      return
    }
    if (!entry || active?.model.type !== entry.kind || (entry.kind === "markdown" && entry.key !== this.editing)) {
      this.shown = null
      this.publish(null)
      return
    }
    const id = entry.key
    if (this.shown !== id) {
      const cursor: CellCursor = this.pending?.key === id ? this.pending.cursor : "first"
      if (!this.terminal.focusCell(id, cursor)) {
        this.shown = null
        this.publish(null)
        return
      }
      this.claimKeys()
    }
    this.shown = id
    this.publish(id)
  }

  /** The cells go to Neovim unless it has them already. False when the session cannot take them. The name only matters to a buffer Neovim creates now: it keeps the one it has when the notebook's language changes. */
  private pushCells(): boolean {
    const cells = this.cells().map(({ key, kind, cell }) => ({ id: key, text: cell.getSource(), kind }))
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

  /** The active cell has no terminal: the keys go to the notebook (command mode `j`, `k`, Enter), which the terminal had. */
  private focusNotebook() {
    if (this.shown === null) this.deps.notebook.activeCell?.node.focus()
  }

  /** Watch the rendering of `widget`, the active markdown cell, and nothing else. */
  private watch(widget: CellWidgetLike | null) {
    if (widget === this.watched) return
    this.watched?.renderedChanged?.disconnect(this.onRendered)
    this.watched = widget
    widget?.renderedChanged?.connect(this.onRendered)
  }

  /** The notebook unrendered the active markdown cell for editing (double-click, Enter): Neovim edits it. Rendered while edited (the toolbar's run): editing is over. */
  private readonly onRendered = (sender: unknown, rendered: boolean) => {
    const entry = this.activeEntry()
    if (sender !== this.watched || entry?.kind !== "markdown") return
    if (rendered) {
      if (this.editing !== entry.key) return
      this.editing = null
    } else {
      if (!this.session || this.editing === entry.key) return
      this.startEditing(entry)
    }
    this.reconcile()
  }

  /** The markdown cell `entry` is edited in Neovim: unrendered, whatever was edited before rendered. */
  private startEditing(entry: NeovimCell) {
    if (this.editing === entry.key) return
    this.stopEditing()
    this.editing = entry.key
    const widget = this.deps.notebook.widgets[entry.index]
    if (widget?.rendered) widget.rendered = false
  }

  /** The markdown cell being edited is rendered again. */
  private stopEditing() {
    const key = this.editing
    if (key === null) return
    this.editing = null
    const entry = this.cell(key)
    const widget = entry && this.deps.notebook.widgets[entry.index]
    if (widget?.rendered === false) widget.rendered = true
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

  /** The notebook node has no size while a hidden tab holds it, and its size is back when the tab is shown again. */
  private readonly onResize = (entries: readonly ResizeObserverEntry[]) => {
    const visible = entries[entries.length - 1].contentRect.height > 0
    const wasHidden = this.visible === false
    this.visible = visible
    if (visible && wasHidden) this.show()
  }

  /** The notebook is on screen again: the browser dropped the scroll position of the hidden tab and the focus of the terminal. */
  private show() {
    if (this.shown === null) return
    this.reveal(() => this.claimKeys())
  }

  /**
   * The terminal of the active cell comes into view. The windowed notebook scrolls to the cell (it measures its cells as they render, an element's own `scrollIntoView` goes by the heights it
   * estimated and the cells above grow afterwards), then the terminal itself is scrolled to: the cell holds its outputs too, and a tall output keeps the cell in view while the terminal at its top is cut.
   */
  private reveal(then?: () => void) {
    const { notebook } = this.deps
    if (this.shown === null || this.visible === false) return
    const settled = () => {
      if (this.shown === null) return
      notebook.activeCell?.node.querySelector(`.${TERMINAL_HOST_CLASS}`)?.scrollIntoView?.({ block: "nearest" })
      then?.()
    }
    notebook.scrollToItem(notebook.activeCellIndex, "auto").then(settled, settled)
  }

  private readonly onModelUpdate = () => {
    this.reconcile()
  }

  /** Neovim's text of the cell `key` goes into it, the cell's index comes back. A key that is none of this notebook's cells is not ours: -1. */
  private edit(key: string, text: string): number {
    const entry = this.cell(key)
    if (!entry) return -1
    const held = this.sent?.find((cell) => cell.id === key)
    if (held) held.text = text
    if (entry.cell.getSource() !== text) this.deps.model.transact(() => setCellSource(entry.cell, text), false)
    return entry.index
  }

  private activate(index: number) {
    if (this.deps.notebook.activeCellIndex !== index) this.deps.notebook.activeCellIndex = index
  }

  /** The cell `entry` becomes the active one with the window pinned as `pending` says. */
  private enter(entry: NeovimCell, pending: PendingEntry | null) {
    this.pending = pending
    this.activate(entry.index)
    this.reconcile()
    this.pending = null
  }

  /**
   * Shift+Enter (`advance`) or Ctrl+Enter in Neovim: the cell with Neovim's text runs (a markdown cell is rendered). Ctrl+Enter stays. Shift+Enter goes on to the next code cell, as `j` crosses cells:
   * Jupyter's advance would stop at a markdown cell between. Without a next code cell the cell stays unless it is the last of the notebook, where Jupyter adds a cell.
   */
  private run(key: string, text: string, advance: boolean) {
    const index = this.edit(key, text)
    if (index < 0) return
    this.activate(index)
    if (this.editing === key) {
      this.stopEditing()
      this.reconcile()
    }
    const { notebook, model } = this.deps
    const next = this.cells().find((candidate) => candidate.kind === "code" && candidate.index > index)
    if (!advance || (!next && index < model.cells.length - 1)) {
      this.deps.run()
    } else {
      this.deps.runAndAdvance()
      if (next && notebook.activeCellIndex === index + 1 && !isCode(notebook.activeCell)) this.activate(next.index)
    }
    this.focusNotebook()
  }

  /** Esc in normal mode in the markdown cell `key`: rendered, it stays the active cell with the keys on the notebook. */
  private leave(key: string) {
    const entry = this.cell(key)
    if (entry?.kind !== "markdown") return
    this.activate(entry.index)
    if (this.editing === key) this.stopEditing()
    this.reconcile()
    this.focusNotebook()
  }

  /**
   * Neovim's cursor left the cell `key` past its edge: the cell next to it in the notebook becomes active, a code cell with the cursor on its line next to the edge.
   * The neighbor is the model's, raw cells included: Neovim's buffers hold no raw cells, so it may have exited toward the markdown cell past a raw one, which the raw cell, the real neighbor, then gets.
   */
  private exit(key: string, direction: CellExitDirection) {
    const entry = this.cell(key)
    if (!entry) return
    const index = entry.index + (direction === "up" ? -1 : 1)
    if (index < 0 || index >= this.deps.model.cells.length) return
    if (this.editing === key) this.stopEditing()
    const target = this.cells().find((candidate) => candidate.index === index)
    if (target?.kind === "code") {
      this.enter(target, { key: target.key, cursor: direction === "up" ? "last" : "first", reveal: true })
    } else {
      this.activate(index)
      this.reconcile()
    }
    this.focusNotebook()
  }

  /** Neovim pinned its window to the cell `key` itself, one of the same kind: the notebook follows, Neovim is not told. A markdown cell is edited there. */
  private focusFromNeovim(key: string) {
    const entry = this.cell(key)
    if (!entry) return
    if (entry.kind === "markdown") this.startEditing(entry)
    this.shown = key
    this.activate(entry.index)
    this.publish(key)
  }
}
