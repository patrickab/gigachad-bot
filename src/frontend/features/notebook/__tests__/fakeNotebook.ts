import type { JsonObject, JsonValue, NotebookCellContent, NotebookContent, SharedCellLike, SharedNotebookLike } from "../notebookSync"

/** A `@jupyter/ydoc` `YNotebook` stand-in with the same semantics for the members `SharedNotebookLike` lists, and its `ydoc.on("update")` firing on every change. */

const clone = <T>(value: T): T => (value === undefined ? value : (JSON.parse(JSON.stringify(value)) as T))

export class FakeCell implements SharedCellLike {
  source: string
  metadata: JsonObject
  outputs: JsonObject[]
  private count: number | null

  constructor(readonly id: string | undefined, readonly cell_type: string, source: string, metadata: JsonObject, outputs: JsonObject[], count: number | null, private readonly notify: () => void) {
    this.source = source
    this.metadata = metadata
    this.outputs = outputs
    this.count = count
  }

  get execution_count(): number | null { return this.count }
  set execution_count(value: number | null) {
    this.count = value
    this.notify()
  }

  getSource() { return this.source }

  updateSource(start: number, end: number, value = "") {
    this.source = this.source.slice(0, start) + value + this.source.slice(end)
    this.updates.push({ start, end, value })
    this.notify()
  }

  /** Every `updateSource` call, to check how little text a merge touches. */
  readonly updates: { start: number; end: number; value: string }[] = []

  getMetadata() { return clone(this.metadata) }
  setMetadata(key: string, value: JsonValue) {
    this.metadata[key] = clone(value)
    this.notify()
  }
  deleteMetadata(key: string) {
    delete this.metadata[key]
    this.notify()
  }

  getOutputs() { return clone(this.outputs) }
  setOutputs(outputs: JsonObject[]) {
    this.outputs = clone(outputs)
    this.notify()
  }

  toJSON(): NotebookCellContent {
    const cell: NotebookCellContent = { id: this.id, cell_type: this.cell_type, source: this.source, metadata: this.getMetadata() }
    if (this.cell_type === "code") {
      cell.outputs = this.getOutputs()
      cell.execution_count = this.count
    }
    return cell
  }
}

export interface FakeNotebookOptions {
  /** Metadata the notebook library adds to every cell it loads. */
  injectEditable?: boolean
}

export class FakeNotebook implements SharedNotebookLike {
  /** `YNotebook.cells`: the list as of the last finished transaction, while the Yjs array behind it (`live`) changes at once. */
  private visible: FakeCell[] = []
  private live: FakeCell[] = []
  private depth = 0
  metadata: JsonObject
  nbformat: number
  nbformat_minor: number
  isDisposed = false
  /** How many transactions ran and whether each was undoable. */
  readonly transactions: (boolean | undefined)[] = []
  private readonly listeners = new Set<() => void>()
  private next = 0

  readonly ydoc = {
    on: (_event: "update", listener: () => void) => { this.listeners.add(listener) },
    off: (_event: "update", listener: () => void) => { this.listeners.delete(listener) },
  }

  constructor(content: NotebookContent, private readonly options: FakeNotebookOptions = {}) {
    this.nbformat = content.nbformat
    this.nbformat_minor = content.nbformat_minor
    const metadata = clone(content.metadata)
    delete metadata.orig_nbformat
    this.metadata = metadata
    // Like jupyter-react's loader: a notebook below 4.5 keeps no ids, and a cell without one gets none
    const useId = content.nbformat === 4 && content.nbformat_minor >= 5
    content.cells.forEach((cell) => this.live.push(this.make(clone(cell), useId ? cell.id : undefined)))
    this.commit()
  }

  get cells(): readonly FakeCell[] { return this.visible }

  private commit() {
    if (this.depth === 0) this.visible = [...this.live]
  }

  private readonly notify = () => { for (const listener of [...this.listeners]) listener() }

  private make(cell: NotebookCellContent, id: string | undefined): FakeCell {
    const metadata = clone(cell.metadata ?? {})
    if (this.options.injectEditable) metadata.editable = true
    const source = Array.isArray(cell.source) ? cell.source.join("") : cell.source
    return new FakeCell(id, cell.cell_type, source, metadata, clone(cell.outputs ?? []), cell.cell_type === "code" ? (cell.execution_count ?? null) : null, this.notify)
  }

  insertCell(index: number, cell: NotebookCellContent) {
    // Like `YNotebook.insertCell`: a cell without an id gets a new one
    const made = this.make(clone(cell), cell.id ?? `generated-${this.next++}`)
    this.live.splice(index, 0, made)
    this.commit()
    this.notify()
    return made
  }

  deleteCellRange(from: number, to: number) {
    this.live.splice(from, to - from)
    this.commit()
    this.notify()
  }

  moveCell(from: number, to: number) {
    // like `YNotebook.moveCells`: the cell is read from the list as of the last finished transaction
    const cell = this.cells[from]
    if (!cell) throw new TypeError("undefined is not an object (evaluating 'this.getCell(fromIndex + idx).toJSON')")
    this.live.splice(from, 1)
    this.live.splice(to, 0, cell)
    this.commit()
    this.notify()
  }

  getMetadata() { return clone(this.metadata) }
  setMetadata(key: string, value: JsonValue) {
    this.metadata[key] = clone(value)
    this.notify()
  }
  deleteMetadata(key: string) {
    delete this.metadata[key]
    this.notify()
  }

  transact(f: () => void, undoable?: boolean) {
    this.transactions.push(undoable)
    this.depth++
    try {
      f()
    } finally {
      this.depth--
      this.commit()
    }
  }

  /** The user types: replace the text of cell `index`. */
  edit(index: number, source: string) {
    this.cells[index].source = source
    this.notify()
  }

  /** Like `YNotebook.toJSON`: no ids below nbformat 4.5. */
  toJSON(): NotebookContent {
    const prune = this.nbformat === 4 && this.nbformat_minor <= 4
    const cells = this.cells.map((cell) => {
      const json = cell.toJSON()
      if (prune) delete json.id
      return json
    })
    return { cells, metadata: this.getMetadata(), nbformat: this.nbformat, nbformat_minor: this.nbformat_minor }
  }
}
