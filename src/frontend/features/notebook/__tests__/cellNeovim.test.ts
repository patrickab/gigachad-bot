import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { CellNeovim, TERMINAL_HOST_CLASS, cellTerminalHeight, languageExtension, type CellModelLike, type CellWidgetLike, type LanguageInfoLike, type NotebookWidgetLike } from "../cellNeovim"
import type { CellCursor, CellTerminal, CellText } from "../cellTerminal"
import type { NotebookCellContent, NotebookContent } from "../notebookSync"
import { FakeNotebook } from "./fakeNotebook"

/** A Lumino signal: slots get the object it belongs to as the sender. */
class Signal<Args> {
  private readonly slots = new Set<(sender: unknown, args: Args) => void>()
  constructor(private readonly sender: unknown = null) {}
  connect(slot: (sender: unknown, args: Args) => void) { this.slots.add(slot) }
  disconnect(slot: (sender: unknown, args: Args) => void) { this.slots.delete(slot) }
  emit(args: Args) { for (const slot of [...this.slots]) slot(this.sender, args) }
  get size() { return this.slots.size }
}

const code = (id: string, source: string): NotebookCellContent => ({ cell_type: "code", id, source, metadata: {}, outputs: [], execution_count: null })
const markdown = (id: string, source: string): NotebookCellContent => ({ cell_type: "markdown", id, source, metadata: {} })
const raw = (id: string, source: string): NotebookCellContent => ({ cell_type: "raw", id, source, metadata: {} })
const content = (cells: NotebookCellContent[]): NotebookContent => ({ cells, metadata: {}, nbformat: 4, nbformat_minor: 5 })

/** A cell widget: the terminal is in every cell's node, as `ActiveCellTerminal` has it in the active one. A markdown cell is rendered until edited, as the notebook loads it. Like the real `MarkdownCell`, it reports a change of `rendered` later (the real one after its render), with the value it has then. */
class FakeCellWidget implements CellWidgetLike {
  readonly node = Object.assign(document.createElement("div"), { innerHTML: `<div class="${TERMINAL_HOST_CLASS}"></div>` })
  readonly renderedChanged?: Signal<boolean>
  /** How often the cell got the keyboard focus. */
  focused = 0
  private shown = true

  constructor(readonly model: { type: string }) {
    if (model.type === "markdown") this.renderedChanged = new Signal<boolean>(this)
    this.node.focus = () => { this.focused++ }
  }

  get rendered(): boolean | undefined { return this.renderedChanged ? this.shown : undefined }
  set rendered(value: boolean | undefined) {
    if (!this.renderedChanged || value === undefined || value === this.shown) return
    this.shown = value
    void Promise.resolve().then(() => this.renderedChanged?.emit(this.shown))
  }
}

/** A notebook widget over a `FakeNotebook`, with the signals of the real one. The cell list is the model's, as the real notebook rebuilds its widgets from it, a cell keeps its widget. */
class FakeWidget {
  readonly activeCellChanged = new Signal<CellWidgetLike | null>()
  readonly stateChanged = new Signal<{ name: string; newValue: unknown }>()
  readonly modes: string[] = []
  readonly node = document.createElement("div")
  /** Every `scrollToItem` call. */
  readonly scrolls: { index: number; align?: string }[] = []
  private index = 0
  private readonly made = new WeakMap<object, FakeCellWidget>()

  constructor(private readonly model: FakeNotebook) {}

  get widgets(): FakeCellWidget[] {
    return this.model.cells.map((cell) => {
      const known = this.made.get(cell)
      if (known) return known
      const widget = new FakeCellWidget({ type: cell.cell_type })
      this.made.set(cell, widget)
      return widget
    })
  }

  get activeCellIndex() { return this.index }
  set activeCellIndex(value: number) {
    const widgets = this.widgets
    this.index = Math.max(0, Math.min(value, widgets.length - 1))
    this.activeCellChanged.emit(widgets[this.index] ?? null)
  }

  get activeCell() { return this.widgets[this.index] ?? null }

  setMode(mode: string) { this.modes.push(mode) }

  scrollToItem(index: number, align?: string) {
    this.scrolls.push({ index, align })
    return Promise.resolve()
  }
}

class FakeTerminal implements CellTerminal {
  state: CellTerminal["state"] = "running"
  session: object | null = {}
  accepts = true
  /** Every `setCells` call, the cells copied. */
  readonly sets: { filename: string; cells: CellText[] }[] = []
  readonly focuses: { id: string; cursor: CellCursor }[] = []
  focused = 0
  restarts = 0

  setCells(filename: string, cells: readonly CellText[]) {
    if (!this.accepts) return false
    this.sets.push({ filename, cells: cells.map((cell) => ({ ...cell })) })
    return true
  }

  focusCell(id: string, cursor: CellCursor) {
    if (!this.accepts) return false
    this.focuses.push({ id, cursor })
    return true
  }

  focus() { this.focused++ }
  restart() { this.restarts++ }
  metrics() { return { fontFamily: "monospace", fontSize: 13, cellWidth: 8, cellHeight: 16, columns: 80 } }
  subscribe() { return () => {} }
  render() { return null }
}

const modelOf = (model: FakeNotebook) => model as unknown as CellModelLike

function setup(cells: NotebookCellContent[] | NotebookContent, language: LanguageInfoLike | null = { name: "python", file_extension: ".py" }) {
  const model = new FakeNotebook(Array.isArray(cells) ? content(cells) : cells)
  const notebook = new FakeWidget(model)
  const terminal = new FakeTerminal()
  const run = vi.fn()
  const runAndAdvance = vi.fn()
  const save = vi.fn()
  const controller = new CellNeovim({ notebook: notebook as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => language, run, runAndAdvance, save })
  controller.start()
  return { model, notebook, terminal, controller, run, runAndAdvance, save }
}

/** What Jupyter's advance does: the next cell becomes active, a code cell is added when the active one is the last. */
function advancing({ model, notebook, runAndAdvance }: ReturnType<typeof setup>) {
  runAndAdvance.mockImplementation(() => {
    if (notebook.activeCellIndex === model.cells.length - 1) model.insertCell(model.cells.length, code("new", ""))
    notebook.activeCellIndex++
  })
}

/** Lets the promise callbacks of the notebook's scrolling run. */
const settled = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

describe("CellNeovim", () => {
  let queued: (() => void)[]
  beforeEach(() => {
    queued = []
    vi.spyOn(globalThis, "queueMicrotask").mockImplementation((callback) => { queued.push(callback) })
  })
  afterEach(() => { Reflect.deleteProperty(Element.prototype, "scrollIntoView") })

  it("hands Neovim the code and markdown cells in order with their kind, raw cells left out, then pins the window to the active one", () => {
    const { terminal } = setup([code("a", "x = 1\ny = 2"), markdown("m", "# t"), raw("r", "raw"), code("b", "z")])
    expect(terminal.sets).toEqual([{ filename: "notebook.py", cells: [{ id: "a", text: "x = 1\ny = 2", kind: "code" }, { id: "m", text: "# t", kind: "markdown" }, { id: "b", text: "z", kind: "code" }] }])
    expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }])
  })

  it("names the buffer by the file extension of the notebook language", () => {
    expect(setup([code("a", "")], { name: "julia", file_extension: ".jl" }).terminal.sets[0].filename).toBe("notebook.jl")
    expect(setup([code("a", "")], { name: "python" }).terminal.sets[0].filename).toBe("notebook.py")
    expect(setup([code("a", "")], null).terminal.sets[0].filename).toBe("notebook.txt")
  })

  it("does not send the buffer again when the notebook only learns its language: Neovim keeps the name it has", () => {
    const model = new FakeNotebook(content([code("a", "x")]))
    const terminal = new FakeTerminal()
    let language: LanguageInfoLike | null = null
    const controller = new CellNeovim({ notebook: new FakeWidget(model) as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => language, run: vi.fn(), runAndAdvance: vi.fn(), save: vi.fn() })
    controller.start()
    language = { name: "python", file_extension: ".py" }
    model.setMetadata("language_info", { name: "python" })
    expect(terminal.sets.map((set) => set.filename)).toEqual(["notebook.txt"])
  })

  it("never hands Neovim a repeated id, across kinds: a later cell with a taken id gets a number and what Neovim reports maps back to the right cell", () => {
    const { controller, model, notebook, runAndAdvance, terminal } = setup([code("a", "x"), code("a", "y"), markdown("a", "t"), code("a#2", "z")])
    expect(terminal.sets[0].cells.map(({ id, kind }) => [id, kind])).toEqual([["a", "code"], ["a#2", "code"], ["a#3", "markdown"], ["a#2#2", "code"]])
    notebook.activeCellIndex = 1
    expect(terminal.focuses.at(-1)).toEqual({ id: "a#2", cursor: "first" })

    controller.listener.edit("a#2", "yy")
    expect(model.cells.map((cell) => cell.getSource())).toEqual(["x", "yy", "t", "z"])
    expect(terminal.sets).toHaveLength(1)
    controller.listener.focus("a#2#2")
    expect(notebook.activeCellIndex).toBe(3)
    expect(controller.getSnapshot().id).toBe("a#2#2")
    controller.listener.run("a", "xx", true)
    expect(model.cells[0].getSource()).toBe("xx")
    expect(notebook.activeCellIndex).toBe(0)
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
  })

  it("gives every cell of a notebook without ids (below nbformat 4.5) its own key for as long as it lives, and what Neovim reports maps back to it", () => {
    const legacy: NotebookContent = { ...content([code("", "a = 1"), markdown("", "# t"), code("", "b = a")]), nbformat_minor: 4 }
    for (const cell of legacy.cells) delete cell.id
    const harness = setup(legacy)
    const { controller, model, notebook, runAndAdvance, terminal } = harness
    advancing(harness)
    expect(model.cells.map((cell) => cell.id)).toEqual([undefined, undefined, undefined])
    const keys = terminal.sets[0].cells.map((cell) => cell.id)
    expect(terminal.sets[0].cells.map((cell) => cell.kind)).toEqual(["code", "markdown", "code"])
    for (const key of keys) expect(key).toMatch(/\S/)
    expect(new Set(keys).size).toBe(3)
    expect(terminal.focuses).toEqual([{ id: keys[0], cursor: "first" }])
    expect(controller.codeCellWidgets().map((widget) => widget.key)).toEqual([keys[0], keys[2]])

    model.insertCell(3, code("added", "c"))
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual([...keys, "added"])
    controller.listener.edit(keys[2], "b = a + 1")
    expect(model.cells.map((cell) => cell.getSource())).toEqual(["a = 1", "# t", "b = a + 1", "c"])
    controller.listener.run(keys[0], "a = 2", true)
    expect(model.cells[0].getSource()).toBe("a = 2")
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
    expect(notebook.activeCellIndex).toBe(2)
    expect(terminal.focuses.at(-1)).toEqual({ id: keys[2], cursor: "first" })
  })

  it("keeps the notebook out of the cell's editor: command mode, keys to the terminal", () => {
    const { notebook, terminal } = setup([code("a", "x")])
    expect(notebook.modes).toEqual(["command"])
    queued.splice(0).forEach((callback) => callback())
    expect(terminal.focused).toBe(1)
  })

  it("takes Neovim's edit into the cell outside the undo stack and writes back nothing", () => {
    const { controller, model, terminal } = setup([code("a", "x = 1")])
    controller.listener.edit("a", "x = 12")
    expect(model.cells[0].getSource()).toBe("x = 12")
    expect(model.transactions.at(-1)).toBe(false)
    expect(terminal.sets).toHaveLength(1)
  })

  it("touches only the text that differs", () => {
    const { controller, model } = setup([code("a", "print(1)\nprint(2)")])
    controller.listener.edit("a", "print(1)\nprint(22)")
    expect((model.cells[0] as unknown as { updates: unknown[] }).updates).toEqual([{ start: 16, end: 16, value: "2" }])
  })

  it("sends a change made in the notebook to Neovim once, of any cell", () => {
    const { model, terminal } = setup([code("a", "x = 1"), code("b", "y")])
    model.cells[1].updateSource(1, 1, "z")
    expect(terminal.sets.at(-1)).toEqual({ filename: "notebook.py", cells: [{ id: "a", text: "x = 1", kind: "code" }, { id: "b", text: "yz", kind: "code" }] })
    expect(terminal.sets).toHaveLength(2)
  })

  it("sends the change again when the session could not take it", () => {
    const { model, terminal } = setup([code("a", "x")])
    terminal.accepts = false
    model.cells[0].updateSource(1, 1, "y")
    terminal.accepts = true
    model.cells[0].updateSource(2, 2, "z")
    expect(terminal.sets.at(-1)?.cells).toEqual([{ id: "a", text: "xyz", kind: "code" }])
  })

  it("sends the cell list again when a cell is added, deleted, moved or changes its type", () => {
    const { model, terminal } = setup([code("a", "x"), code("b", "y")])
    model.insertCell(1, code("c", "z"))
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual(["a", "c", "b"])
    model.moveCell(0, 2)
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual(["c", "b", "a"])
    model.deleteCellRange(1, 2)
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual(["c", "a"])
    // A type change replaces the cell
    model.deleteCellRange(0, 1)
    model.insertCell(0, markdown("c", "z"))
    expect(terminal.sets.at(-1)?.cells).toEqual([{ id: "c", text: "z", kind: "markdown" }, { id: "a", text: "x", kind: "code" }])
  })

  it("sends a markdown edit made in the notebook, not a metadata change", () => {
    const { model, terminal } = setup([code("a", "x"), markdown("m", "# t")])
    model.cells[1].updateSource(0, 1, "##")
    expect(terminal.sets.at(-1)?.cells[1]).toEqual({ id: "m", text: "## t", kind: "markdown" })
    model.setMetadata("k", "v")
    expect(terminal.sets).toHaveLength(2)
  })

  it("pins the window to the active code cell however it became active, cursor on its first line", () => {
    const { notebook, terminal } = setup([code("a", "x"), markdown("m", "# t"), code("b", "y\nz")])
    // keyboard, click, run and advance, all set the active index
    notebook.activeCellIndex = 1
    notebook.activeCellIndex = 2
    expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }, { id: "b", cursor: "first" }])
    notebook.activeCellIndex = 0
    expect(terminal.focuses.at(-1)).toEqual({ id: "a", cursor: "first" })
  })

  it("shows no terminal in a markdown cell", () => {
    const { controller, notebook } = setup([code("a", "x"), markdown("m", "# t")])
    expect(controller.getSnapshot().id).toBe("a")
    notebook.activeCellIndex = 1
    expect(controller.getSnapshot().id).toBeNull()
  })

  it("does not pin the same cell twice", () => {
    const { model, notebook, terminal } = setup([code("a", "x")])
    model.cells[0].updateSource(1, 1, "y")
    notebook.activeCellIndex = 0
    expect(terminal.focuses).toHaveLength(1)
  })

  it("runs the cell with the text Neovim ended with, then advances", () => {
    const { controller, model, runAndAdvance } = setup([code("a", "x"), code("b", "y")])
    controller.listener.run("a", "x = 1", true)
    expect(model.cells[0].getSource()).toBe("x = 1")
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
  })

  it("activates the cell a run is for when another one is active", () => {
    const { controller, notebook, runAndAdvance } = setup([code("a", "x"), code("b", "y")])
    notebook.activeCellIndex = 1
    controller.listener.run("a", "x", true)
    expect(notebook.activeCellIndex).toBe(0)
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
  })

  it("runs on to the next code cell, not the markdown cell between: Neovim has no markdown cell to show", () => {
    const harness = setup([code("a", "x"), markdown("m", "# t"), markdown("n", "text"), code("b", "y")])
    advancing(harness)
    harness.controller.listener.run("a", "x", true)
    expect(harness.notebook.activeCellIndex).toBe(3)
    expect(harness.terminal.focuses.at(-1)).toEqual({ id: "b", cursor: "first" })
    expect(harness.controller.getSnapshot().id).toBe("b")
    expect(harness.run).not.toHaveBeenCalled()
  })

  it("lets Jupyter advance onto a code cell that follows directly", () => {
    const harness = setup([code("a", "x"), code("b", "y")])
    advancing(harness)
    harness.controller.listener.run("a", "x", true)
    expect(harness.notebook.activeCellIndex).toBe(1)
    expect(harness.terminal.focuses.at(-1)).toEqual({ id: "b", cursor: "first" })
  })

  it("stays in the cell when only markdown cells follow, Neovim stays on it", () => {
    const harness = setup([code("a", "x"), markdown("m", "# done")])
    advancing(harness)
    harness.controller.listener.run("a", "x = 2", true)
    expect(harness.run).toHaveBeenCalledTimes(1)
    expect(harness.runAndAdvance).not.toHaveBeenCalled()
    expect(harness.notebook.activeCellIndex).toBe(0)
    expect(harness.controller.getSnapshot().id).toBe("a")
    expect(harness.model.cells[0].getSource()).toBe("x = 2")
  })

  it("lets Jupyter add the cell after the last one of the notebook, Neovim follows into it", () => {
    const harness = setup([markdown("m", "# t"), code("a", "x")])
    advancing(harness)
    harness.controller.listener.run("a", "x", true)
    expect(harness.model.cells.map((cell) => cell.id)).toEqual(["m", "a", "new"])
    expect(harness.notebook.activeCellIndex).toBe(2)
    expect(harness.terminal.focuses.at(-1)).toEqual({ id: "new", cursor: "first" })
  })

  it("does not move when Jupyter did not advance, the kernel is not ready", () => {
    const harness = setup([code("a", "x"), markdown("m", "# t"), code("b", "y")])
    harness.controller.listener.run("a", "x", true)
    expect(harness.notebook.activeCellIndex).toBe(0)
  })

  it("runs the cell in place on Ctrl+Enter: Neovim stays on it", () => {
    const harness = setup([code("a", "x"), code("b", "y")])
    advancing(harness)
    harness.controller.listener.run("a", "x = 3", false)
    expect(harness.model.cells[0].getSource()).toBe("x = 3")
    expect(harness.run).toHaveBeenCalledTimes(1)
    expect(harness.runAndAdvance).not.toHaveBeenCalled()
    expect(harness.notebook.activeCellIndex).toBe(0)
    expect(harness.controller.getSnapshot().id).toBe("a")
  })

  it("opens a notebook that starts with a markdown cell on its first code cell, without any click", () => {
    const { controller, notebook, terminal } = setup([markdown("t", "# Title"), markdown("q", "Question"), code("a", "x"), code("b", "y")])
    expect(notebook.activeCellIndex).toBe(2)
    expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }])
    expect(controller.getSnapshot().id).toBe("a")
  })

  it("places the first code cell once: a markdown cell chosen later stays active", () => {
    const { controller, notebook } = setup([markdown("t", "# Title"), code("a", "x")])
    expect(notebook.activeCellIndex).toBe(1)
    notebook.activeCellIndex = 0
    expect(notebook.activeCellIndex).toBe(0)
    expect(controller.getSnapshot().id).toBeNull()
  })

  it("places the first code cell when the session comes up after the notebook did", () => {
    const model = new FakeNotebook(content([markdown("t", "# Title"), code("a", "x")]))
    const notebook = new FakeWidget(model)
    const terminal = new FakeTerminal()
    terminal.session = null
    terminal.state = "starting"
    const controller = new CellNeovim({ notebook: notebook as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => null, run: vi.fn(), runAndAdvance: vi.fn(), save: vi.fn() })
    controller.start()
    expect(notebook.activeCellIndex).toBe(1)
    terminal.session = {}
    controller.update(terminal)
    expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }])
  })

  it("leaves a notebook without code cells on its first cell", () => {
    const { notebook, terminal } = setup([markdown("t", "# Title"), markdown("q", "Question")])
    expect(notebook.activeCellIndex).toBe(0)
    expect(terminal.focuses).toEqual([])
  })

  it("follows Neovim into another cell without telling Neovim", () => {
    const { controller, notebook, terminal } = setup([code("a", "x"), markdown("m", "# t"), code("b", "y")])
    controller.listener.focus("b")
    expect(notebook.activeCellIndex).toBe(2)
    expect(controller.getSnapshot().id).toBe("b")
    expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }])
  })

  it("ignores a focus on an id that is none of the notebook's cells", () => {
    const { controller, notebook } = setup([code("a", "x")])
    controller.listener.focus("gone")
    expect(notebook.activeCellIndex).toBe(0)
    expect(controller.getSnapshot().id).toBe("a")
  })

  it("claims the keys again when the notebook goes to edit mode under the terminal", () => {
    const { notebook, terminal } = setup([code("a", "x")])
    queued.splice(0)
    notebook.stateChanged.emit({ name: "mode", newValue: "edit" })
    expect(notebook.modes).toEqual(["command", "command"])
    queued.splice(0).forEach((callback) => callback())
    expect(terminal.focused).toBe(1)
  })

  it("ignores edits of cells that are none of this notebook's", () => {
    const { controller, model } = setup([code("a", "x")])
    controller.listener.edit("other", "scratch")
    expect(model.cells[0].getSource()).toBe("x")
  })

  it("sends everything again once a new session runs", () => {
    const { controller, terminal } = setup([code("a", "x")])
    terminal.session = null
    terminal.state = "starting"
    controller.update(terminal)
    expect(controller.getSnapshot().id).toBeNull()
    terminal.session = {}
    terminal.state = "running"
    controller.update(terminal)
    expect(terminal.sets).toHaveLength(2)
    expect(terminal.focuses).toHaveLength(2)
  })

  it("attaches to the first active cell once the session came up, not before, without any click", () => {
    const model = new FakeNotebook(content([code("a", "x")]))
    const terminal = new FakeTerminal()
    terminal.session = null
    terminal.state = "starting"
    const controller = new CellNeovim({ notebook: new FakeWidget(model) as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => ({ name: "python", file_extension: ".py" }), run: vi.fn(), runAndAdvance: vi.fn(), save: vi.fn() })
    controller.start()
    expect(terminal.sets).toEqual([])
    expect(controller.getSnapshot().id).toBeNull()
    terminal.session = {}
    controller.update(terminal)
    expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }])
    expect(controller.getSnapshot().id).toBe("a")
  })

  it("retries when the session is up but cannot take the frames yet", () => {
    const model = new FakeNotebook(content([code("a", "x")]))
    const terminal = new FakeTerminal()
    terminal.accepts = false
    const controller = new CellNeovim({ notebook: new FakeWidget(model) as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => ({ name: "python" }), run: vi.fn(), runAndAdvance: vi.fn(), save: vi.fn() })
    controller.start()
    expect(controller.getSnapshot().id).toBeNull()
    terminal.accepts = true
    controller.update(terminal)
    expect(controller.getSnapshot().id).toBe("a")
  })

  it("restarts a stopped session when a code cell is activated", () => {
    const { notebook, terminal } = setup([code("a", "x"), code("b", "y")])
    terminal.state = "stopped"
    terminal.session = null
    notebook.activeCellIndex = 1
    expect(terminal.restarts).toBe(1)
  })

  it("reports the shown cell and its line count until Neovim tells the screen rows its text takes", () => {
    const { controller } = setup([code("a", "x"), code("b", "y")])
    expect(controller.getSnapshot()).toMatchObject({ id: "a", rows: 1 })
    controller.listener.edit("a", "x\ny\nz")
    expect(controller.getSnapshot().rows).toBe(3)

    controller.listener.height("a", 5) // a line of it wraps in Neovim's window
    expect(controller.getSnapshot().rows).toBe(5)
    controller.listener.height("b", 9) // another cell's rows, kept for when it is shown
    controller.listener.height("a", 0)
    expect(controller.getSnapshot()).toMatchObject({ id: "a", rows: 5 })
  })

  it("sizes a cell entered without its screen rows to the rows of its render", () => {
    const { controller, notebook } = setup([code("a", "x"), code("b", "y")])
    controller.listener.render("b", { styles: [{}], lines: [[["y", 0]]], base: {}, rows: [0], height: 4 })
    notebook.activeCellIndex = 1
    expect(controller.getSnapshot()).toMatchObject({ id: "b", rows: 4 })
    controller.listener.height("b", 2)
    expect(controller.getSnapshot().rows).toBe(2)
  })

  it("brings the cell that becomes active into view, the terminal in it too, not a markdown cell", async () => {
    const reveal = vi.fn()
    Element.prototype.scrollIntoView = reveal
    const { notebook } = setup([code("a", "x"), markdown("m", "# t"), code("b", "y")])
    await settled()
    expect(notebook.scrolls).toEqual([{ index: 0, align: "auto" }])
    expect(reveal).toHaveBeenCalledTimes(1)
    expect(reveal).toHaveBeenCalledWith({ block: "nearest" })

    notebook.activeCellIndex = 2
    await settled()
    expect(notebook.scrolls.at(-1)).toEqual({ index: 2, align: "auto" })
    expect(reveal).toHaveBeenCalledTimes(2)

    notebook.activeCellIndex = 1
    await settled()
    expect(notebook.scrolls).toHaveLength(2)
  })

  it("does not scroll again for an edit of the cell that is shown", async () => {
    const { controller, notebook } = setup([code("a", "x")])
    await settled()
    controller.listener.edit("a", "x\ny\nz")
    await settled()
    expect(notebook.scrolls).toHaveLength(1)
  })

  describe("when the browser tab holding the notebook is hidden and shown again", () => {
    afterEach(() => { vi.unstubAllGlobals() })

    /** A `ResizeObserver` that reports what the test says: the notebook's height. */
    function observing() {
      const callbacks: ResizeObserverCallback[] = []
      vi.stubGlobal("ResizeObserver", class {
        constructor(callback: ResizeObserverCallback) { callbacks.push(callback) }
        observe() {}
        disconnect() {}
      })
      return (height: number) => callbacks.forEach((callback) => callback([{ contentRect: { height } } as ResizeObserverEntry], {} as ResizeObserver))
    }

    it("scrolls the active cell back into view, then the terminal in it, and gives the terminal the keys: the browser dropped all of it", async () => {
      const report = observing()
      const reveal = vi.fn()
      Element.prototype.scrollIntoView = reveal
      const { notebook, terminal } = setup([markdown("t", "# Title"), code("a", "x")])
      await settled()
      notebook.scrolls.splice(0)
      reveal.mockClear()
      report(700)
      report(0)
      queued.splice(0)
      expect(notebook.scrolls).toEqual([])

      report(700)
      await settled()
      expect(notebook.scrolls).toEqual([{ index: 1, align: "auto" }])
      expect(reveal).toHaveBeenCalledTimes(1)
      expect(reveal).toHaveBeenCalledWith({ block: "nearest" })
      queued.splice(0).forEach((callback) => callback())
      expect(terminal.focused).toBe(1)
    })

    it("does nothing for the first size report, the notebook was never hidden", () => {
      const report = observing()
      const { notebook } = setup([code("a", "x")])
      report(700)
      report(650)
      expect(notebook.scrolls).toEqual([])
    })

    it("does nothing while no cell has the terminal", () => {
      const report = observing()
      const { notebook } = setup([markdown("t", "# Title")])
      report(0)
      report(700)
      expect(notebook.scrolls).toEqual([])
    })

    it("stops watching the size when disposed", () => {
      const disconnect = vi.fn()
      vi.stubGlobal("ResizeObserver", class {
        observe() {}
        disconnect() { disconnect() }
      })
      const { controller } = setup([code("a", "x")])
      controller.dispose()
      expect(disconnect).toHaveBeenCalledTimes(1)
    })
  })

  it("stops listening when disposed", () => {
    const { controller, model, notebook, terminal } = setup([code("a", "x")])
    controller.dispose()
    model.cells[0].updateSource(1, 1, "y")
    expect(terminal.sets).toHaveLength(1)
    expect(notebook.activeCellChanged.size).toBe(0)
    expect(notebook.stateChanged.size).toBe(0)
  })

  describe("a markdown cell", () => {
    /** A double-click on the rendered cell `index`, or Enter on it: the notebook activates and unrenders it. */
    const edit = async ({ notebook }: ReturnType<typeof setup>, index: number) => {
      notebook.activeCellIndex = index
      notebook.widgets[index].rendered = false
      await settled()
    }

    it("stays rendered and without the terminal while it is only the active cell", async () => {
      const { controller, notebook, terminal } = setup([code("a", "x"), markdown("m", "# t")])
      notebook.activeCellIndex = 1
      await settled()
      expect(notebook.widgets[1].rendered).toBe(true)
      expect(controller.getSnapshot().id).toBeNull()
      expect(terminal.focuses).toEqual([{ id: "a", cursor: "first" }])
    })

    it("is edited in Neovim once the notebook unrenders it: the window on it, the keys to the terminal", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t\n\ntext")])
      await edit(harness, 1)
      expect(harness.terminal.focuses.at(-1)).toEqual({ id: "m", cursor: "first" })
      expect(harness.controller.getSnapshot()).toMatchObject({ id: "m", rows: 3 })
      expect(harness.notebook.modes.at(-1)).toBe("command")
    })

    it("keeps the notebook's own editor while Neovim is not running", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t")])
      harness.terminal.session = null
      harness.terminal.state = "starting"
      harness.controller.update(harness.terminal)
      await edit(harness, 1)
      expect(harness.terminal.focuses).toHaveLength(1)
      expect(harness.controller.getSnapshot().id).toBeNull()
      expect(harness.notebook.widgets[1].rendered).toBe(false)
    })

    it("takes Neovim's edits", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t")])
      await edit(harness, 1)
      harness.controller.listener.edit("m", "# new")
      expect(harness.model.cells[1].getSource()).toBe("# new")
      expect(harness.terminal.sets).toHaveLength(1)
    })

    it("is rendered again on Esc and stays the active cell, the keys on the notebook", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t")])
      await edit(harness, 1)
      harness.controller.listener.leave("m")
      await settled()
      expect(harness.notebook.widgets[1].rendered).toBe(true)
      expect(harness.notebook.activeCellIndex).toBe(1)
      expect(harness.controller.getSnapshot().id).toBeNull()
      expect(harness.notebook.widgets[1].focused).toBe(1)
    })

    it("is rendered in place with Neovim's text on Ctrl+Enter", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t"), code("b", "y")])
      advancing(harness)
      await edit(harness, 1)
      harness.controller.listener.run("m", "# done", false)
      await settled()
      expect(harness.model.cells[1].getSource()).toBe("# done")
      expect(harness.notebook.widgets[1].rendered).toBe(true)
      expect(harness.run).toHaveBeenCalledTimes(1)
      expect(harness.runAndAdvance).not.toHaveBeenCalled()
      expect(harness.notebook.activeCellIndex).toBe(1)
      expect(harness.controller.getSnapshot().id).toBeNull()
      expect(harness.notebook.widgets[1].focused).toBe(1)
    })

    it("is rendered on Shift+Enter and the next code cell gets the window", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t"), markdown("n", "text"), code("b", "y")])
      advancing(harness)
      await edit(harness, 1)
      harness.controller.listener.run("m", "# t", true)
      await settled()
      expect(harness.notebook.widgets[1].rendered).toBe(true)
      expect(harness.notebook.activeCellIndex).toBe(3)
      expect(harness.terminal.focuses.at(-1)).toEqual({ id: "b", cursor: "first" })
      expect(harness.controller.getSnapshot().id).toBe("b")
    })

    it("is rendered when another cell becomes active", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t")])
      await edit(harness, 1)
      harness.notebook.activeCellIndex = 0
      await settled()
      expect(harness.notebook.widgets[1].rendered).toBe(true)
      expect(harness.terminal.focuses.at(-1)).toEqual({ id: "a", cursor: "first" })
      expect(harness.controller.getSnapshot().id).toBe("a")
    })

    it("lets the terminal go when the notebook renders it itself", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t")])
      await edit(harness, 1)
      harness.notebook.widgets[1].rendered = true
      await settled()
      expect(harness.controller.getSnapshot().id).toBeNull()
      expect(harness.notebook.activeCellIndex).toBe(1)
    })

    it("follows Neovim into the markdown cell below: that one is edited, the one left is rendered", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t"), markdown("n", "text")])
      await edit(harness, 1)
      harness.controller.listener.focus("n")
      await settled()
      expect(harness.notebook.activeCellIndex).toBe(2)
      expect(harness.notebook.widgets[1].rendered).toBe(true)
      expect(harness.notebook.widgets[2].rendered).toBe(false)
      expect(harness.controller.getSnapshot().id).toBe("n")
      expect(harness.terminal.focuses.at(-1)).toEqual({ id: "m", cursor: "first" })
    })

    it("is rendered when Neovim leaves it past its edge, the code cell below gets the window on its first line", async () => {
      const harness = setup([code("a", "x"), markdown("m", "# t"), code("b", "y\nz")])
      await edit(harness, 1)
      harness.controller.listener.exit("m", "down")
      await settled()
      expect(harness.notebook.activeCellIndex).toBe(2)
      expect(harness.notebook.widgets[1].rendered).toBe(true)
      expect(harness.terminal.focuses.at(-1)).toEqual({ id: "b", cursor: "first" })
      expect(harness.controller.getSnapshot().id).toBe("b")
    })

    it("gives the code cell above the window on its last line when Neovim leaves upward", async () => {
      const harness = setup([code("a", "x\ny"), markdown("m", "# t")])
      await edit(harness, 1)
      harness.controller.listener.exit("m", "up")
      await settled()
      expect(harness.notebook.activeCellIndex).toBe(0)
      expect(harness.terminal.focuses.at(-1)).toEqual({ id: "a", cursor: "last" })
    })
  })

  it("moves from a code cell past its edge onto the markdown cell next to it, rendered, the keys on the notebook", async () => {
    const { controller, notebook } = setup([code("a", "x"), markdown("m", "# t"), code("b", "y")])
    controller.listener.exit("a", "down")
    await settled()
    expect(notebook.activeCellIndex).toBe(1)
    expect(notebook.widgets[1].rendered).toBe(true)
    expect(controller.getSnapshot().id).toBeNull()
    expect(notebook.widgets[1].focused).toBe(1)
  })

  it("stays at the edge of the notebook", () => {
    const { controller, notebook } = setup([code("a", "x"), markdown("m", "# t")])
    controller.listener.exit("a", "up")
    expect(notebook.activeCellIndex).toBe(0)
    expect(controller.getSnapshot().id).toBe("a")
  })

  it("saves the notebook on :w", () => {
    const { controller, save } = setup([code("a", "x")])
    controller.listener.save()
    expect(save).toHaveBeenCalledTimes(1)
  })
})

describe("cellTerminalHeight", () => {
  it("holds the rows the text and the status and command lines need, at any row height", () => {
    for (const rowHeight of [15, 16, 16.5, 17.34, 21]) {
      for (const lines of [1, 2, 5, 40]) expect(cellTerminalHeight(lines, rowHeight)).toBeGreaterThanOrEqual((lines + 2) * rowHeight)
    }
  })

  it("is one row taller for every line of the text", () => {
    expect(cellTerminalHeight(6, 16.5) - cellTerminalHeight(5, 16.5)).toBeGreaterThanOrEqual(16.5)
    expect(cellTerminalHeight(6, 16.5) - cellTerminalHeight(5, 16.5)).toBeLessThan(18)
  })

  it("assumes a row height until the terminal measures its own", () => {
    expect(cellTerminalHeight(3, 0)).toBeGreaterThan(0)
    expect(cellTerminalHeight(3, 0)).toBe(cellTerminalHeight(3, 17))
  })
})

describe("languageExtension", () => {
  it("takes the extension the kernel reports, else the name of the language", () => {
    expect(languageExtension({ name: "python", file_extension: ".py" })).toBe("py")
    expect(languageExtension({ name: "Python" })).toBe("py")
    expect(languageExtension({ name: "R", file_extension: ".r" })).toBe("r")
    expect(languageExtension({ name: "Julia" })).toBe("julia")
    expect(languageExtension({})).toBeNull()
    expect(languageExtension(null)).toBeNull()
  })
})
