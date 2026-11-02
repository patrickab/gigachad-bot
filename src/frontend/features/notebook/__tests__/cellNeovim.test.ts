import { beforeEach, describe, expect, it, vi } from "vitest"
import { CellNeovim, TERMINAL_HOST_CLASS, cellTerminalHeight, languageExtension, type CellModelLike, type CellWidgetLike, type LanguageInfoLike, type NotebookWidgetLike } from "../cellNeovim"
import type { CellCursor, CellTerminal, CellText } from "../cellTerminal"
import type { NotebookCellContent, NotebookContent } from "../notebookSync"
import { FakeNotebook } from "./fakeNotebook"

class Signal<Args> {
  private readonly slots = new Set<(sender: unknown, args: Args) => void>()
  connect(slot: (sender: unknown, args: Args) => void) { this.slots.add(slot) }
  disconnect(slot: (sender: unknown, args: Args) => void) { this.slots.delete(slot) }
  emit(args: Args) { for (const slot of [...this.slots]) slot(this, args) }
  get size() { return this.slots.size }
}

const code = (id: string, source: string): NotebookCellContent => ({ cell_type: "code", id, source, metadata: {}, outputs: [], execution_count: null })
const markdown = (id: string, source: string): NotebookCellContent => ({ cell_type: "markdown", id, source, metadata: {} })
const content = (cells: NotebookCellContent[]): NotebookContent => ({ cells, metadata: {}, nbformat: 4, nbformat_minor: 5 })

/** A notebook widget over a `FakeNotebook`, with the signals of the real one. The cell list is the model's, as the real notebook rebuilds its widgets from it. */
class FakeWidget {
  readonly activeCellChanged = new Signal<CellWidgetLike | null>()
  readonly stateChanged = new Signal<{ name: string; newValue: unknown }>()
  readonly modes: string[] = []
  private index = 0

  constructor(private readonly model: FakeNotebook) {}

  get widgets(): CellWidgetLike[] {
    // The terminal is in every cell's node, as `ActiveCellTerminal` has it in the active one
    return this.model.cells.map((cell) => ({ node: Object.assign(document.createElement("div"), { innerHTML: `<div class="${TERMINAL_HOST_CLASS}"></div>` }), model: { id: cell.id, type: cell.cell_type } }))
  }

  get activeCellIndex() { return this.index }
  set activeCellIndex(value: number) {
    const widgets = this.widgets
    this.index = Math.max(0, Math.min(value, widgets.length - 1))
    this.activeCellChanged.emit(widgets[this.index] ?? null)
  }

  get activeCell() { return this.widgets[this.index] ?? null }

  setMode(mode: string) { this.modes.push(mode) }
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
  cellHeight() { return 16 }
  subscribe() { return () => {} }
  render() { return null }
}

const modelOf = (model: FakeNotebook) => model as unknown as CellModelLike

function setup(cells: NotebookCellContent[], language: LanguageInfoLike | null = { name: "python", file_extension: ".py" }) {
  const model = new FakeNotebook(content(cells))
  const notebook = new FakeWidget(model)
  const terminal = new FakeTerminal()
  const runAndAdvance = vi.fn()
  const save = vi.fn()
  const controller = new CellNeovim({ notebook: notebook as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => language, runAndAdvance, save })
  controller.start()
  return { model, notebook, terminal, controller, runAndAdvance, save }
}

describe("CellNeovim", () => {
  let queued: (() => void)[]
  beforeEach(() => {
    queued = []
    vi.spyOn(globalThis, "queueMicrotask").mockImplementation((callback) => { queued.push(callback) })
  })

  it("hands Neovim the code cells in order, markdown and raw cells left out, then pins the window to the active one", () => {
    const { terminal } = setup([code("a", "x = 1\ny = 2"), markdown("m", "# t"), code("b", "z")])
    expect(terminal.sets).toEqual([{ filename: "notebook.py", cells: [{ id: "a", text: "x = 1\ny = 2" }, { id: "b", text: "z" }] }])
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
    const controller = new CellNeovim({ notebook: new FakeWidget(model) as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => language, runAndAdvance: vi.fn(), save: vi.fn() })
    controller.start()
    language = { name: "python", file_extension: ".py" }
    model.setMetadata("language_info", { name: "python" })
    expect(terminal.sets.map((set) => set.filename)).toEqual(["notebook.txt"])
  })

  it("never hands Neovim a repeated id: a later cell with a taken id gets a number and what Neovim reports maps back to the right cell", () => {
    const { controller, model, notebook, runAndAdvance, terminal } = setup([code("a", "x"), code("a", "y"), markdown("m", "t"), code("a#2", "z")])
    expect(terminal.sets[0].cells).toEqual([{ id: "a", text: "x" }, { id: "a#2", text: "y" }, { id: "a#2#2", text: "z" }])
    notebook.activeCellIndex = 1
    expect(terminal.focuses.at(-1)).toEqual({ id: "a#2", cursor: "first" })

    controller.listener.edit("a#2", "yy")
    expect(model.cells.map((cell) => cell.getSource())).toEqual(["x", "yy", "t", "z"])
    expect(terminal.sets).toHaveLength(1)
    controller.listener.focus("a#2#2")
    expect(notebook.activeCellIndex).toBe(3)
    expect(controller.getSnapshot().id).toBe("a#2#2")
    controller.listener.run("a", "xx")
    expect(model.cells[0].getSource()).toBe("xx")
    expect(notebook.activeCellIndex).toBe(0)
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
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
    expect(terminal.sets.at(-1)).toEqual({ filename: "notebook.py", cells: [{ id: "a", text: "x = 1" }, { id: "b", text: "yz" }] })
    expect(terminal.sets).toHaveLength(2)
  })

  it("sends the change again when the session could not take it", () => {
    const { model, terminal } = setup([code("a", "x")])
    terminal.accepts = false
    model.cells[0].updateSource(1, 1, "y")
    terminal.accepts = true
    model.cells[0].updateSource(2, 2, "z")
    expect(terminal.sets.at(-1)?.cells).toEqual([{ id: "a", text: "xyz" }])
  })

  it("sends the cell list again when a code cell is added, deleted or moved", () => {
    const { model, terminal } = setup([code("a", "x"), code("b", "y")])
    model.insertCell(1, code("c", "z"))
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual(["a", "c", "b"])
    model.moveCell(0, 2)
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual(["c", "b", "a"])
    model.deleteCellRange(1, 2)
    expect(terminal.sets.at(-1)?.cells.map((cell) => cell.id)).toEqual(["c", "a"])
  })

  it("does not send what Neovim has: a markdown edit, a metadata change", () => {
    const { model, terminal } = setup([code("a", "x"), markdown("m", "# t")])
    model.cells[1].updateSource(0, 1, "##")
    model.setMetadata("k", "v")
    expect(terminal.sets).toHaveLength(1)
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
    controller.listener.run("a", "x = 1")
    expect(model.cells[0].getSource()).toBe("x = 1")
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
  })

  it("activates the cell a run is for when another one is active", () => {
    const { controller, notebook, runAndAdvance } = setup([code("a", "x"), code("b", "y")])
    notebook.activeCellIndex = 1
    controller.listener.run("a", "x")
    expect(notebook.activeCellIndex).toBe(0)
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
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
    const controller = new CellNeovim({ notebook: new FakeWidget(model) as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => ({ name: "python", file_extension: ".py" }), runAndAdvance: vi.fn(), save: vi.fn() })
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
    const controller = new CellNeovim({ notebook: new FakeWidget(model) as unknown as NotebookWidgetLike, model: modelOf(model), terminal, language: () => ({ name: "python" }), runAndAdvance: vi.fn(), save: vi.fn() })
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

  it("reports the shown cell and its line count", () => {
    const { controller } = setup([code("a", "x")])
    expect(controller.getSnapshot()).toMatchObject({ id: "a", lines: 1 })
    controller.listener.edit("a", "x\ny\nz")
    expect(controller.getSnapshot().lines).toBe(3)
  })

  it("stops listening when disposed", () => {
    const { controller, model, notebook, terminal } = setup([code("a", "x")])
    controller.dispose()
    model.cells[0].updateSource(1, 1, "y")
    expect(terminal.sets).toHaveLength(1)
    expect(notebook.activeCellChanged.size).toBe(0)
    expect(notebook.stateChanged.size).toBe(0)
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
