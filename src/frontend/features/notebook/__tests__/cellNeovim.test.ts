import { beforeEach, describe, expect, it, vi } from "vitest"
import { CellNeovim, TERMINAL_HOST_CLASS, cellTerminalHeight, languageExtension, type CellModelLike, type CellWidgetLike, type LanguageInfoLike, type NotebookWidgetLike } from "../cellNeovim"
import type { CellCursor, CellDirection, CellTerminal } from "../cellTerminal"
import type { NotebookCellContent, NotebookContent, SharedCellLike } from "../notebookSync"
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

/** A notebook widget over a `FakeNotebook`, with the signals of the real one. */
class FakeWidget {
  readonly widgets: (CellWidgetLike & { changed: Signal<unknown>; renderedChanged: Signal<boolean> })[] = []
  readonly activeCellChanged = new Signal<CellWidgetLike | null>()
  readonly stateChanged = new Signal<{ name: string; newValue: unknown }>()
  readonly modes: string[] = []
  private index = 0

  constructor(private readonly model: FakeNotebook) {
    model.cells.forEach((cell) => this.widgets.push(this.widgetOf(cell)))
  }

  private widgetOf(cell: SharedCellLike & { source?: string }) {
    const changed = new Signal<unknown>()
    const renderedChanged = new Signal<boolean>()
    const model = this.model
    // A cell widget's `rendered` (markdown) starts true
    let rendered = cell.cell_type === "markdown"
    const widget = {
      // The terminal is in every cell's node, as `ActiveCellTerminal` has it in the active one
      node: Object.assign(document.createElement("div"), { innerHTML: `<div class="${TERMINAL_HOST_CLASS}"></div>` }),
      model: {
        id: cell.id,
        type: cell.cell_type,
        sharedModel: {
          getSource: () => model.cells.find((candidate) => candidate.id === cell.id)?.getSource() ?? "",
          updateSource: (start: number, end: number, value?: string) => model.cells.find((candidate) => candidate.id === cell.id)?.updateSource(start, end, value),
          changed,
        },
      },
      get rendered() { return rendered },
      set rendered(value: boolean) {
        rendered = value
        renderedChanged.emit(value)
      },
      changed,
      renderedChanged,
    }
    return widget as unknown as CellWidgetLike & { changed: Signal<unknown>; renderedChanged: Signal<boolean> }
  }

  get activeCellIndex() { return this.index }
  set activeCellIndex(value: number) {
    this.index = Math.max(0, Math.min(value, this.widgets.length - 1))
    this.activeCellChanged.emit(this.widgets[this.index] ?? null)
  }

  get activeCell() { return this.widgets[this.index] ?? null }

  setMode(mode: string) { this.modes.push(mode) }

  /** What the notebook does when the source of cell `index` changes under it (the user, a file merge). */
  sourceChanged(index: number) { this.widgets[index].changed.emit({ sourceChange: true }) }
}

class FakeTerminal implements CellTerminal {
  state: CellTerminal["state"] = "running"
  session: object | null = {}
  accepts = true
  readonly calls: string[] = []
  readonly opens: { key: string; filename: string; text: string; cursor: CellCursor }[] = []
  readonly sets: { key: string; text: string }[] = []
  readonly closes: string[] = []
  focused = 0
  restarts = 0

  open(key: string, filename: string, text: string, cursor: CellCursor) {
    if (!this.accepts) return false
    this.opens.push({ key, filename, text, cursor })
    return true
  }

  set(key: string, text: string) {
    if (!this.accepts) return false
    this.sets.push({ key, text })
    return true
  }

  close(key: string) {
    this.closes.push(key)
    return true
  }

  focus() { this.focused++ }
  restart() { this.restarts++ }
  cellHeight() { return 16 }
  subscribe() { return () => {} }
  render() { return null }
}

/** The shared notebook as the controller sees it: the cell list as of now, whatever the model does later. */
function modelOf(model: FakeNotebook, changed: Signal<unknown>): CellModelLike {
  return { get cells() { return model.cells }, transact: (f: () => void, undoable?: boolean) => model.transact(f, undoable), changed } as unknown as CellModelLike
}

function setup(cells: NotebookCellContent[], language: LanguageInfoLike | null = { name: "python", file_extension: ".py" }) {
  const model = new FakeNotebook(content(cells))
  const notebook = new FakeWidget(model)
  const terminal = new FakeTerminal()
  const runAndAdvance = vi.fn()
  const save = vi.fn()
  // `changed` reports the cell list: the fake notebook has none, the test emits it
  const changed = new Signal<unknown>()
  const controller = new CellNeovim({
    tabId: "tab",
    notebook: notebook as unknown as NotebookWidgetLike,
    model: modelOf(model, changed),
    terminal,
    language: () => language,
    runAndAdvance,
    save,
  })
  controller.start()
  return { model, notebook, terminal, controller, runAndAdvance, save, changed }
}

const key = (id: string) => `tab#${id}`

describe("CellNeovim", () => {
  let queued: (() => void)[]
  beforeEach(() => {
    queued = []
    vi.spyOn(globalThis, "queueMicrotask").mockImplementation((callback) => { queued.push(callback) })
  })

  it("opens the active cell with its text and a filetype from the notebook language", () => {
    const { terminal } = setup([code("a", "x = 1\ny = 2"), markdown("b", "# t")])
    expect(terminal.opens).toEqual([{ key: key("a"), filename: "cell.py", text: "x = 1\ny = 2", cursor: "first" }])
  })

  it("names the filetype by the file extension of the notebook language", () => {
    expect(setup([code("a", "")], { name: "julia", file_extension: ".jl" }).terminal.opens[0].filename).toBe("cell.jl")
    expect(setup([code("a", "")], null).terminal.opens[0].filename).toBe("cell.txt")
  })

  it("keeps the notebook out of the cell's editor: command mode, keys to the terminal", () => {
    const { notebook, terminal } = setup([code("a", "x")])
    expect(notebook.modes).toEqual(["command"])
    queued.splice(0).forEach((callback) => callback())
    expect(terminal.focused).toBe(1)
  })

  it("does not focus a terminal that is still in another cell", () => {
    const { notebook, terminal } = setup([code("a", "x"), code("b", "y")])
    queued.splice(0)
    notebook.widgets[1].node.innerHTML = ""
    notebook.activeCellIndex = 1
    queued.splice(0).forEach((callback) => callback())
    expect(terminal.focused).toBe(0)
  })

  it("takes Neovim's edit into the cell outside the undo stack and writes back nothing", () => {
    const { controller, model, terminal, notebook } = setup([code("a", "x = 1")])
    controller.listener.edit(key("a"), "x = 12")
    expect(model.cells[0].getSource()).toBe("x = 12")
    expect(model.transactions.at(-1)).toBe(false)
    // the notebook reports the change it just got: it is what Neovim sent
    notebook.sourceChanged(0)
    expect(terminal.sets).toEqual([])
  })

  it("touches only the text that differs", () => {
    const { controller, model } = setup([code("a", "print(1)\nprint(2)")])
    controller.listener.edit(key("a"), "print(1)\nprint(22)")
    expect((model.cells[0] as unknown as { updates: unknown[] }).updates).toEqual([{ start: 16, end: 16, value: "2" }])
  })

  it("sends a change made in the notebook to Neovim once", () => {
    const { model, terminal, notebook } = setup([code("a", "x = 1")])
    model.cells[0].updateSource(4, 5, "7")
    notebook.sourceChanged(0)
    notebook.sourceChanged(0)
    expect(terminal.sets).toEqual([{ key: key("a"), text: "x = 7" }])
  })

  it("sends the change again when the session could not take it", () => {
    const { model, terminal, notebook } = setup([code("a", "x")])
    terminal.accepts = false
    model.cells[0].updateSource(1, 1, "y")
    notebook.sourceChanged(0)
    terminal.accepts = true
    notebook.sourceChanged(0)
    expect(terminal.sets).toEqual([{ key: key("a"), text: "xy" }])
  })

  it("does not send the typing of a cell Neovim does not show", () => {
    const { model, terminal, notebook } = setup([code("a", "x"), code("b", "y")])
    model.cells[1].updateSource(1, 1, "z")
    notebook.widgets[1].changed.emit({})
    expect(terminal.sets).toEqual([])
  })

  it("runs the cell with the text Neovim ended with, then advances", () => {
    const { controller, model, runAndAdvance } = setup([code("a", "x"), code("b", "y")])
    controller.listener.run(key("a"), "x = 1")
    expect(model.cells[0].getSource()).toBe("x = 1")
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
  })

  it("activates the cell a run is for when another one is active", () => {
    const { controller, notebook, runAndAdvance } = setup([code("a", "x"), code("b", "y")])
    notebook.activeCellIndex = 1
    controller.listener.run(key("a"), "x")
    expect(notebook.activeCellIndex).toBe(0)
    expect(runAndAdvance).toHaveBeenCalledTimes(1)
  })

  it("opens the next cell when the active one changes, cursor on its first line", () => {
    const { notebook, terminal } = setup([code("a", "x"), code("b", "y\nz")])
    notebook.activeCellIndex = 1
    expect(terminal.opens.at(-1)).toEqual({ key: key("b"), filename: "cell.py", text: "y\nz", cursor: "first" })
  })

  it("enters the next cell on its first line and the previous one on its last line", () => {
    const { controller, notebook, terminal } = setup([code("a", "x"), code("b", "y"), code("c", "z")])
    controller.listener.exit(key("a"), "next", "x")
    expect(notebook.activeCellIndex).toBe(1)
    expect(terminal.opens.at(-1)?.cursor).toBe("first")
    controller.listener.exit(key("b"), "prev" satisfies CellDirection, "y")
    expect(notebook.activeCellIndex).toBe(0)
    expect(terminal.opens.at(-1)).toMatchObject({ key: key("a"), cursor: "last" })
  })

  it("takes the text of the cell it leaves", () => {
    const { controller, model } = setup([code("a", "x"), code("b", "y")])
    controller.listener.exit(key("a"), "next", "x = 5")
    expect(model.cells[0].getSource()).toBe("x = 5")
  })

  it("stays put at the first and last cell", () => {
    const { controller, notebook, terminal } = setup([code("a", "x"), code("b", "y")])
    controller.listener.exit(key("a"), "prev", "x")
    expect(notebook.activeCellIndex).toBe(0)
    notebook.activeCellIndex = 1
    const opens = terminal.opens.length
    controller.listener.exit(key("b"), "next", "y")
    expect(notebook.activeCellIndex).toBe(1)
    expect(terminal.opens).toHaveLength(opens)
  })

  it("enters a rendered markdown cell as an editable one", () => {
    const { controller, notebook, terminal } = setup([code("a", "x"), markdown("b", "# t")])
    controller.listener.exit(key("a"), "next", "x")
    expect(notebook.widgets[1].rendered).toBe(false)
    expect(terminal.opens.at(-1)).toMatchObject({ key: key("b"), filename: "cell.md" })
  })

  it("renders the markdown cell it was left from, as running it does", () => {
    const { controller, notebook } = setup([code("a", "x"), markdown("b", "# t"), code("c", "z")])
    controller.listener.exit(key("a"), "next", "x")
    expect(notebook.widgets[1].rendered).toBe(false)
    controller.listener.exit(key("b"), "next", "# t")
    expect(notebook.widgets[1].rendered).toBe(true)
    expect(notebook.activeCellIndex).toBe(2)
  })

  it("keeps a markdown cell in its editor when there is no cell to go to", () => {
    const { controller, notebook } = setup([markdown("a", "# t")])
    notebook.widgets[0].rendered = false
    controller.listener.exit(key("a"), "next", "# t")
    expect(notebook.widgets[0].rendered).toBe(false)
  })

  it("leaves a rendered markdown cell to the notebook, and takes it once it is edited", () => {
    const { notebook, terminal } = setup([markdown("a", "# t")])
    expect(terminal.opens).toEqual([])
    notebook.widgets[0].rendered = false
    expect(terminal.opens).toEqual([{ key: key("a"), filename: "cell.md", text: "# t", cursor: "first" }])
  })

  it("claims the keys again when the notebook goes to edit mode under the terminal", () => {
    const { notebook, terminal } = setup([code("a", "x")])
    queued.splice(0)
    notebook.stateChanged.emit({ name: "mode", newValue: "edit" })
    expect(notebook.modes).toEqual(["command", "command"])
    queued.splice(0).forEach((callback) => callback())
    expect(terminal.focused).toBe(1)
  })

  it("wipes the buffer of a deleted cell", () => {
    const { model, terminal, notebook, changed } = setup([code("a", "x"), code("b", "y")])
    notebook.activeCellIndex = 1
    model.deleteCellRange(0, 1)
    changed.emit({})
    expect(terminal.closes).toEqual([key("a")])
  })

  it("ignores edits of buffers that are no cell of this notebook", () => {
    const { controller, model } = setup([code("a", "x")])
    controller.listener.edit("tab", "scratch")
    controller.listener.edit("other#a", "scratch")
    expect(model.cells[0].getSource()).toBe("x")
  })

  it("opens the active cell again once a new session runs", () => {
    const { controller, terminal } = setup([code("a", "x")])
    terminal.session = null
    terminal.state = "starting"
    controller.update(terminal)
    expect(terminal.opens).toHaveLength(1)
    terminal.session = {}
    terminal.state = "running"
    controller.update(terminal)
    expect(terminal.opens).toHaveLength(2)
  })

  it("opens the cell after the session came up, not before", () => {
    const model = new FakeNotebook(content([code("a", "x")]))
    const notebook = new FakeWidget(model)
    const terminal = new FakeTerminal()
    terminal.session = null
    terminal.state = "starting"
    const controller = new CellNeovim({ tabId: "tab", notebook: notebook as unknown as NotebookWidgetLike, model: modelOf(model, new Signal()), terminal, language: () => ({ name: "python", file_extension: ".py" }), runAndAdvance: vi.fn(), save: vi.fn() })
    controller.start()
    expect(terminal.opens).toEqual([])
    expect(controller.getSnapshot().key).toBeNull()
    terminal.session = {}
    controller.update(terminal)
    expect(terminal.opens).toEqual([{ key: key("a"), filename: "cell.py", text: "x", cursor: "first" }])
    expect(controller.getSnapshot().key).toBe(key("a"))
  })

  it("restarts a stopped session when a cell is activated", () => {
    const { notebook, terminal } = setup([code("a", "x"), code("b", "y")])
    terminal.state = "stopped"
    terminal.session = null
    notebook.activeCellIndex = 1
    expect(terminal.restarts).toBe(1)
  })

  it("reports the shown cell and its line count", () => {
    const { controller, model, notebook } = setup([code("a", "x")])
    expect(controller.getSnapshot()).toMatchObject({ key: key("a"), lines: 1 })
    controller.listener.edit(key("a"), "x\ny\nz")
    notebook.sourceChanged(0)
    expect(controller.getSnapshot().lines).toBe(3)
    expect(model.cells[0].getSource()).toBe("x\ny\nz")
  })

  it("wipes what it opened and stops listening when disposed", () => {
    const { controller, notebook, terminal } = setup([code("a", "x"), code("b", "y")])
    notebook.activeCellIndex = 1
    controller.dispose()
    expect(terminal.closes.sort()).toEqual([key("a"), key("b")])
    expect(notebook.activeCellChanged.size).toBe(0)
    expect(notebook.stateChanged.size).toBe(0)
    expect(notebook.widgets[1].changed.size).toBe(0)
  })

  it("saves the notebook on :w, with the text of the cell", () => {
    const { controller, save, model } = setup([code("a", "x")])
    controller.listener.save(key("a"), "x = 2")
    expect(model.cells[0].getSource()).toBe("x = 2")
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
    expect(languageExtension({ name: "R", file_extension: ".r" })).toBe("r")
    expect(languageExtension({ name: "Julia" })).toBe("julia")
    expect(languageExtension({})).toBeNull()
    expect(languageExtension(null)).toBeNull()
  })
})
