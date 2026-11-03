import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render } from "@testing-library/react"
import { ActiveCellTerminal } from "../ActiveCellTerminal"
import { CellNeovim, cellTerminalHeight, type CellModelLike, type CellWidgetLike, type NotebookWidgetLike } from "../cellNeovim"
import { readCellRendering } from "../cellRender"
import { SNAPSHOT_CELL_CLASS } from "../CellSnapshots"
import type { CellCursor, CellTerminal, CellTerminalListener } from "../cellTerminal"
import type { NotebookCellContent } from "../notebookSync"
import { FakeNotebook } from "./fakeNotebook"

const controller = vi.hoisted(() => ({ current: null as unknown }))
vi.mock("../useActiveCellNeovim", () => ({ useActiveCellNeovim: () => controller.current }))

const ROW = 16
const METRICS = { fontFamily: "monospace", fontSize: 13, cellWidth: 8, cellHeight: ROW, columns: 80 }
const code = (id: string, source: string): NotebookCellContent => ({ cell_type: "code", id, source, metadata: {}, outputs: [], execution_count: null })
const markdown = (id: string, source: string): NotebookCellContent => ({ cell_type: "markdown", id, source, metadata: {} })

class Signal<Args> {
  private readonly slots = new Set<(sender: unknown, args: Args) => void>()
  connect(slot: (sender: unknown, args: Args) => void) { this.slots.add(slot) }
  disconnect(slot: (sender: unknown, args: Args) => void) { this.slots.delete(slot) }
  emit(args: Args) { for (const slot of [...this.slots]) slot(this, args) }
}

/** The notebook widget: one cell node per cell, each with the input area the real notebook builds, all of them in its scrolling node. */
class FakeWidget {
  readonly activeCellChanged = new Signal<CellWidgetLike | null>()
  readonly stateChanged = new Signal<{ name: string; newValue: unknown }>()
  readonly node = document.createElement("div")
  private readonly nodes = new Map<string, HTMLElement>()
  private index = 0

  constructor(private readonly model: FakeNotebook) {
    document.body.appendChild(this.node)
  }

  get widgets(): CellWidgetLike[] {
    return this.model.cells.map((cell) => {
      let node = this.nodes.get(cell.id)
      if (!node) {
        node = Object.assign(document.createElement("div"), { className: "jp-Cell", innerHTML: '<div class="jp-InputArea"><div class="jp-InputArea-editor"></div></div>' })
        this.nodes.set(cell.id, node)
        this.node.appendChild(node)
      }
      return { node, model: { id: cell.id, type: cell.cell_type } }
    })
  }

  get activeCellIndex() { return this.index }
  set activeCellIndex(value: number) {
    this.index = value
    this.activeCellChanged.emit(this.widgets[value] ?? null)
  }

  get activeCell() { return this.widgets[this.index] ?? null }
  setMode() {}
  scrollToItem() { return Promise.resolve() }
}

class FakeTerminal implements CellTerminal {
  state: CellTerminal["state"] = "running"
  session: object | null = {}
  readonly focuses: { id: string; cursor: CellCursor }[] = []
  listener: CellTerminalListener | null = null
  setCells() { return true }
  focusCell(id: string, cursor: CellCursor) { this.focuses.push({ id, cursor }); return true }
  focus() {}
  restart() {}
  metrics() { return METRICS }
  subscribe(listener: CellTerminalListener) { this.listener = listener; return () => {} }
  render() { return <div data-testid="xterm" /> }
}

/** A frame as the backend sends it: `lines` of `[text, style]` runs, `rows` the cell line each one shows. */
const frame = (lines: [string, number][][], styles: unknown[] = [{}], base: unknown = {}, rows?: number[]) => ({ type: "cell_render", styles, lines, base, rows })

function setup(cells: NotebookCellContent[]) {
  const model = new FakeNotebook({ cells, metadata: {}, nbformat: 4, nbformat_minor: 5 })
  const notebook = new FakeWidget(model)
  const terminal = new FakeTerminal()
  const neovim = new CellNeovim({
    notebook: notebook as unknown as NotebookWidgetLike,
    model: model as unknown as CellModelLike,
    terminal,
    language: () => ({ name: "python" }),
    run: () => {},
    runAndAdvance: () => {},
    save: () => {},
  })
  terminal.subscribe(neovim.listener)
  neovim.start()
  controller.current = neovim
  render(<ActiveCellTerminal tabId="tab" terminal={terminal} save={async () => {}} />)
  const node = (index: number) => notebook.widgets[index].node
  const snapshot = (index: number) => node(index).querySelector<HTMLElement>("[data-testid=cell-snapshot]")
  const host = (index: number) => node(index).querySelector<HTMLElement>("[data-testid=cell-neovim]")
  const draw = (id: string, value: unknown) => act(() => terminal.listener?.render(id, value))
  return { model, notebook, terminal, neovim, node, snapshot, host, draw }
}

afterEach(() => {
  cleanup()
  document.body.innerHTML = ""
  controller.current = null
})

describe("CellSnapshots", () => {
  it("shows every code cell but the active one as Neovim drew it, with its editor hidden, once Neovim sent the render", () => {
    const { node, snapshot, host, draw } = setup([code("a", "x = 1"), markdown("m", "# t"), code("b", "y = 2\nz = 3")])
    expect(host(0)).not.toBeNull()
    expect(snapshot(2)).toBeNull() // CodeMirror until Neovim drew the cell

    draw("b", frame([[["  1 ", 1], ["y = 2", 0]], [["  2 ", 1], ["z = 3", 0]]], [{}, { fg: "#a1968a" }]))
    expect(snapshot(2)?.textContent).toBe("  1 y = 2  2 z = 3")
    expect(node(2).classList.contains(SNAPSHOT_CELL_CLASS)).toBe(true)
    expect(snapshot(2)?.querySelector("span")?.style.color).toBe("rgb(161, 150, 138)")

    draw("a", frame([[["x = 1", 0]]]))
    expect(snapshot(0)).toBeNull() // the active cell keeps the terminal
    expect(node(1).querySelector("[data-testid=cell-snapshot]")).toBeNull()
  })

  it("draws markup from the file as text and drops styles that are not plain colors", () => {
    const { snapshot, draw } = setup([code("a", ""), code("b", "s")])
    const hostile = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
    draw("b", frame([[[hostile, 1], ["!", 2], ["?", 7]]], [{}, { fg: "red; background: url(javascript:alert(3))", bold: "yes", underline: "wavy" }, { bg: "#ABCDEF", italic: true, underline: "curl", sp: "#bb4f04" }]))
    const shown = snapshot(1)!
    expect(shown.textContent).toBe(`${hostile}!`)
    expect(shown.querySelector("img, script")).toBeNull()
    const [plain, styled] = shown.querySelectorAll("span")
    expect(plain.getAttribute("style")).toBeNull()
    expect(styled.style.backgroundColor).toBe("rgb(171, 205, 239)")
    expect(styled.style.fontStyle).toBe("italic")
    expect(styled.style.textDecorationLine).toBe("underline")
    expect(styled.style.textDecorationStyle).toBe("") // straight, as xterm draws Neovim's undercurl
  })

  it("takes the room the terminal takes, so the cells do not move when the terminal goes from one to the other", () => {
    const { snapshot, host, draw, neovim } = setup([code("a", "x = 1"), code("b", "y = 2\nz = 3\nw = 4")])
    draw("a", frame([[["x = 1", 0]]]))
    draw("b", frame([[["y = 2", 0]], [["z = 3", 0]], [["w = 4", 0]]]))
    const heightOf = (element: HTMLElement | null) => parseFloat(element!.style.height)
    const terminalOfA = heightOf(host(0))
    const snapshotOfB = heightOf(snapshot(1))
    expect(snapshotOfB).toBe(cellTerminalHeight(3, ROW))

    act(() => neovim.activateAt("b", 0))
    expect(heightOf(host(1))).toBe(snapshotOfB)
    expect(heightOf(snapshot(0))).toBe(terminalOfA)
  })

  it("takes the screen rows the cell took in Neovim and wraps its lines as Neovim's window does", () => {
    const { snapshot, host, draw, neovim } = setup([code("a", "x = 1"), code("b", "long")])
    draw("b", { ...frame([[["  1 ", 1], ["long", 0]]], [{}, { fg: "#a1968a" }]), height: 3, wrap: 4 })
    const heightOf = (element: HTMLElement | null) => parseFloat(element!.style.height)
    expect(heightOf(snapshot(1))).toBe(cellTerminalHeight(3, ROW))
    const line = snapshot(1)!.querySelector<HTMLElement>("[data-testid=cell-snapshot] > div > div > div")!
    expect([line.style.whiteSpace, line.style.wordBreak]).toEqual(["break-spaces", "break-all"])
    expect(parseFloat(line.style.paddingLeft)).toBe(-parseFloat(line.style.textIndent)) // a wrapped line goes on after the number column

    act(() => neovim.activateAt("b", 0))
    expect(heightOf(host(1))).toBe(cellTerminalHeight(3, ROW)) // the terminal takes the render's rows until Neovim tells its own
    act(() => neovim.listener.height("b", 4)) // typed a line more
    expect(heightOf(host(1))).toBe(cellTerminalHeight(4, ROW))
  })

  it("makes a pressed cell the active one with Neovim's cursor on the pressed line, before the notebook handles the press", () => {
    const { notebook, terminal, snapshot, host, draw } = setup([code("a", "x = 1"), markdown("m", "# t"), code("b", "y = 2\nz = 3\nw = 4")])
    draw("b", frame([[["y = 2", 0]], [["z = 3", 0]], [["w = 4", 0]]]))
    const text = snapshot(2)!.querySelector<HTMLElement>("[data-testid=cell-snapshot] > div > div")!
    vi.spyOn(text, "getBoundingClientRect").mockReturnValue({ top: 100 } as DOMRect)
    const notebookSaw = vi.fn((event: MouseEvent) => event.defaultPrevented)
    notebook.node.addEventListener("mousedown", notebookSaw)

    act(() => { text.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, button: 0, clientY: 100 + ROW * 1.5 })) })
    expect(notebook.activeCellIndex).toBe(2)
    expect(terminal.focuses.at(-1)).toEqual({ id: "b", cursor: { line: 1 } })
    expect(notebookSaw).toHaveReturnedWith(true) // the notebook's own handler skips a press whose default is prevented
    expect(host(2)).not.toBeNull()
    expect(snapshot(2)).toBeNull()
  })

  it("puts the cursor on the line below a line a language server added (`virt_lines`) when that one is pressed", () => {
    const { terminal, snapshot, draw } = setup([code("a", "x = 1"), code("b", "y = 2\nz = 3")])
    draw("b", frame([[["y = 2", 0]], [["  hint", 0]], [["z = 3", 0]]], [{}], {}, [0, 1, 1]))
    const text = () => snapshot(1)!.querySelector<HTMLElement>("[data-testid=cell-snapshot] > div > div")!
    const press = (clientY: number) => {
      vi.spyOn(text(), "getBoundingClientRect").mockReturnValue({ top: 100 } as DOMRect)
      act(() => { text().dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, button: 0, clientY })) })
    }

    press(100 + ROW * 1.5) // the hint above z = 3
    expect(terminal.focuses.at(-1)).toEqual({ id: "b", cursor: { line: 1 } })
  })

  it("shows a cell's editor again when Neovim takes its render back, the other renders stay", () => {
    const { node, snapshot, draw } = setup([code("a", ""), code("b", "y"), code("c", "w")])
    draw("b", frame([[["y", 0]]]))
    draw("c", frame([[["w", 0]]]))
    expect(snapshot(1)).not.toBeNull()

    draw("b", { type: "cell_render", id: "b" }) // the render failed, or the notebook grew past what Neovim renders
    expect(snapshot(1)).toBeNull()
    expect(node(1).classList.contains(SNAPSHOT_CELL_CLASS)).toBe(false)
    expect(snapshot(2)?.textContent).toBe("w")

    draw("c", { type: "cell_render", id: "c" })
    expect(snapshot(2)).toBeNull()
  })

  it("shows nothing but the editors without a running session, and forgets the renders of a session that ended", () => {
    const { terminal, neovim, snapshot, draw } = setup([code("a", ""), code("b", "y")])
    draw("b", frame([[["y", 0]]]))
    expect(snapshot(1)).not.toBeNull()

    act(() => {
      terminal.state = "stopped"
      terminal.session = null
      neovim.update(terminal)
    })
    expect(snapshot(1)).toBeNull()
    expect(neovim.getRenders().size).toBe(0)
  })
})

describe("readCellRendering", () => {
  it("takes only what has the shape of a render", () => {
    for (const bad of [null, "x", [], { styles: [] }, { lines: [] }, { styles: {}, lines: [] }]) expect(readCellRendering(bad)).toBeNull()
    expect(readCellRendering({ styles: [{}, { fg: "#000000", onclick: "x" }], lines: [[["a", 1], ["b", 2], ["c", -1], ["d", 0.5], [1, 0], ["e"]], "nope", []], base: { fg: "#ffffff", bg: "black" }, rows: [0, -1], height: 1.5, wrap: -2 })).toEqual({
      styles: [{}, { fg: "#000000" }],
      lines: [[["a", 1]], [], []],
      base: { fg: "#ffffff" },
      rows: [0, 1, 2], // one per line, the line's own index where none fits
      height: 3, // the line count without a valid one
      wrap: null,
    })
    expect(readCellRendering({ styles: [], lines: [[]], height: 4, wrap: 0 })).toMatchObject({ height: 4, wrap: 0 })
  })
})
