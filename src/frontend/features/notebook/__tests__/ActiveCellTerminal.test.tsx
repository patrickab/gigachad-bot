import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render } from "@testing-library/react"
import { useEffect, useRef } from "react"
import { CommandRegistry } from "@lumino/commands"
import { ActiveCellTerminal } from "../ActiveCellTerminal"
import type { CellNeovimSnapshot, CellWidgetLike } from "../cellNeovim"
import type { CellTerminal } from "../cellTerminal"

const controller = vi.hoisted(() => ({ current: null as unknown }))
vi.mock("../useActiveCellNeovim", () => ({ useActiveCellNeovim: () => controller.current }))

/** What xterm does to its host: it creates the input element once it is open, after the host is in the page. */
function Xterm({ lateInput }: { lateInput: HTMLTextAreaElement }) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => { ref.current?.appendChild(lateInput) }, [lateInput])
  return <div ref={ref} />
}

function fakeTerminal(input: HTMLTextAreaElement): CellTerminal {
  return {
    state: "running",
    session: {},
    setCells: () => true,
    focusCell: () => true,
    focus: () => {},
    restart: () => {},
    cellHeight: () => 16,
    subscribe: () => () => {},
    render: () => <Xterm lateInput={input} />,
  }
}

/** A cell as the notebook builds it: the editor sits in an input area inside the cell's node. */
function cellWidget(): CellWidgetLike {
  const node = document.createElement("div")
  node.className = "jp-Cell"
  node.innerHTML = '<div class="jp-InputArea"><div class="jp-InputArea-editor"></div></div>'
  document.body.appendChild(node)
  return { node } as unknown as CellWidgetLike
}

function showing(snapshot: CellNeovimSnapshot) {
  let current = snapshot
  const listeners = new Set<() => void>()
  controller.current = {
    subscribe: (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener) } },
    getSnapshot: () => current,
  }
  return (next: CellNeovimSnapshot) => { current = next; act(() => listeners.forEach((listener) => listener())) }
}

afterEach(() => {
  cleanup()
  document.body.innerHTML = ""
  controller.current = null
})

describe("ActiveCellTerminal", () => {
  it("keeps the keys typed in the terminal away from the notebook's shortcuts", () => {
    // The notebook's command mode binding for `d`: it matches the focused element, whatever that is
    const commands = new CommandRegistry()
    const deleteCell = vi.fn()
    commands.addCommand("delete-cell", { execute: deleteCell })
    commands.addKeyBinding({ keys: ["D"], selector: ".jp-Notebook.jp-mod-commandMode :focus", command: "delete-cell" })
    const press = (target: HTMLElement) => target.dispatchEvent(new KeyboardEvent("keydown", { key: "d", keyCode: 68, bubbles: true, cancelable: true }))

    const notebook = document.createElement("div")
    notebook.className = "jp-Notebook jp-mod-commandMode"
    notebook.addEventListener("keydown", (event) => commands.processKeydownEvent(event))
    document.body.appendChild(notebook)
    const cell = cellWidget()
    notebook.appendChild(cell.node)

    // A text field of the notebook itself still gets the shortcut, so the harness sees a shortcut when there is one
    const other = document.createElement("textarea")
    notebook.appendChild(other)
    other.focus()
    press(other)
    expect(deleteCell).toHaveBeenCalledTimes(1)

    const input = document.createElement("textarea")
    showing({ id: "a", cell, lines: 2 })
    render(<ActiveCellTerminal tabId="tab" terminal={fakeTerminal(input)} save={async () => {}} />)
    expect(input.isConnected).toBe(true)
    input.focus()
    expect(document.activeElement).toBe(input)
    press(input)
    press(input)
    expect(deleteCell).toHaveBeenCalledTimes(1)
  })

  it("marks the input of a terminal that appears in another cell as well", () => {
    const first = cellWidget()
    const second = cellWidget()
    const input = document.createElement("textarea")
    const show = showing({ id: "a", cell: first, lines: 1 })
    render(<ActiveCellTerminal tabId="tab" terminal={fakeTerminal(input)} save={async () => {}} />)
    expect(first.node.querySelector(".jp-InputArea [data-testid=cell-neovim]")).not.toBeNull()

    show({ id: "b", cell: second, lines: 1 })
    expect(first.node.querySelector("[data-testid=cell-neovim]")).toBeNull()
    const host = second.node.querySelector(".jp-InputArea [data-testid=cell-neovim]")
    expect(host).not.toBeNull()
    expect(host?.hasAttribute("data-lm-suppress-shortcuts")).toBe(true)
    expect(host?.querySelector("textarea")?.hasAttribute("data-lm-suppress-shortcuts")).toBe(true)
  })

  it("hides the cell's editor while the terminal is in it, and gives it back", () => {
    const cell = cellWidget()
    const show = showing({ id: "a", cell, lines: 3 })
    const { unmount } = render(<ActiveCellTerminal tabId="tab" terminal={fakeTerminal(document.createElement("textarea"))} save={async () => {}} />)
    expect(cell.node.classList.contains("gc-nvim-cell")).toBe(true)

    show({ id: null, cell: null, lines: 1 })
    expect(cell.node.classList.contains("gc-nvim-cell")).toBe(false)
    expect(cell.node.querySelector("[data-testid=cell-neovim]")).toBeNull()

    show({ id: "a", cell, lines: 3 })
    expect(cell.node.classList.contains("gc-nvim-cell")).toBe(true)
    unmount()
    expect(cell.node.classList.contains("gc-nvim-cell")).toBe(false)
  })

  it("attaches to an active cell whose input area the windowed notebook renders later", async () => {
    const cell = cellWidget()
    cell.node.innerHTML = ""
    showing({ id: "a", cell, lines: 1 })
    render(<ActiveCellTerminal tabId="tab" terminal={fakeTerminal(document.createElement("textarea"))} save={async () => {}} />)
    expect(cell.node.querySelector("[data-testid=cell-neovim]")).toBeNull()

    await act(async () => {
      cell.node.innerHTML = '<div class="jp-InputArea"><div class="jp-InputArea-editor"></div></div>'
      await Promise.resolve()
    })

    expect(cell.node.querySelector(".jp-InputArea [data-testid=cell-neovim]")).not.toBeNull()
  })

  it("makes the terminal as tall as the cell's text needs", () => {
    const cell = cellWidget()
    const show = showing({ id: "a", cell, lines: 2 })
    render(<ActiveCellTerminal tabId="tab" terminal={fakeTerminal(document.createElement("textarea"))} save={async () => {}} />)
    const host = () => cell.node.querySelector<HTMLElement>("[data-testid=cell-neovim]")!
    const short = parseFloat(host().style.height)
    show({ id: "a", cell, lines: 6 })
    expect(parseFloat(host().style.height)).toBeGreaterThanOrEqual(short + 4 * 16)
  })
})
