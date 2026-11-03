import { memo, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useSyncExternalStore, type CSSProperties } from "react"
import { createPortal } from "react-dom"
import { cn } from "@/lib/utils"
import { TERMINAL_BOX_CLASS, TERMINAL_HOST_CLASS, cellTerminalHeight, type CellNeovim, type CellNeovimSnapshot, type CellWidgetLike } from "./cellNeovim"
import { runStyle, type CellRendering } from "./cellRender"
import type { CellTerminal, CellTextMetrics } from "./cellTerminal"

/** The class of a cell that shows its render, its CodeMirror editor stays hidden meanwhile. */
export const SNAPSHOT_CELL_CLASS = "gc-nvim-snapshot-cell"

const NO_CELL: CellNeovimSnapshot = { id: null, cell: null, rows: 1 }
const NO_RENDERS: ReadonlyMap<string, CellRendering> = new Map()
/** Where DOM changes never add or remove a cell's input area: the terminal, the renders and the outputs. */
const IGNORED_MUTATIONS = `.${TERMINAL_HOST_CLASS}, .gc-nvim-snapshot, .jp-OutputArea`

/** The width the browser gives a character of the terminal's font, by font. xterm puts each one in a cell of its own width (whole device pixels). */
const charWidths = new Map<string, number>()

function charWidth(fontSize: number, fontFamily: string): number {
  const font = `${fontSize}px ${fontFamily}`
  const known = charWidths.get(font)
  if (known !== undefined) return known
  const probe = document.createElement("span")
  probe.textContent = "M".repeat(32)
  probe.style.cssText = "position:absolute;visibility:hidden;white-space:pre"
  probe.style.font = font
  document.body.appendChild(probe)
  const width = probe.getBoundingClientRect().width / 32
  probe.remove()
  // Nothing measured (no layout, or the font is not there yet) is not kept
  if (width > 0) charWidths.set(font, width)
  return width
}

/**
 * How Neovim draws the code cell `cell` (`rendering`), in place of its editor: the same box, font, cell size and height (the screen rows the cell takes in
 * Neovim) as the terminal would have in that cell, so that the cell takes the same room either way. A window that wraps lines wraps them here too, a
 * wrapped line going on after the number column. The text is text nodes with checked colors (`readCellRendering`). A press on a line makes the cell
 * the active one with Neovim's cursor on that line: before the notebook's own handler, which would take the keys from the terminal.
 */
const CellSnapshot = memo(function CellSnapshot({ cellKey, cell, rendering, metrics, controller }: {
  cellKey: string
  cell: CellWidgetLike
  rendering: CellRendering
  metrics: CellTextMetrics
  controller: CellNeovim
}) {
  const boxRef = useRef<HTMLDivElement>(null)
  const textRef = useRef<HTMLDivElement>(null)
  const { fontFamily, fontSize, cellWidth, cellHeight, columns } = metrics
  const { wrap } = rendering

  useLayoutEffect(() => {
    cell.node.classList.add(SNAPSHOT_CELL_CLASS)
    return () => cell.node.classList.remove(SNAPSHOT_CELL_CLASS)
  }, [cell])

  useEffect(() => {
    const box = boxRef.current
    if (!box) return
    // The drawn line under the pointer (the last one below the text): a row each, or as many rows as a wrapped line takes
    const drawnLine = (y: number) => {
      const lines = [...(textRef.current?.children ?? [])]
      const last = Math.max(0, lines.length - 1)
      if (wrap === null) return Math.min(Math.max(0, Math.floor((y - (textRef.current?.getBoundingClientRect().top ?? 0)) / cellHeight)), last)
      const under = lines.findIndex((line) => y < line.getBoundingClientRect().bottom)
      return under < 0 ? last : under
    }
    const press = (event: MouseEvent) => {
      if (event.button !== 0 || event.shiftKey) return
      event.preventDefault()
      controller.activateAt(cellKey, rendering.rows[drawnLine(event.clientY)] ?? 0)
    }
    box.addEventListener("mousedown", press)
    return () => box.removeEventListener("mousedown", press)
  }, [controller, cellKey, cellHeight, rendering, wrap])

  const letterSpacing = useMemo(() => {
    const natural = charWidth(fontSize, fontFamily)
    return natural > 0 ? cellWidth - natural : 0
  }, [fontSize, fontFamily, cellWidth])

  // Neovim breaks a wrapped line at the last column, whatever the character, and goes on at the `wrap` column
  const wrapped: CSSProperties | null = wrap === null ? null : { whiteSpace: "break-spaces", wordBreak: "break-all", paddingLeft: wrap * cellWidth, textIndent: -wrap * cellWidth }
  return (
    <div ref={boxRef} data-testid="cell-snapshot" className="gc-nvim-snapshot" style={{ height: cellTerminalHeight(rendering.height, cellHeight) }}>
      <div className={cn("relative cursor-text overflow-hidden text-ink", TERMINAL_BOX_CLASS)} style={{ color: rendering.base.fg, backgroundColor: rendering.base.bg }}>
        <div ref={textRef} style={{ width: columns * cellWidth, overflow: "hidden", fontFamily, fontSize, lineHeight: `${cellHeight}px`, letterSpacing, whiteSpace: "pre" }}>
          {rendering.lines.map((line, index) => (
            <div key={index} style={wrapped ?? { height: cellHeight }}>
              {line.map(([text, style], run) => <span key={run} style={runStyle(rendering.styles[style])}>{text}</span>)}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
})

/**
 * Every code cell but the one with the terminal shows how Neovim draws it, once Neovim sent that (CodeMirror until then, and without a running session).
 * The windowed notebook builds a cell's input area when the cell first comes near the viewport: the cells are looked at again when the notebook's DOM changes.
 */
export function CellSnapshots({ controller, terminal }: { controller: CellNeovim; terminal: CellTerminal }) {
  const { id: active } = useSyncExternalStore(controller.subscribe, controller.getSnapshot, () => NO_CELL)
  const renders = useSyncExternalStore(controller.subscribeRenders, controller.getRenders, () => NO_RENDERS)
  const [, recount] = useReducer((count: number) => count + 1, 0)

  useEffect(() => {
    let frame = 0
    // Only the notebook's cells coming and going count: the terminal's repaints, the renders themselves and streaming outputs change no input area
    const observer = new MutationObserver((records) => {
      if (frame || records.every(({ target }) => target instanceof Element && target.closest(IGNORED_MUTATIONS))) return
      frame = requestAnimationFrame(() => {
        frame = 0
        recount()
      })
    })
    observer.observe(controller.notebookNode, { childList: true, subtree: true })
    return () => {
      observer.disconnect()
      cancelAnimationFrame(frame)
    }
  }, [controller])

  const metrics = terminal.metrics()
  if (terminal.state !== "running" || !metrics || renders.size === 0) return null
  return controller.codeCellWidgets().map(({ key, cell }) => {
    const rendering = renders.get(key)
    const area = key === active || !rendering ? null : cell.node.querySelector<HTMLElement>(".jp-InputArea")
    if (!area || !rendering) return null
    return createPortal(<CellSnapshot cellKey={key} cell={cell} rendering={rendering} metrics={metrics} controller={controller} />, area, key)
  })
}
