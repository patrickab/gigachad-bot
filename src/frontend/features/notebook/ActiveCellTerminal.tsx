"use client"

import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react"
import { createPortal } from "react-dom"
import { TERMINAL_BOX_CLASS, TERMINAL_HOST_CLASS, cellTerminalHeight, type CellNeovimSnapshot } from "./cellNeovim"
import { CellSnapshots } from "./CellSnapshots"
import type { CellTerminal } from "./cellTerminal"
import { useActiveCellNeovim } from "./useActiveCellNeovim"

const NOTHING_SHOWN: CellNeovimSnapshot = { id: null, cell: null, rows: 1 }
const noSubscription = () => () => {}
/** Lumino skips the key bindings of an event whose target or ancestor has this attribute. */
const SUPPRESS_SHORTCUTS = "data-lm-suppress-shortcuts"

/**
 * The Neovim terminal of the active notebook cell (`useActiveCellNeovim`), inside that cell's input area in place of its CodeMirror editor, which stays
 * hidden while the terminal is there. The terminal is one element that moves from cell to cell with the session alive. It sits in the notebook's DOM, so
 * the notebook must leave its keys alone (`SUPPRESS_SHORTCUTS`), and its height follows the screen rows Neovim draws the cell's text in: the rows are what keep the cell tall enough.
 * Every other code cell shows how Neovim draws it (`CellSnapshots`).
 */
export function ActiveCellTerminal({ tabId, terminal, save }: { tabId: string; terminal: CellTerminal; save: () => Promise<void> }) {
  const controller = useActiveCellNeovim(tabId, terminal, save)
  const { id, cell, rows, reveal } = useSyncExternalStore(controller?.subscribe ?? noSubscription, controller?.getSnapshot ?? (() => NOTHING_SHOWN), () => NOTHING_SHOWN)
  const hostRef = useRef<HTMLDivElement>(null)
  const [, rerender] = useState(0)
  const parent = id === null ? null : cell?.node.querySelector<HTMLElement>(".jp-InputArea") ?? null

  // The notebook renders only the cells near the viewport (windowing): a cell that became active by keyboard or run-and-advance may get its input area a moment later
  useEffect(() => {
    if (id === null || !cell || parent) return
    const observer = new MutationObserver(() => {
      if (!cell.node.querySelector(".jp-InputArea")) return
      observer.disconnect()
      rerender((count) => count + 1)
    })
    observer.observe(cell.node, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [id, cell, parent])

  useLayoutEffect(() => {
    if (!cell || !parent) return
    cell.node.classList.add("gc-nvim-cell")
    return () => cell.node.classList.remove("gc-nvim-cell")
  }, [cell, parent])

  // The terminal needs a height before it can be fitted. Once it is on screen its row height is known and the height is set again.
  const setHeight = () => {
    if (hostRef.current) hostRef.current.style.height = `${cellTerminalHeight(rows, terminal.metrics()?.cellHeight ?? 0)}px`
  }
  useLayoutEffect(setHeight)
  useEffect(setHeight)

  // The notebook's bindings (`dd`, `a`, `b`...) match the focused element itself before the search for the attribute reaches the host, so the terminal's own input element carries it too
  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const mark = () => host.querySelectorAll("textarea").forEach((input) => input.setAttribute(SUPPRESS_SHORTCUTS, "true"))
    mark()
    const observer = new MutationObserver(mark)
    observer.observe(host, { childList: true, subtree: true })
    return () => observer.disconnect()
  }, [parent])

  // The terminal itself comes into view, not its cell: the cell holds the outputs too, and a tall output keeps the cell "visible" while the terminal at its top is cut at the edge
  useEffect(() => { if (parent && reveal !== false) hostRef.current?.scrollIntoView?.({ block: "nearest" }) }, [cell, parent, reveal])

  return (
    <>
      {parent && createPortal(
        <div ref={hostRef} {...{ [SUPPRESS_SHORTCUTS]: "true" }} data-testid="cell-neovim" className={TERMINAL_HOST_CLASS}>
          {terminal.render(TERMINAL_BOX_CLASS)}
        </div>,
        parent,
      )}
      {controller && <CellSnapshots controller={controller} terminal={terminal} />}
    </>
  )
}
