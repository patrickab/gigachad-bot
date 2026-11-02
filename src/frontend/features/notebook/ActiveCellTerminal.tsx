"use client"

import { useEffect, useLayoutEffect, useRef, useSyncExternalStore } from "react"
import { createPortal } from "react-dom"
import { TERMINAL_HOST_CLASS, cellTerminalHeight, type CellNeovimSnapshot } from "./cellNeovim"
import type { CellTerminal } from "./cellTerminal"
import { useActiveCellNeovim } from "./useActiveCellNeovim"

const NOTHING_SHOWN: CellNeovimSnapshot = { key: null, cell: null, lines: 1 }
const noSubscription = () => () => {}
/** Lumino skips the key bindings of an event whose target or ancestor has this attribute. */
const SUPPRESS_SHORTCUTS = "data-lm-suppress-shortcuts"

/**
 * The Neovim terminal of the active notebook cell (`useActiveCellNeovim`), inside that cell's input area in place of its CodeMirror editor, which stays
 * hidden while the terminal is there. The terminal is one element that moves from cell to cell with the session alive. It sits in the notebook's DOM, so
 * the notebook must leave its keys alone (`SUPPRESS_SHORTCUTS`), and its height follows the cell's lines: the rows are what keep the cell tall enough.
 */
export function ActiveCellTerminal({ tabId, terminal, save }: { tabId: string; terminal: CellTerminal; save: () => Promise<void> }) {
  const controller = useActiveCellNeovim(tabId, terminal, save)
  const { key, cell, lines } = useSyncExternalStore(controller?.subscribe ?? noSubscription, controller?.getSnapshot ?? (() => NOTHING_SHOWN), () => NOTHING_SHOWN)
  const hostRef = useRef<HTMLDivElement>(null)
  const parent = key === null ? null : cell?.node.querySelector<HTMLElement>(".jp-InputArea") ?? null

  useLayoutEffect(() => {
    if (!cell || !parent) return
    cell.node.classList.add("gc-nvim-cell")
    return () => cell.node.classList.remove("gc-nvim-cell")
  }, [cell, parent])

  // The terminal needs a height before it can be fitted. Once it is on screen its row height is known and the height is set again.
  const setHeight = () => {
    if (hostRef.current) hostRef.current.style.height = `${cellTerminalHeight(lines, terminal.cellHeight())}px`
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

  useEffect(() => { if (parent) cell?.node.scrollIntoView?.({ block: "nearest" }) }, [cell, parent])

  if (!parent) return null
  return createPortal(
    <div ref={hostRef} {...{ [SUPPRESS_SHORTCUTS]: "true" }} data-testid="cell-neovim" className={TERMINAL_HOST_CLASS}>
      {terminal.render("h-full rounded-md border border-ink-subtle bg-surface-elevated px-2 py-1")}
    </div>,
    parent,
  )
}
