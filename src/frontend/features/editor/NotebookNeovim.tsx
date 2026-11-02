"use client"

import { useCallback, useMemo, useRef, useState } from "react"
import { NotebookDocument } from "@/features/notebook/NotebookDocument"
import type { CellTerminal, CellTerminalListener } from "@/features/notebook/cellTerminal"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal } from "./NeovimEditor"

/**
 * The Neovim session of the notebook tab `tabId`, as the terminal its cells are edited in. The session starts with the tab and ends with it, in the directory of
 * the notebook at `path`, and has the cells as its only buffers. Null while Neovim is off for this device or after `:q`: the notebook edits cells in CodeMirror then.
 */
function useNotebookCellTerminal(tabId: string, path: string): CellTerminal | null {
  const { enabled } = useNeovimSetting()
  const [quit, setQuit] = useState(false)
  const listener = useRef<CellTerminalListener | null>(null)
  const subscribe = useCallback((next: CellTerminalListener) => {
    listener.current = next
    return () => { if (listener.current === next) listener.current = null }
  }, [])

  const session = useNeovimTerminal(
    // A cell session has no document: its filename and content are never sent
    enabled && !quit ? { key: tabId, path, filename: "", content: "" } : null,
    // Cell buffers have no file to save, `:w` arrives as `onCellSave`
    async (text) => text,
    () => setQuit(true),
    {
      cellMode: true,
      onCellEdit: (key, text) => listener.current?.edit(key, text),
      onCellRun: (key, text) => listener.current?.run(key, text),
      onCellExit: (key, direction, text) => listener.current?.exit(key, direction, text),
      onCellSave: (key, text) => listener.current?.save(key, text),
    },
  )

  const { terminal, status, restart, openCell, setCell, closeCell } = session
  return useMemo(() => {
    if (!enabled || quit) return null
    return {
      state: status.state === "stopped" ? "stopped" : terminal ? "running" : "starting",
      session: status.state === "running" ? terminal : null,
      open: openCell,
      set: setCell,
      close: closeCell,
      focus: () => terminal?.focus(),
      restart,
      cellHeight: () => terminal?.cellHeight() ?? 0,
      subscribe,
      render: (className) => <NeovimEditor terminal={terminal} status={status} onRestart={restart} className={className} fitHeightAtOnce />,
    }
  }, [enabled, quit, status, terminal, restart, openCell, setCell, closeCell, subscribe])
}

/** A notebook with its cells edited in Neovim when that is on. */
export function NotebookWithNeovim({ path, tabId }: { path: string; tabId: string }) {
  const cellTerminal = useNotebookCellTerminal(tabId, path)
  return <NotebookDocument path={path} tabId={tabId} cellTerminal={cellTerminal} />
}
