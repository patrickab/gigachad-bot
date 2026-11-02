"use client"

import { useCallback, useMemo, useRef, useState } from "react"
import { NotebookDocument } from "@/features/notebook/NotebookDocument"
import type { CellTerminal, CellTerminalListener } from "@/features/notebook/cellTerminal"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal } from "./NeovimEditor"

/**
 * The Neovim session of the notebook tab `tabId`, as the terminal its cells are edited in. The session starts with the tab and ends with it, in the directory of
 * the notebook at `path`, and holds the notebook's code cells as its one buffer. Null while Neovim is off for this device or after `:q`: the notebook edits cells in CodeMirror then.
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
    // The notebook buffer has no file to save, `:w` arrives as `onCellSave`
    async (text) => text,
    () => setQuit(true),
    {
      cellMode: true,
      onCellEdit: (id, text) => listener.current?.edit(id, text),
      onCellFocus: (id) => listener.current?.focus(id),
      onCellRun: (id, text) => listener.current?.run(id, text),
      onCellSave: () => listener.current?.save(),
    },
  )

  const { terminal, status, restart, setCells, focusCell } = session
  return useMemo(() => {
    if (!enabled || quit) return null
    return {
      state: status.state === "stopped" ? "stopped" : terminal ? "running" : "starting",
      session: status.state === "running" ? terminal : null,
      setCells,
      focusCell,
      focus: () => terminal?.focus(),
      restart,
      cellHeight: () => terminal?.cellHeight() ?? 0,
      subscribe,
      render: (className) => <NeovimEditor terminal={terminal} status={status} onRestart={restart} className={className} fitHeightAtOnce />,
    }
  }, [enabled, quit, status, terminal, restart, setCells, focusCell, subscribe])
}

/** A notebook with its cells edited in Neovim when that is on. */
export function NotebookWithNeovim({ path, tabId }: { path: string; tabId: string }) {
  const cellTerminal = useNotebookCellTerminal(tabId, path)
  return <NotebookDocument path={path} tabId={tabId} cellTerminal={cellTerminal} />
}
