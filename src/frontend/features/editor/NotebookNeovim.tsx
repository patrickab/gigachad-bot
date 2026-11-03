"use client"

import { useCallback, useMemo, useRef, useState } from "react"
import { NotebookDocument, useEditorPath, type EditorPath } from "@/features/notebook/NotebookDocument"
import type { CellTerminal, CellTerminalListener } from "@/features/notebook/cellTerminal"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal, type NeovimSelection } from "./NeovimEditor"

/**
 * The Neovim session of the notebook tab `tabId` showing `editor`, as the terminal its cells are edited in. The session starts with the tab and ends with it, in
 * the directory of the notebook, and holds the notebook's code cells as one buffer, its markdown cells as another. Another file opened in the tab is another session (`generation`), a
 * rename keeps it and names its buffer after the new file. Null while Neovim is off for this device or after `:q`: the notebook edits cells in CodeMirror then.
 * With `onSelection` visual-mode `<leader>a` reports the selected lines of the buffer it is in (`<notebook>.ipynb.py` or `.ipynb.md`).
 */
function useNotebookCellTerminal(tabId: string, editor: EditorPath, onSelection?: (selection: NeovimSelection) => void): CellTerminal | null {
  const { enabled } = useNeovimSetting()
  const [quit, setQuit] = useState(false)
  const listener = useRef<CellTerminalListener | null>(null)
  const subscribe = useCallback((next: CellTerminalListener) => {
    listener.current = next
    return () => { if (listener.current === next) listener.current = null }
  }, [])

  const session = useNeovimTerminal(
    // A cell session has no document: its filename and content are never sent
    enabled && !quit ? { key: `${tabId}:${editor.generation}`, path: editor.path, filename: "", content: "" } : null,
    // The notebook buffer has no file to save, `:w` arrives as `onCellSave`
    async (text) => text,
    () => setQuit(true),
    {
      cellMode: true,
      askKeymap: onSelection !== undefined,
      onSelection,
      onCellEdit: (id, text) => listener.current?.edit(id, text),
      onCellFocus: (id) => listener.current?.focus(id),
      onCellRun: (id, text, advance) => listener.current?.run(id, text, advance),
      onCellLeave: (id) => listener.current?.leave(id),
      onCellExit: (id, direction) => listener.current?.exit(id, direction),
      onCellSave: () => listener.current?.save(),
      onCellRender: (id, rendering) => listener.current?.render(id, rendering),
      onCellHeight: (id, rows) => listener.current?.height(id, rows),
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
      metrics: () => terminal?.metrics() ?? null,
      subscribe,
      render: (className) => <NeovimEditor terminal={terminal} status={status} onRestart={restart} className={className} fitHeightAtOnce />,
    }
  }, [enabled, quit, status, terminal, restart, setCells, focusCell, subscribe])
}

/** A notebook with its cells edited in Neovim when that is on. `onSelection` gets the lines visual-mode `<leader>a` picks in Neovim. */
export function NotebookWithNeovim({ path, tabId, onSelection }: { path: string; tabId: string; onSelection?: (selection: NeovimSelection) => void }) {
  const editor = useEditorPath(path)
  const cellTerminal = useNotebookCellTerminal(tabId, editor, onSelection)
  return <NotebookDocument editor={editor} tabId={tabId} cellTerminal={cellTerminal} />
}
