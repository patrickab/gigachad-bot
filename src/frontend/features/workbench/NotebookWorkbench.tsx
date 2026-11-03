"use client"

import { EditorShell } from "@/features/editor/EditorShell"
import type { NeovimSelection } from "@/features/editor/NeovimEditor"
import { NotebookWithNeovim } from "@/features/editor/NotebookNeovim"
import { fileLead } from "@/features/omp/fileLead"
import { dirName, extOf } from "@/lib/paths"
import { quoteSelection } from "./quote"
import { WorkbenchChat, useWorkbenchChat, type WorkbenchChatProps } from "./WorkbenchChat"

/** The fence language of lines picked in the notebook's Neovim buffer `path`: its markdown cells (`.ipynb.md`), else its code cells, named by their extension (`.ipynb.py`). */
function quoteLanguage(path: string): string {
  const extension = extOf(path)
  return extension === "md" ? "markdown" : extension === "py" ? "python" : extension
}

/**
 * A notebook (`.ipynb`) shown by the tab `tabId`, with the workbench's omp chat drawer.
 *
 * - The chat is linked to the notebook's path, a new session starts in the notebook's directory and its first message names the file (`File: <name>`)
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines of the active cell into the chat draft as the notebook's lines (counted from the cell's first line), fenced in the cell's language, opening the drawer if it is closed
 * - The notebook's run and kernel controls and the chat's model controls render into the top bar's `workbenchStatus` and `workbenchAgent` slots
 */
export function NotebookWorkbench({ path, tabId, ...chatProps }: WorkbenchChatProps & { path: string; tabId: string }) {
  const cwd = dirName(path)
  const chat = useWorkbenchChat(path, cwd, fileLead(path, cwd), chatProps)
  const quote = (selection: NeovimSelection) => chat.quote(quoteSelection({ ...selection, path }, quoteLanguage(selection.path)))
  return (
    <EditorShell overlay fullscreen={false}>
      <NotebookWithNeovim path={path} tabId={tabId} onSelection={quote} />
      <WorkbenchChat state={chat} {...chatProps} />
    </EditorShell>
  )
}
