"use client"

import { useEffect, useState } from "react"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { EditorShell } from "@/features/editor/EditorShell"
import type { NeovimSelection } from "@/features/editor/NeovimEditor"
import { NotebookWithNeovim } from "@/features/editor/NotebookNeovim"
import { fileLead } from "@/features/omp/fileLead"
import { fetchNotebookDirectory } from "@/lib/backend/api"
import { extOf } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"
import { quoteSelection } from "./quote"
import { WorkbenchChat, useWorkbenchChat, type WorkbenchChatProps } from "./WorkbenchChat"

/** The fence language of lines picked in the notebook's Neovim buffer `path`: its markdown cells (`.ipynb.md`), else its code cells, named by their extension (`.ipynb.py`). */
function quoteLanguage(path: string): string {
  const extension = extOf(path)
  return extension === "md" ? "markdown" : extension === "py" ? "python" : extension
}

type NotebookWorkbenchProps = WorkbenchChatProps & { path: string; tabId: string }

/**
 * A notebook (`.ipynb`) shown by the tab `tabId`, with the workbench's omp chat drawer.
 *
 * - The chat is linked to the notebook's path, a new session starts where its kernel runs (the mounted root holding it, else its folder) and its first message names the file relative to there (`File: <path>`), so relative paths the chat links resolve against that root
 * - The notebook waits for that directory; a moved notebook keeps the last one until its own arrives
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines of the active cell into the chat draft as the notebook's lines (counted from the cell's first line), fenced in the cell's language, opening the drawer if it is closed
 * - The notebook's run and kernel controls render centered in the top bar's `center` slot, the chat's model controls into its `workbenchAgent` slot
 */
export function NotebookWorkbench(props: NotebookWorkbenchProps) {
  const { path } = props
  const [cwd, setCwd] = useState<string | null>(null)
  const [failure, setFailure] = useState<{ path: string; message: string } | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    fetchNotebookDirectory(path, controller.signal).then(
      (directory) => { setFailure(null); setCwd(directory) },
      (cause: unknown) => { if (!controller.signal.aborted) setFailure({ path, message: errorMessage(cause, "Could not find the notebook's folder") }) },
    )
    return () => controller.abort()
  }, [path, attempt])

  if (failure?.path === path && cwd === null) return <LoadFailure message={failure.message} onRetry={() => { setFailure(null); setAttempt((count) => count + 1) }} />
  if (cwd === null) return <LoadingPlaceholder />
  return <PreparedNotebookWorkbench {...props} cwd={cwd} />
}

function PreparedNotebookWorkbench({ path, tabId, cwd, ...chatProps }: NotebookWorkbenchProps & { cwd: string }) {
  const chat = useWorkbenchChat(path, cwd, fileLead(path, cwd), chatProps)
  const quote = (selection: NeovimSelection) => chat.quote(quoteSelection({ ...selection, path }, quoteLanguage(selection.path)))
  return (
    <EditorShell overlay fullscreen={false}>
      <NotebookWithNeovim path={path} tabId={tabId} onSelection={quote} />
      <WorkbenchChat state={chat} {...chatProps} />
    </EditorShell>
  )
}
