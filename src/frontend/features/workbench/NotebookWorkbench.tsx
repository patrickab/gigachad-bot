"use client"

import { useEffect, useEffectEvent, useState, useSyncExternalStore } from "react"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, NeovimStopped, useNeovimTerminal, type NeovimSelection } from "@/features/editor/NeovimEditor"
import { NotebookDocument, useEditorPath } from "@/features/notebook/NotebookDocument"
import { createNotebookTextStore } from "@/features/notebook/notebookTextStore"
import { fileLead } from "@/features/omp/fileLead"
import { fetchNotebookDirectory } from "@/lib/backend/api"
import { baseName, extOf } from "@/lib/paths"
import { cn, errorMessage } from "@/lib/utils"
import { EDITOR_PAGE } from "./pages"
import { quoteSelection } from "./quote"
import { WorkbenchChat, useWorkbenchChat, type WorkbenchChatProps } from "./WorkbenchChat"
import { EditorNote, WorkbenchFrame } from "./WorkbenchFrame"

/** The fence language of lines picked in the notebook's Neovim buffer `path` (`<notebook>.ipynb.py`), named by its extension. */
function quoteLanguage(path: string): string {
  const extension = extOf(path)
  return extension === "py" ? "python" : extension
}

type NotebookWorkbenchProps = WorkbenchChatProps & { path: string; tabId: string }

/**
 * A notebook (`.ipynb`) shown by the tab `tabId`, with the workbench's omp chat drawer.
 *
 * - With Neovim on, a buffer holds the notebook as percent text (`<notebook>.ipynb<ext>`, never on disk). The notebook stays the source of truth: `:w` applies the text, every change of the notebook reaches the buffer
 * - `:q` leaves a "Start Neovim" button that opens a fresh buffer of the notebook as it is then
 * - The chat starts where the notebook's kernel runs (the mount root holding it, else its folder), so relative links resolve there
 * - Visual-mode `<leader>a` quotes the selected lines into the chat draft
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
  const neovim = useNeovimSetting()
  const chat = useWorkbenchChat(path, cwd, fileLead(path, cwd), chatProps)
  const editor = useEditorPath(path)
  const sessionKey = `${tabId}:${editor.generation}`
  // One text store per session, so another notebook opened in the tab never starts its session with the previous notebook's text
  const [held, setHeld] = useState(() => ({ key: sessionKey, store: createNotebookTextStore() }))
  if (held.key !== sessionKey) setHeld({ key: sessionKey, store: createNotebookTextStore() })
  const textStore = held.store
  const textState = useSyncExternalStore(textStore.subscribe, textStore.state, textStore.state)
  // The session `:q` closed: another notebook opened in the tab is a new key, so it opens Neovim again
  const [closedKey, setClosedKey] = useState<string | null>(null)
  const closed = closedKey === sessionKey
  // The render that renews the store still reads the previous one, it never opens the new key's session on that text
  const open = held.key === sessionKey && neovim.enabled && !closed && textState !== null

  const buffer = open ? { key: sessionKey, path: editor.path + textState.extension, filename: baseName(editor.path) + textState.extension, content: textState.text, notebook: true } : null
  const session = useNeovimTerminal(
    buffer,
    (edited) => textStore.save(edited),
    () => setClosedKey(sessionKey),
    {
      askKeymap: true,
      onSelection: (selection: NeovimSelection) => chat.quote(quoteSelection({ ...selection, path }, quoteLanguage(selection.path))),
      onRunCell: (_, line) => textStore.runAt(line),
    },
  )

  const push = useEffectEvent((text: string) => {
    if (buffer) session.setContent(text, buffer.path)
  })
  // Followed while bound, Neovim open or not, so a session opens on the current text
  const bound = textState !== null
  useEffect(() => (bound ? textStore.follow((text) => push(text)) : undefined), [textStore, bound])

  const neovimPage = closed
    ? <div className={cn("relative min-h-0 flex-1", EDITOR_PAGE)}><NeovimStopped action="Start Neovim" onAction={() => setClosedKey(null)} /></div>
    : textState === null ? <EditorNote>Loading…</EditorNote>
    : <NeovimEditor terminal={session.terminal} status={session.status} onRestart={session.restart} className={EDITOR_PAGE} />

  return (
    <WorkbenchFrame
      editor={neovim.enabled ? neovimPage : null}
      renderPane={() => <NotebookDocument editor={editor} tabId={tabId} textStore={textStore} />}
      chat={<WorkbenchChat state={chat} {...chatProps} />}
    />
  )
}
