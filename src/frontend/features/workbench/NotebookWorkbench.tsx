"use client"

import { useCallback, useEffect, useEffectEvent, useLayoutEffect, useRef, useState } from "react"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { NeovimEditor, useNeovimTerminal, type NeovimSelection } from "@/features/editor/NeovimEditor"
import { TopBarPill } from "@/features/navigation/TopBar"
import { InTopBar } from "@/features/navigation/TopBarSlots"
import { NotebookDocument, useEditorPath } from "@/features/notebook/NotebookDocument"
import type { NotebookTextBridge } from "@/features/notebook/notebookTextBridge"
import { fileLead } from "@/features/omp/fileLead"
import { fetchNotebookDirectory } from "@/lib/backend/api"
import { baseName, extOf } from "@/lib/paths"
import { cn, errorMessage } from "@/lib/utils"
import { EDITOR_PAGE } from "./pages"
import { quoteSelection } from "./quote"
import { WorkbenchChat, useWorkbenchChat, type WorkbenchChatProps } from "./WorkbenchChat"
import { WorkbenchViewSwitch, usePaneLayout, useWorkbenchView } from "./WorkbenchSplit"

/** The fence language of lines picked in the notebook's Neovim buffer `path` (`<notebook>.ipynb.py`), named by its extension. */
function quoteLanguage(path: string): string {
  const extension = extOf(path)
  return extension === "py" ? "python" : extension
}

type NotebookWorkbenchProps = WorkbenchChatProps & { path: string; tabId: string }

const CENTERED_NOTE = "flex min-h-0 flex-1 items-center justify-center text-xs text-ink-faint"

/**
 * A notebook (`.ipynb`) shown by the tab `tabId`, with the workbench's omp chat drawer.
 *
 * - With Neovim on, the notebook sits right of a Neovim buffer holding it as percent text (`<notebook>.ipynb<ext>`, never on disk), split like the Markdown workbench. The notebook stays the source of truth: `:w` applies the text to it, every change of it reaches the buffer
 * - `:q` leaves the notebook alone with a "Start Neovim" button in Neovim's place, which opens a fresh buffer of the notebook as it is then
 * - The chat is linked to the notebook's path, a new session starts where its kernel runs (the mounted root holding it, else its folder) and its first message names the file relative to there (`File: <path>`), so relative paths the chat links resolve against that root
 * - The notebook waits for that directory; a moved notebook keeps the last one until its own arrives
 * - Visual-mode `<leader>a` in Neovim quotes the selected lines into the chat draft as the notebook's lines, fenced in the buffer's language, opening the drawer if it is closed
 * - The notebook's run and kernel controls render centered in the top bar's `center` slot, the chat's model controls into its `workbenchAgent` slot, the view switch into `workbenchStatus`
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
  // The bridge with the session it was announced for: the previous notebook's bridge, gone one render after a new notebook opened, never starts the new session
  const [held, setHeld] = useState<{ key: string; bridge: NotebookTextBridge } | null>(null)
  const bridge = held?.key === sessionKey ? held.bridge : null
  // The notebook's text Neovim starts or restarts from: the bridge's projection, kept current by every save and change
  const [text, setText] = useState("")
  // The session `:q` closed: another notebook opened in the tab is a new key, so it opens Neovim again
  const [closedKey, setClosedKey] = useState<string | null>(null)
  const closed = closedKey === sessionKey
  // Read by the announce: a layout effect runs before the notebook's passive effects announce a bridge in the same commit
  const keyRef = useRef(sessionKey)
  useLayoutEffect(() => { keyRef.current = sessionKey }, [sessionKey])
  const takeBridge = useCallback((next: NotebookTextBridge | null) => {
    setHeld(next ? { key: keyRef.current, bridge: next } : null)
    if (next) setText(next.text())
  }, [])

  const virtualPath = bridge ? editor.path + bridge.extension : null
  const session = useNeovimTerminal(
    neovim.enabled && !closed && virtualPath !== null ? { key: sessionKey, path: virtualPath, filename: baseName(virtualPath), content: text, notebook: true } : null,
    async (edited) => {
      if (!bridge) throw new Error("The notebook is not loaded")
      const saved = await bridge.save(edited)
      setText(saved)
      return saved
    },
    () => setClosedKey(sessionKey),
    {
      askKeymap: true,
      onSelection: (selection: NeovimSelection) => chat.quote(quoteSelection({ ...selection, path }, quoteLanguage(selection.path))),
      onRunCell: (_, line) => bridge?.runAt(line),
    },
  )

  const pushText = useEffectEvent((next: string) => {
    setText(next)
    if (virtualPath !== null) session.setContent(next, virtualPath)
  })
  useEffect(() => bridge?.subscribe((next) => pushText(next)), [bridge])

  const start = () => {
    if (bridge) setText(bridge.text())
    setClosedKey(null)
  }

  const [workbenchView, setWorkbenchView] = useWorkbenchView()
  const view = neovim.enabled ? workbenchView : "render"
  const splitRef = useRef<HTMLDivElement>(null)
  const layout = usePaneLayout(splitRef, view)

  // Like the Markdown workbench: Neovim and the notebook share the width, a hidden pane stays mounted, so Neovim keeps its session and the notebook its kernel
  return (
    <div className="flex h-full">
      <div className={cn("flex min-w-0 flex-1", neovim.enabled && "px-3")}>
        <div ref={splitRef} className="flex min-w-0 flex-1">
          {neovim.enabled && (
            <div className={cn("flex min-w-0 flex-1 flex-col items-center", view === "render" && "hidden")}>
              <div className="flex flex-col" style={{ marginTop: layout.top, width: layout.width, height: layout.height }}>
                {closed ? <NeovimClosed onStart={start} />
                  : bridge === null ? <div className={CENTERED_NOTE}>Loading…</div>
                  : <NeovimEditor terminal={session.terminal} status={session.status} onRestart={session.restart} className={EDITOR_PAGE} />}
              </div>
            </div>
          )}
          <div className={cn("relative min-w-0 flex-1", view === "editor" && "hidden")}>
            <NotebookDocument editor={editor} tabId={tabId} onTextBridge={takeBridge} />
          </div>
        </div>
      </div>
      <WorkbenchChat state={chat} {...chatProps} />
      {neovim.enabled && (
        <InTopBar name="workbenchStatus">
          <TopBarPill>
            <WorkbenchViewSwitch view={workbenchView} onChange={setWorkbenchView} />
          </TopBarPill>
        </InTopBar>
      )}
    </div>
  )
}

/** Neovim's page after `:q`, the look of `NeovimEditor`'s stopped state. */
function NeovimClosed({ onStart }: { onStart: () => void }) {
  return (
    <div className={cn("flex min-h-0 flex-1 items-center justify-center text-xs", EDITOR_PAGE)}>
      <button onClick={onStart} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
        Start Neovim
      </button>
    </div>
  )
}
