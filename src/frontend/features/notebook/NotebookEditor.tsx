"use client"

import { JupyterReactTheme, Notebook } from "@datalayer/jupyter-react"
import type { INotebookContent } from "@jupyterlab/nbformat"
import type { NotebookModel } from "@jupyterlab/notebook"
import type { ServiceManager, Session } from "@jupyterlab/services"
import { useCallback, useEffect, useRef, useState } from "react"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { ErrorText } from "@/components/ui/ErrorText"
import { useThemeName } from "@/hooks/useThemeName"
import { PANE_TOP_PX } from "@/features/workbench/WorkbenchSplit"
import { appMarkdownFactory } from "./markdownRenderer"
import { ActiveCellTerminal } from "./ActiveCellTerminal"
import type { CellTerminal } from "./cellTerminal"
import { NotebookToolbar } from "./NotebookToolbar"
import { createNotebookServiceManager } from "./serviceManager"
import { NOTEBOOK_EDITOR_EXTENSIONS } from "./syntaxHighlight"
import { useNotebookFile } from "./useNotebookFile"
import type { SharedNotebookLike } from "./notebookSync"
import { useNotebookKernel } from "./useNotebookKernel"
import "./notebook-theme.css"

const RENDERERS = [appMarkdownFactory]
/** The notebook runs under the top bar: its cells start where the markdown preview's page does (see `notebook-theme.css`). */
const NOTEBOOK_TOP = { "--gc-nb-top": `${PANE_TOP_PX}px` } as React.CSSProperties

/** Ctrl or Cmd+S inside `root` saves through the file API: the library's own save command talks to a Jupyter contents API this app does not expose. */
function useSaveShortcut(root: React.RefObject<HTMLElement | null>, save: () => Promise<void>) {
  const latest = useRef(save)
  useEffect(() => { latest.current = save })
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== "s" || !(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return
      if (!(event.target instanceof Node) || !root.current?.contains(event.target)) return
      event.preventDefault()
      event.stopPropagation()
      void latest.current()
    }
    window.addEventListener("keydown", onKeyDown, true)
    return () => window.removeEventListener("keydown", onKeyDown, true)
  }, [root])
}

/** One service manager per editor. It polls the proxy, so it is disposed when the editor goes, unless React's development remount brings the editor straight back. */
function useNotebookServiceManager(): ServiceManager {
  const [manager] = useState(() => createNotebookServiceManager())
  const mounted = useRef(false)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      setTimeout(() => { if (!mounted.current) manager.dispose() }, 0)
    }
  }, [manager])
  return manager
}

/** One notebook of the file at `path`, shown by the tab `tabId`: JupyterLab's notebook panel on this app's file, kernel and theme. With a `cellTerminal` the active cell is edited in Neovim. */
export function NotebookEditor({ path, tabId, cellTerminal }: { path: string; tabId: string; cellTerminal: CellTerminal | null }) {
  const file = useNotebookFile(path)
  const kernel = useNotebookKernel(path, file.initial !== null)
  const theme = useThemeName()
  const rootRef = useRef<HTMLDivElement>(null)
  const serviceManager = useNotebookServiceManager()
  useSaveShortcut(rootRef, file.save)

  // The library reads `nbformat` once per model: `initial` never changes, later file changes reach the open model through `bindModel`.
  const initial = file.initial as unknown as INotebookContent | null

  const bindRef = useRef(file.bindModel)
  useEffect(() => { bindRef.current = file.bindModel })
  const modelRef = useRef<NotebookModel | null>(null)
  const onModel = useCallback((model: NotebookModel | null) => {
    modelRef.current = model
    bindRef.current(model ? model.sharedModel as unknown as SharedNotebookLike : null)
  }, [])

  const sessionRef = useRef<Session.ISessionConnection | undefined>(undefined)
  const onSession = useCallback((session: Session.ISessionConnection | undefined) => { sessionRef.current = session }, [])

  const { select, kernelName } = kernel
  const startKernelOf = useCallback(async (name: string) => {
    // The backend replaces the kernel under the open session, which the server then drops: end the session first so the library starts a fresh one on the new kernel.
    await sessionRef.current?.shutdown().catch(() => {})
    const spec = await select(name)
    // The choice lives in the notebook, the file hook saves it with the next change.
    if (spec) modelRef.current?.sharedModel.setMetadata("kernelspec", { name: spec.name, display_name: spec.display_name, language: spec.language })
  }, [select])
  const selectKernel = useCallback(async (name: string) => {
    if (name !== kernelName) await startKernelOf(name)
  }, [startKernelOf, kernelName])
  // A kernel that died, or that the Jupyter server lost (it restarts with the backend), cannot restart: a new one of the same spec takes its place.
  const restartKernel = useCallback(async () => {
    const running = sessionRef.current?.kernel
    if (running && running.status !== "dead") {
      try {
        await running.restart()
        return
      } catch { /* gone from the server: start a new one */ }
    }
    if (kernelName) await startKernelOf(kernelName)
  }, [startKernelOf, kernelName])

  if (file.error) return <LoadFailure message={file.error} onRetry={file.reload} />

  return (
    <div ref={rootRef} className="flex h-full min-h-0 flex-col bg-paper">
      <NotebookToolbar
        notebookId={tabId}
        specs={kernel.specs}
        kernelName={kernel.kernelName}
        kernelFailed={kernel.error !== null}
        onSelectKernel={(name) => void selectKernel(name)}
        onRestartKernel={() => void restartKernel()}
      />
      <div className="relative min-h-0 flex-1">
        <ErrorText className="absolute inset-x-0 top-11 z-10 bg-paper px-6 py-1.5 text-center">{kernel.error ?? file.saveError}</ErrorText>
        {initial ? (
          <div className="gc-nb absolute inset-0" style={NOTEBOOK_TOP}>
            <JupyterReactTheme colormode={theme} useBaseStyles={false} backgroundColor="var(--paper)">
              <Notebook
                id={tabId}
                nbformat={initial}
                serviceManager={serviceManager}
                kernelId={kernel.kernelId ?? undefined}
                renderers={RENDERERS}
                editorExtensions={NOTEBOOK_EDITOR_EXTENSIONS}
                cellSidebarMargin={0}
                height="100%"
                maxHeight="100%"
                onNotebookModelChanged={onModel}
                onSessionConnection={onSession}
              />
            </JupyterReactTheme>
          </div>
        ) : <LoadingPlaceholder />}
        {initial && cellTerminal && <ActiveCellTerminal tabId={tabId} terminal={cellTerminal} save={file.save} />}
      </div>
    </div>
  )
}
