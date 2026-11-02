"use client"

import { NotebookCommandIds, useNotebookStore } from "@datalayer/jupyter-react"
import { useEffect, useRef, useState } from "react"
import { CellNeovim, type CellModelLike, type LanguageInfoLike, type NotebookWidgetLike } from "./cellNeovim"
import type { CellTerminal } from "./cellTerminal"

/** `CellNeovim` on the notebook shown by the tab `tabId`, null until the notebook panel exists. `save` is `:w` in Neovim. */
export function useActiveCellNeovim(tabId: string, terminal: CellTerminal, save: () => Promise<void>): CellNeovim | null {
  const adapter = useNotebookStore((state) => state.notebooks.get(tabId)?.adapter)
  const model = useNotebookStore((state) => state.notebooks.get(tabId)?.model)
  const [controller, setController] = useState<CellNeovim | null>(null)
  const terminalRef = useRef(terminal)
  const saveRef = useRef(save)
  useEffect(() => {
    terminalRef.current = terminal
    saveRef.current = save
  })
  const subscribe = terminal.subscribe

  useEffect(() => {
    if (!adapter || !model) return
    const shared = model.sharedModel as unknown as CellModelLike
    const next = new CellNeovim({
      notebook: adapter.notebook as unknown as NotebookWidgetLike,
      model: shared,
      terminal: terminalRef.current,
      language: () => {
        const info = model.sharedModel.getMetadata("language_info") as LanguageInfoLike | undefined
        const spec = model.sharedModel.getMetadata("kernelspec") as { language?: string } | undefined
        return info ?? (spec?.language ? { name: spec.language } : null)
      },
      run: () => { void adapter.commands.execute(NotebookCommandIds.run) },
      runAndAdvance: () => { void adapter.commands.execute(NotebookCommandIds.runAndAdvance) },
      save: () => { void saveRef.current() },
    })
    const unsubscribe = subscribe(next.listener)
    next.start()
    setController(next)
    return () => {
      unsubscribe()
      next.dispose()
      setController(null)
    }
  }, [adapter, model, tabId, subscribe])

  useEffect(() => { controller?.update(terminal) }, [controller, terminal])
  return controller
}
