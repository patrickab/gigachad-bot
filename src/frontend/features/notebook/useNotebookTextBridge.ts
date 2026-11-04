"use client"

import { useEffect, useRef } from "react"
import { notebookOf, type SharedNotebookLike } from "./notebookSync"
import { applyNotebookText, cellAtLine, notebookToText } from "./notebookText"
import type { NotebookTextBridge } from "./notebookTextBridge"

/** Quiet time after a model change before `subscribe` reports the new text. */
export const NOTEBOOK_TEXT_DELAY_MS = 150

/** A buffer extension the backend accepts. */
const EXTENSION = /^\.[A-Za-z0-9]{1,8}$/

/** The bridge on `model`: `persist` writes the model at once and resolves false when that failed, `run` runs the cell at an index. */
export function createNotebookTextBridge(model: SharedNotebookLike, persist: () => Promise<boolean>, run: (index: number) => void): NotebookTextBridge {
  const info = notebookOf(model).metadata.language_info
  const declared = typeof info === "object" && info && !Array.isArray(info) ? info.file_extension : undefined
  const extension = typeof declared === "string" && EXTENSION.test(declared) ? declared : ".py"
  const listeners = new Set<(text: string) => void>()
  const projection = () => notebookToText(notebookOf(model))
  // What the editor holds: the projection at creation until `text`, `save` or a listener hands out another
  let handedOut = projection()
  let saved: string | null = null
  let timer: ReturnType<typeof setTimeout> | undefined
  const text = () => {
    handedOut = projection()
    return handedOut
  }
  // Hand the projection to the listeners when it is not the one they last got
  const report = () => {
    clearTimeout(timer)
    const next = projection()
    if (next === handedOut) return
    handedOut = next
    for (const listener of listeners) listener(next)
  }
  const onUpdate = () => {
    clearTimeout(timer)
    timer = setTimeout(report, NOTEBOOK_TEXT_DELAY_MS)
  }
  return {
    extension,
    text,
    async save(edited) {
      // The model moved on since the text was handed out: applying it would silently revert that change
      if (projection() !== handedOut) {
        report()
        throw new Error("The notebook changed, write again")
      }
      applyNotebookText(model, edited)
      // Before the persist: the update it caused is no news to the editor that saved
      handedOut = projection()
      if (!(await persist())) throw new Error("Saving the notebook failed")
      saved = text()
      return saved
    },
    runAt(line) {
      const index = cellAtLine(saved ?? text(), line)
      if (index !== null) run(index)
    },
    subscribe(listener) {
      if (listeners.size === 0) model.ydoc.on("update", onUpdate)
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
        if (listeners.size > 0) return
        clearTimeout(timer)
        model.ydoc.off("update", onUpdate)
      }
    },
  }
}

/** Hand `onTextBridge` the bridge on `model` while it is bound, null when it goes. `persist` and `run` are read at call time. */
export function useNotebookTextBridge(model: SharedNotebookLike | null, persist: () => Promise<boolean>, run: (index: number) => void, onTextBridge?: (bridge: NotebookTextBridge | null) => void) {
  const latest = useRef({ persist, run, onTextBridge })
  useEffect(() => { latest.current = { persist, run, onTextBridge } })
  useEffect(() => {
    const announce = latest.current.onTextBridge
    if (!model || !announce) return
    announce(createNotebookTextBridge(model, () => latest.current.persist(), (index) => latest.current.run(index)))
    return () => announce(null)
  }, [model])
}
