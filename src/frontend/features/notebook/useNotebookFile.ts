"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useDebouncedCall } from "@/hooks/useDebouncedSave"
import { useRootFileDocument, type DocumentCodec, type FileLoadFailure } from "@/hooks/useRootFileDocument"
import { writeRootFile } from "@/lib/backend/api"
import {
  applyIncomingNotebook,
  normalizeNotebook,
  notebookOf,
  parseNotebook,
  rejoinNotebook,
  serializeNotebook,
  type NotebookContent,
  type SharedNotebookLike,
} from "./notebookSync"

/** Quiet time after the last model change before it is saved. */
export const NOTEBOOK_SAVE_DELAY_MS = 800
/** Longest a changed model waits for that quiet, so a cell streaming output keeps saving. */
export const NOTEBOOK_SAVE_MAX_WAIT_MS = 5000

/** The notebook as last read from or written to the disk. */
interface DiskState {
  /** Its text. */
  text: string
  /** Parsed as stored, with its lists of lines and key order, the style a save keeps. */
  original: NotebookContent
  /** The same with text rejoined, what a merge starts from. */
  base: NotebookContent
  /** `normalizeNotebook` of it, for comparisons. */
  norm: string
}

function diskStateOf(text: string): DiskState {
  const original = parseNotebook(text)
  return { text, original, base: rejoinNotebook(original), norm: normalizeNotebook(original, original) }
}

const NOTEBOOK_CODEC: DocumentCodec<DiskState> = { parse: diskStateOf, serialize: (state) => state.text }

const LOAD_FAILURES: Record<FileLoadFailure, string> = {
  forbidden: "This notebook is outside the folders the app may read.",
  missing: "This notebook no longer exists.",
  binary: "This is not a text file, so it cannot be opened as a notebook.",
  failed: "Failed to load this notebook, or it is not a valid notebook. Its stored content was left untouched.",
}

/** The disk state after saving `model` over `saved`, null when the file already holds what the model does. */
function savedModel(model: SharedNotebookLike, saved: DiskState | null): DiskState | null {
  if (!saved || model.isDisposed) return null
  const content = notebookOf(model)
  if (normalizeNotebook(content, saved.original) === saved.norm) return null
  return diskStateOf(serializeNotebook(content, saved.original))
}

/**
 * Load, save and follow one notebook file. The notebook editor owns the content once it runs: `initial` is for its first render only.
 *
 * - `initial` is the file's notebook with text as strings, null while loading. A blank file loads as an empty notebook, any other file that does not parse sets `error`
 * - `bindModel` gives the hook the editor's shared model (`null` when it goes away, which saves what is pending). Changes to it save after `NOTEBOOK_SAVE_DELAY_MS` of quiet,
 *   in the file's own style, and a save that would not change the file does not write
 * - `save` writes at once (Ctrl+S, Neovim) and resolves false when that failed. A failed save of either kind sets `saveError` until the next one succeeds
 * - Changes written on disk by anyone else are merged into the bound model (`applyIncomingNotebook`), this hook's own writes are ignored when they echo back
 * - A new `path` (rename) moves the reads and writes to it without a reload. `reload` retries a failed load
 *
 * Load, echo guard, follow and the ordering of saves and reads are `useRootFileDocument`'s, the debounce is `useDebouncedCall`'s. What is left here is the model: when to save it and how a disk change merges into it.
 */
export function useNotebookFile(path: string) {
  const write = useCallback((text: string) => writeRootFile(path, text), [path])
  const modelRef = useRef<SharedNotebookLike | null>(null)
  const file = useRootFileDocument({
    path,
    codec: NOTEBOOK_CODEC,
    write,
    blankWhenMissing: false,
    onFollowed: (text, _changed, replaced) => {
      const model = modelRef.current
      if (!model || model.isDisposed || !replaced) return
      let incoming: DiskState
      try {
        incoming = diskStateOf(text)
      } catch {
        // Half written by someone else, the write that finishes it reports again
        return
      }
      if (incoming.norm !== replaced.norm) applyIncomingNotebook(model, replaced.base, incoming.base)
    },
  })
  // What the callbacks below call is read at the time, a rename gives them new functions without giving the model new ones.
  const latest = useRef(file)
  useEffect(() => { latest.current = file })

  // The model the editor was created from: what a model bound after a disk change must catch up with.
  const [initial, setInitial] = useState<NotebookContent | null>(null)
  if (initial === null && file.value) setInitial(file.value.base)
  const initialRef = useRef(initial)
  useEffect(() => { initialRef.current = initial })

  const { run: runSave, error: saveFailure } = useAsyncAction(
    useCallback((model: SharedNotebookLike) => latest.current.persistFrom((saved) => savedModel(model, saved)), []),
    "Save failed",
  )
  const { schedule, flush, cancel } = useDebouncedCall(NOTEBOOK_SAVE_DELAY_MS, NOTEBOOK_SAVE_MAX_WAIT_MS)

  const save = useCallback(async () => {
    cancel()
    const model = modelRef.current
    return model ? runSave(model) : true
  }, [cancel, runSave])

  const detachRef = useRef<(() => void) | null>(null)
  const bindModel = useCallback((model: SharedNotebookLike | null) => {
    if (modelRef.current === model) return
    flush()
    detachRef.current?.()
    detachRef.current = null
    modelRef.current = model
    if (!model) return

    // The file may have changed between the load and this model: catch up with it
    const saved = latest.current.saved()
    const shown = initialRef.current
    if (saved && shown && normalizeNotebook(shown, shown) !== saved.norm) applyIncomingNotebook(model, shown, saved.base)

    const onUpdate = () => schedule(() => { void runSave(model) })
    model.ydoc.on("update", onUpdate)
    detachRef.current = () => model.ydoc.off("update", onUpdate)
  }, [flush, schedule, runSave])

  const reloadRef = useRef(file.reload)
  useEffect(() => { reloadRef.current = file.reload })
  // Once: a rename gives `reload` a new path, which must not read the file again.
  useEffect(() => { reloadRef.current() }, [])

  // The tab closing saves what is pending. The model stays bound, a dev-mode remount keeps it and a disposed one stops sending changes
  useEffect(() => () => flush(), [flush])

  return {
    initial,
    error: file.loadError && LOAD_FAILURES[file.loadError],
    saveError: saveFailure || null,
    bindModel,
    save,
    reload: file.reload,
  }
}
