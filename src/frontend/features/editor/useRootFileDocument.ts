"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import { ApiError, loadFileViewerText } from "@/lib/backend/api"
import { subscribeToChanges, type ChangeEvent } from "@/lib/backend/syncStream"

export type FileLoadFailure = "failed" | "binary" | "forbidden" | "missing"

/** Stored text to and from the edited value. Pass a module constant, the hook's callbacks depend on it. */
export interface DocumentCodec<T> {
  parse: (text: string) => T
  serialize: (value: T) => string
}

/** Plain text documents edit the stored text itself. */
export const TEXT_CODEC: DocumentCodec<string> = { parse: (text) => text, serialize: (text) => text }

interface RootFileDocumentOptions<T> {
  path: string
  codec: DocumentCodec<T>
  /** Write `text` to its store. */
  write: (text: string) => Promise<void>
  /** A 404 starts a blank document (an editor creating it on save) instead of failing with "missing". */
  blankWhenMissing: boolean
  /** Re-read the stored text after each save, for stores that normalize what they keep. */
  readBackAfterSave?: boolean
  onSaved?: (text: string) => void
}

function loadFailureOf(error: unknown): FileLoadFailure {
  if (!(error instanceof ApiError)) return "failed"
  if (error.status === 403) return "forbidden"
  if (error.status === 404) return "missing"
  if (error.status === 415) return "binary"
  return "failed"
}

/**
 * Load, edit and save one file the backend serves, as `codec` reads its text.
 *
 * - Saves are chained so an earlier slow write can never resolve after, and clobber, a newer one
 * - `dirty` while the value is not the one last loaded or saved; reloads on a remote change of the document only when clean
 * - The caller starts the first load, `reload()` in an effect
 * - Any load failure except a tolerated 404 leaves `value` null, so nothing can save a blank document over the stored file
 */
export function useRootFileDocument<T>({ path, codec, write, blankWhenMissing, readBackAfterSave = false, onSaved }: RootFileDocumentOptions<T>) {
  const [value, setValueState] = useState<T | null>(null)
  const [loadedPath, setLoadedPath] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<FileLoadFailure | null>(null)
  const [dirty, setDirtyState] = useState(false)
  // Mirrors `dirty` synchronously, so a remote change arriving before the next commit still sees an edit.
  const dirtyRef = useRef(false)
  const setDirty = useCallback((next: boolean) => {
    dirtyRef.current = next
    setDirtyState(next)
  }, [])
  const savedRef = useRef<{ value: T | null; text: string }>({ value: null, text: "" })

  const reload = useCallback(() => {
    const apply = (text: string) => {
      const loaded = codec.parse(text)
      setLoadError(null)
      setValueState(loaded)
      setLoadedPath(path)
      savedRef.current = { value: loaded, text: codec.serialize(loaded) }
    }
    return loadFileViewerText(path).then(apply).catch((error: unknown) => {
      const failure = loadFailureOf(error)
      if (failure === "missing" && blankWhenMissing) apply("")
      else setLoadError(failure)
    })
  }, [path, codec, blankWhenMissing])

  const setValue = useCallback((next: T) => {
    setValueState(next)
    setDirty(next !== savedRef.current.value)
  }, [setDirty])

  // `resource_key` may be a path suffix.
  const reloadOnRemoteChange = useEffectEvent((event: ChangeEvent) => {
    if (dirtyRef.current || event.resource_kind !== "document") return
    if (event.resource_key !== path && !path.endsWith(`/${event.resource_key}`)) return
    reload()
  })
  useEffect(() => subscribeToChanges((event) => reloadOnRemoteChange(event)), [])

  const persistNow = useCallback(async (next: T, text: string) => {
    await write(text)
    savedRef.current = { value: next, text }
    setDirty(false)
    if (readBackAfterSave) {
      const stored = await loadFileViewerText(path).catch(() => text)
      if (stored !== text) {
        const parsed = codec.parse(stored)
        setValueState(parsed)
        savedRef.current = { value: parsed, text: stored }
      }
    }
    onSaved?.(text)
  }, [write, path, codec, readBackAfterSave, onSaved, setDirty])

  const saveChain = useRef(Promise.resolve())
  /** Queue a save of `next`; `onlyIfChanged` skips text equal to the last loaded or saved one. */
  const persist = useCallback((next: T, onlyIfChanged = false) => {
    const text = codec.serialize(next)
    if (onlyIfChanged && text === savedRef.current.text) return Promise.resolve()
    const run = saveChain.current.then(() => persistNow(next, text))
    saveChain.current = run.catch(() => {})
    return run
  }, [codec, persistNow])

  /** Neovim owns the text while it runs: persist `text`, then show and return what was stored. */
  const saveFromNeovim = useCallback(async (text: string) => {
    await persist(codec.parse(text))
    setValueState(savedRef.current.value)
    return savedRef.current.text
  }, [codec, persist])

  return { value, loadedPath, loadError, dirty, reload, setValue, persist, saveFromNeovim }
}
