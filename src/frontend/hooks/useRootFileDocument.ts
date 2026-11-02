"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import { useFollowFiles } from "@/hooks/useFollowFiles"
import { ApiError, loadFileViewerText } from "@/lib/backend/api"
import { subscribeToChanges, syncCursorReady, type ChangeEvent } from "@/lib/backend/syncStream"

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
  /** Called with the text of a followed disk change whatever the state of the edit, and the value it replaces (as last loaded or saved), for editors that merge it themselves. */
  onFollowed?: (text: string, path: string, replaced: T | null) => void
}

const isDiskPath = (path: string) => path.startsWith("/")

/** Why a failed read of a file could not load it. */
export function loadFailureOf(error: unknown): FileLoadFailure {
  if (!(error instanceof ApiError)) return "failed"
  if (error.status === 403) return "forbidden"
  if (error.status === 404) return "missing"
  if (error.status === 415) return "binary"
  return "failed"
}

/**
 * Load, edit and save one file the backend serves, as `codec` reads its text.
 *
 * - Saves are chained so an earlier slow write can never resolve after, and clobber, a newer one, and a followed disk change is read behind them, so the echo of an own write is recognised
 * - `dirty` while the value is not the one last loaded or saved
 * - Follows the file: a change of an absolute `path` on disk (by anyone) and a remote change event of a stored key reload the value only when clean
 * - The caller starts the first load, `reload()` in an effect
 * - Any load failure except a tolerated 404 leaves `value` null, so nothing can save a blank document over the stored file
 */
export function useRootFileDocument<T>({ path, codec, write, blankWhenMissing, readBackAfterSave = false, onSaved, onFollowed }: RootFileDocumentOptions<T>) {
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
  // The file's text as last read or written, which a codec may have normalized in `savedRef`.
  const diskTextRef = useRef<string | null>(null)

  const apply = useCallback((text: string) => {
    const loaded = codec.parse(text)
    setLoadError(null)
    setValueState(loaded)
    setLoadedPath(path)
    savedRef.current = { value: loaded, text: codec.serialize(loaded) }
    diskTextRef.current = text
  }, [path, codec])

  const reload = useCallback(() => {
    // After the change feed holds its head, so a change racing this read still arrives as an event.
    return syncCursorReady().then(() => loadFileViewerText(path)).then(apply).catch((error: unknown) => {
      const failure = loadFailureOf(error)
      if (failure === "missing" && blankWhenMissing) apply("")
      else setLoadError(failure)
    })
  }, [path, blankWhenMissing, apply])

  const setValue = useCallback((next: T) => {
    setValueState(next)
    setDirty(next !== savedRef.current.value)
  }, [setDirty])

  // Stored keys follow their change events (`resource_key` may be a path suffix), absolute paths follow the disk below.
  const reloadOnRemoteChange = useEffectEvent((event: ChangeEvent) => {
    if (dirtyRef.current || isDiskPath(path) || event.resource_kind !== "document") return
    if (event.resource_key !== path && !path.endsWith(`/${event.resource_key}`)) return
    reload()
  })
  useEffect(() => subscribeToChanges((event) => reloadOnRemoteChange(event)), [])

  // Saves and followed reads run one at a time: a read after a write sees the text that write recorded, so its echo is ignored.
  const saveChain = useRef(Promise.resolve())

  // A read still running when the path changes must not land on the new document.
  const pathRef = useRef(path)
  useEffect(() => { pathRef.current = path }, [path])
  useFollowFiles(isDiskPath(path) ? [path] : [], (changed) => {
    saveChain.current = saveChain.current.then(() => loadFileViewerText(changed)).then((text) => {
      if (changed !== pathRef.current || text === diskTextRef.current) return
      onFollowed?.(text, changed, savedRef.current.value)
      if (!dirtyRef.current) apply(text)
    }).catch(() => {})
  })

  const persistNow = useCallback(async (next: T, text: string) => {
    await write(text)
    savedRef.current = { value: next, text }
    diskTextRef.current = text
    setDirty(false)
    if (readBackAfterSave) {
      const stored = await loadFileViewerText(path).catch(() => text)
      if (stored !== text) {
        const parsed = codec.parse(stored)
        setValueState(parsed)
        savedRef.current = { value: parsed, text: stored }
        diskTextRef.current = stored
      }
    }
    onSaved?.(text)
  }, [write, path, codec, readBackAfterSave, onSaved, setDirty])

  /** The value as last loaded or saved, null before the first load. */
  const saved = useCallback(() => savedRef.current.value, [])

  /**
   * Queue a save of what `produce` returns once every earlier save and followed read has finished, null saves nothing.
   * `produce` gets the value as last loaded or saved, which a value changing outside React state needs to be judged against.
   */
  const persistFrom = useCallback((produce: (saved: T | null) => T | null) => {
    const run = saveChain.current.then(async () => {
      const next = produce(saved())
      if (next !== null) await persistNow(next, codec.serialize(next))
    })
    saveChain.current = run.catch(() => {})
    return run
  }, [codec, persistNow, saved])

  /** Queue a save of `next`; `onlyIfChanged` skips text equal to the last loaded or saved one. */
  const persist = useCallback((next: T, onlyIfChanged = false) => {
    return persistFrom(() => onlyIfChanged && codec.serialize(next) === savedRef.current.text ? null : next)
  }, [codec, persistFrom])


  /** Neovim owns the text while it runs: persist `text`, then show and return what was stored. */
  const saveFromNeovim = useCallback(async (text: string) => {
    await persist(codec.parse(text))
    setValueState(savedRef.current.value)
    return savedRef.current.text
  }, [codec, persist])

  return { value, loadedPath, loadError, dirty, reload, setValue, persist, persistFrom, saved, saveFromNeovim }
}
