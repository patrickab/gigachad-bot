"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import { ApiError, loadFileViewerText } from "@/lib/backend/api"
import { subscribeToChanges, type ChangeEvent } from "@/lib/backend/syncStream"

export type FileLoadFailure = "failed" | "binary" | "forbidden" | "missing"

interface RootFileDocumentOptions {
  path: string
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
 * Load, edit and save one text file the backend serves.
 *
 * - Saves are chained so an earlier slow write can never resolve after, and clobber, a newer one
 * - Reloads on a remote change of the document unless there are unsaved local edits
 * - The caller starts the first load, `reload()` in an effect
 * - Any load failure except a tolerated 404 leaves `content` null, so nothing can save blank text over the stored file
 */
export function useRootFileDocument({ path, write, blankWhenMissing, readBackAfterSave = false, onSaved }: RootFileDocumentOptions) {
  const [content, setContent] = useState<string | null>(null)
  const [loadedPath, setLoadedPath] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<FileLoadFailure | null>(null)
  const [dirty, setDirty] = useState(false)
  const savedContentRef = useRef("")

  const reload = useCallback(() => {
    const apply = (text: string) => {
      setLoadError(null)
      setContent(text)
      setLoadedPath(path)
      savedContentRef.current = text
    }
    return loadFileViewerText(path).then(apply).catch((error: unknown) => {
      const failure = loadFailureOf(error)
      if (failure === "missing" && blankWhenMissing) apply("")
      else setLoadError(failure)
    })
  }, [path, blankWhenMissing])

  const handleTextChange = useCallback((value: string) => {
    setContent(value)
    setDirty(value !== savedContentRef.current)
  }, [])

  // `resource_key` may be a path suffix.
  const reloadOnRemoteChange = useEffectEvent((event: ChangeEvent) => {
    if (dirty || event.resource_kind !== "document") return
    if (event.resource_key !== path && !path.endsWith(`/${event.resource_key}`)) return
    reload()
  })
  useEffect(() => subscribeToChanges((event) => reloadOnRemoteChange(event)), [])

  const persistNow = useCallback(async (next: string) => {
    await write(next)
    savedContentRef.current = next
    setDirty(false)
    if (readBackAfterSave) {
      const stored = await loadFileViewerText(path).catch(() => next)
      if (stored !== next) { setContent(stored); savedContentRef.current = stored }
    }
    onSaved?.(next)
  }, [write, path, readBackAfterSave, onSaved])

  const saveChain = useRef(Promise.resolve())
  const persist = useCallback((next: string) => {
    const run = saveChain.current.then(() => persistNow(next))
    saveChain.current = run.catch(() => {})
    return run
  }, [persistNow])

  /** Neovim owns the text while it runs: persist `text`, then show and return what was stored. */
  const saveFromNeovim = useCallback(async (text: string) => {
    await persist(text)
    setContent(savedContentRef.current)
    return savedContentRef.current
  }, [persist])

  return { content, loadedPath, loadError, dirty, reload, handleTextChange, persist, saveFromNeovim }
}
