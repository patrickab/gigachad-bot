"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState, type RefObject } from "react"
import { useNeovimTerminal, type NeovimSelection, type NeovimStatus, type NeovimTerminal } from "@/features/editor/NeovimEditor"
import { loadFailureOf, type FileLoadFailure } from "@/hooks/useRootFileDocument"
import { loadFileViewerText, writeRootFile } from "@/lib/backend/api"
import { onFileMove } from "@/lib/backend/fileEvents"
import { syncCursorReady } from "@/lib/backend/syncStream"
import { baseName } from "@/lib/paths"
import { errorMessage } from "@/lib/utils"

/** Time between tries to show a file while Neovim's session is not open yet. */
const OPEN_RETRY_MS = 200

interface WorkbenchEditorOptions {
  /** The shown file. The first path starts the session, another one under the same session switches Neovim's buffer to it. */
  path: string
  /** Whether Neovim may run. */
  enabled: boolean
  /** Neovim shows another buffer (`:bnext`), or a file that could not be opened goes back to the one it shows. */
  onPathChange: (path: string) => void
  /** A `:w` stored `text` as the file at `path`. */
  onSaved?: (text: string, path: string) => void
  /** `:q` in Neovim. */
  onClose: () => void
  onSelection: (selection: NeovimSelection) => void
}

interface WorkbenchEditor {
  terminal: NeovimTerminal | null
  status: NeovimStatus
  restart: () => void
  /** The text the session opens with: the first file's, null while it loads or after a failed load. */
  content: string | null
  /** Why the first file could not load, null otherwise. */
  loadError: FileLoadFailure | null
  /** The text Neovim and the disk last agreed on for the shown file, null while unknown. */
  shownText: string | null
  /** The last thing that went wrong beside the editor (a file that did not open), null after the next success. */
  notice: string | null
  /** Every file open in Neovim and the text Neovim and the disk last agreed on. */
  synced: RefObject<ReadonlyMap<string, string>>
  /** The paths of the files open in Neovim. */
  openPaths: readonly string[]
  /** Hand text of the file at `path`, already on disk, to Neovim. False while it cannot take it. */
  setContent: (text: string, path: string) => boolean
}

/**
 * One Neovim session over the files of a workbench: the shown file and every other file shown since, as hidden buffers.
 *
 * - The session starts with the first `path` and learns later ones through `openFile`, loading each from disk once
 * - A `:bnext` in Neovim reports the new buffer to `onPathChange`, the owner's `path` follows
 * - Saves go to the path Neovim names, one write at a time
 * - A file this device renames (`onFileMove`) or any device does (a change event with `moved_to`) is renamed in Neovim, unsaved edits stay
 * - A restart keeps the first file and the shown one, the other buffers are gone
 */
export function useWorkbenchEditor({ path, enabled, onPathChange, onSaved, onClose, onSelection }: WorkbenchEditorOptions): WorkbenchEditor {
  // The session is named after its first file for good, its Neovim buffer follows the file's renames
  const [sessionKey] = useState(path)
  const [startPath, setStartPath] = useState(path)
  const [synced, setSynced] = useState<ReadonlyMap<string, string>>(() => new Map())
  // The agent sync and the Neovim callbacks read the map between renders, so it lives in a ref as well
  const syncedRef = useRef(synced)
  const [loadError, setLoadError] = useState<FileLoadFailure | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  // The path Neovim shows. Not state: it must be right before the owner's path catches up
  const shown = useRef(path)
  // The path Neovim showed before the latest `openFile`, back on screen when that file cannot load
  const shownBefore = useRef(path)
  const [restarts, setRestarts] = useState(0)

  const updateSynced = useCallback((change: (next: Map<string, string>) => void) => {
    const next = new Map(syncedRef.current)
    change(next)
    syncedRef.current = next
    setSynced(next)
  }, [])

  useEffect(() => {
    let cancelled = false
    // After the change feed holds its head, so a change racing this read still arrives as an event
    syncCursorReady().then(() => loadFileViewerText(sessionKey)).then(
      (text) => { if (!cancelled) updateSynced((next) => { next.set(sessionKey, text) }) },
      (cause: unknown) => { if (!cancelled) setLoadError(loadFailureOf(cause)) },
    )
    return () => { cancelled = true }
  }, [sessionKey, updateSynced])

  const saveChain = useRef<Promise<unknown>>(Promise.resolve())
  const save = (text: string, savedPath: string) => {
    const run = saveChain.current.then(() => writeRootFile(savedPath, text))
    saveChain.current = run.catch(() => {})
    return run.then(() => {
      updateSynced((next) => { if (next.has(savedPath)) next.set(savedPath, text) })
      onSaved?.(text, savedPath)
      return text
    })
  }

  const switchedTo = (target: string) => {
    if (target === shown.current) return
    shown.current = target
    onPathChange(target)
  }

  const openFailed = (failedPath: string, message: string) => {
    updateSynced((next) => { next.delete(failedPath) })
    setNotice(message)
    if (failedPath !== shown.current) return
    shown.current = shownBefore.current
    onPathChange(shown.current)
  }

  const openPaths = [...synced.keys()]

  const startText = synced.get(startPath)
  const opening = enabled && startText !== undefined ? { key: sessionKey, path: startPath, filename: baseName(startPath), content: startText } : null
  const editor = useNeovimTerminal(opening, save, onClose, { askKeymap: true, onSelection, onSwitched: switchedTo, onOpenFailed: openFailed, shownPath: () => shown.current })
  const { setContent: pushToNeovim } = editor

  // Without Neovim there is no buffer to merge into: the disk text is what the rendered pane shows
  const setContent = useCallback((text: string, bufferPath: string) => {
    const taken = !enabled || pushToNeovim(text, bufferPath)
    if (taken) updateSynced((next) => { if (next.has(bufferPath)) next.set(bufferPath, text) })
    return taken
  }, [enabled, pushToNeovim, updateSynced])

  const openInEditor = useEffectEvent((target: string) => editor.openFile(target))
  const showShownAgain = useEffectEvent(() => onPathChange(shown.current))
  // The owner's path moved on: load the file when Neovim has not got it yet, then show it.
  // Neovim answers `false` while its session is not yet there, the next try follows
  useEffect(() => {
    if (!enabled || path === shown.current) return
    let cancelled = false
    let timer: number | undefined
    const attempt = async () => {
      let loaded: string | undefined
      if (!syncedRef.current.has(path)) {
        try {
          loaded = await loadFileViewerText(path)
        } catch (cause) {
          if (cancelled) return
          setNotice(errorMessage(cause, `Could not open ${baseName(path)}`))
          showShownAgain()
          return
        }
      }
      if (cancelled) return
      if (!openInEditor(path)) {
        timer = window.setTimeout(() => void attempt(), OPEN_RETRY_MS)
        return
      }
      shownBefore.current = shown.current
      shown.current = path
      setNotice(null)
      const text = loaded
      if (text !== undefined) updateSynced((next) => { next.set(path, text) })
    }
    void attempt()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [path, enabled, restarts, updateSynced])

  const renameBuffer = useEffectEvent((from: string, to: string) => {
    const text = syncedRef.current.get(from)
    if (text === undefined) return
    editor.renameFile(from, to)
    updateSynced((next) => {
      next.delete(from)
      next.set(to, text)
    })
    if (shown.current === from) shown.current = to
    setStartPath((current) => current === from ? to : current)
  })
  useEffect(() => onFileMove(({ from, to }) => renameBuffer(from, to)), [])

  const restart = () => {
    editor.restart()
    updateSynced((next) => {
      for (const key of next.keys()) if (key !== startPath) next.delete(key)
    })
    shown.current = startPath
    shownBefore.current = startPath
    setRestarts((count) => count + 1)
  }

  return {
    terminal: editor.terminal,
    status: editor.status,
    restart,
    content: startText ?? null,
    loadError,
    shownText: synced.get(path) ?? null,
    notice,
    synced: syncedRef,
    openPaths,
    setContent,
  }
}
