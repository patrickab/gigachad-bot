"use client"

import { useCallback, useEffect, useEffectEvent, useState, type RefObject } from "react"
import { usePolling } from "@/hooks/usePolling"
import { loadFileViewerText } from "@/lib/backend/api"

/** Time between reads of the file while the agent runs. */
const POLL_MS = 500

/**
 * Mirror the agent's edits of the file at `path` into Neovim while it runs, then hand over once the run ends.
 *
 * - While `streaming`, the file on disk is read every 500 ms and pushed through `setContent` when it differs from `syncedRef`
 * - `syncedRef` holds the text Neovim and the disk last agreed on, the owner updates it on every save
 * - A run ending reads the file one last time, then awaits `onRunEnd` (the build). `editing` covers the run and that handover
 * - A push that Neovim could not take yet (`setContent` false) is retried by the next read
 */
export function useAgentSync({ path, streaming, setContent, syncedRef, onRunEnd }: {
  path: string
  streaming: boolean
  setContent: (text: string) => boolean
  syncedRef: RefObject<string | null>
  onRunEnd: () => Promise<void>
}): { editing: boolean } {
  const [handingOver, setHandingOver] = useState(false)
  const [wasStreaming, setWasStreaming] = useState(streaming)
  if (wasStreaming !== streaming) {
    setWasStreaming(streaming)
    if (wasStreaming) setHandingOver(true)
  }

  const pull = useCallback(async () => {
    try {
      const text = await loadFileViewerText(path)
      if (text !== syncedRef.current && setContent(text)) syncedRef.current = text
    } catch {
      // The next read retries, a missing file mid-write is normal
    }
  }, [path, setContent, syncedRef])

  usePolling(pull, { intervalMs: POLL_MS, active: streaming })

  const finishRun = useEffectEvent(async () => {
    await pull()
    await onRunEnd()
  })
  useEffect(() => {
    if (!handingOver) return
    let cancelled = false
    void finishRun().finally(() => { if (!cancelled) setHandingOver(false) })
    return () => { cancelled = true }
  }, [handingOver])

  return { editing: streaming || handingOver }
}
