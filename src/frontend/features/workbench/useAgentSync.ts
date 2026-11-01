"use client"

import { useCallback, useEffect, useEffectEvent, useRef, type RefObject } from "react"
import { usePolling } from "@/hooks/usePolling"
import { loadFileViewerText } from "@/lib/backend/api"
import { subscribeToChanges, type ChangeEvent } from "@/lib/backend/syncStream"

/** Time between reads of the file while the agent runs. */
const POLL_MS = 500

/**
 * Mirror edits of the file at `path` from outside Neovim into it: the agent's while it runs, then once more when the run ends, and other devices' saves.
 *
 * - While `streaming`, the file on disk is read every 500 ms and pushed through `setContent` when it differs from `syncedRef`
 * - `syncedRef` holds the text Neovim and the disk last agreed on, the owner updates it on every save
 * - Every pushed text goes to `onPulled`, so the owner can follow the agent's edits as they arrive
 * - A run ending reads the file one last time, then calls `onRunEnd`
 * - A `document` change event for `path` (a save on another device) reads the file and pushes it the same way, the backend merges it into unsaved edits
 * - A push that Neovim could not take yet (`setContent` false) is retried by the next read
 */
export function useAgentSync({ path, streaming, setContent, syncedRef, onPulled, onRunEnd }: {
  path: string
  streaming: boolean
  setContent: (text: string) => boolean
  syncedRef: RefObject<string | null>
  onPulled?: (text: string) => void
  onRunEnd?: () => void
}): void {
  const pull = useCallback(async () => {
    let text: string
    try {
      text = await loadFileViewerText(path)
    } catch {
      // The next read retries, a missing file mid-write is normal
      return
    }
    if (text === syncedRef.current || !setContent(text)) return
    syncedRef.current = text
    onPulled?.(text)
  }, [path, setContent, syncedRef, onPulled])

  usePolling(pull, { intervalMs: POLL_MS, active: streaming })

  const pullOnRemoteChange = useEffectEvent((event: ChangeEvent) => {
    if (event.resource_kind === "document" && event.resource_key === path) void pull()
  })
  useEffect(() => subscribeToChanges((event) => pullOnRemoteChange(event)), [])

  const finishRun = useEffectEvent(async () => {
    await pull()
    onRunEnd?.()
  })
  const wasStreaming = useRef(streaming)
  useEffect(() => {
    if (wasStreaming.current && !streaming) void finishRun()
    wasStreaming.current = streaming
  }, [streaming])
}
