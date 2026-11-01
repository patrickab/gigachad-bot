"use client"

import { useEffect, useEffectEvent, useRef, type RefObject } from "react"
import { usePolling } from "@/hooks/usePolling"
import { loadFileViewerText } from "@/lib/backend/api"
import { onFileChange } from "@/lib/backend/fileEvents"

/** Time between reads of the open files while the agent runs. */
const POLL_MS = 500

/**
 * Mirror edits of the open files from outside Neovim into it: the agent's while it runs, then once more when the run ends, and other devices' saves.
 *
 * - `synced` maps every file open in Neovim to the text Neovim and the disk last agreed on, the owner updates it on every save and every accepted push
 * - While `streaming`, every open file on disk is read every 500 ms and pushed through `setContent` when it differs from its synced text
 * - Every pushed text goes to `onPulled`, so the owner can follow the agent's edits as they arrive
 * - A run ending reads every open file one last time, then calls `onRunEnd`
 * - A `document` change event for an open file (a save on another device) reads it and pushes it the same way, the backend merges it into unsaved edits
 * - A push that Neovim could not take yet (`setContent` false) is retried by the next read
 */
export function useAgentSync({ synced, streaming, setContent, onPulled, onRunEnd }: {
  synced: RefObject<ReadonlyMap<string, string>>
  streaming: boolean
  setContent: (text: string, path: string) => boolean
  onPulled?: (text: string, path: string) => void
  onRunEnd?: () => void
}): void {
  const pull = async (path: string) => {
    let text: string
    try {
      text = await loadFileViewerText(path)
    } catch {
      // The next read retries, a missing file mid-write is normal
      return
    }
    // A file closed or renamed while it was read is not pushed
    if (!synced.current.has(path) || text === synced.current.get(path) || !setContent(text, path)) return
    onPulled?.(text, path)
  }

  const pullAll = () => Promise.all([...synced.current.keys()].map(pull))

  usePolling(pullAll, { intervalMs: POLL_MS, active: streaming })

  const pullIfOpen = useEffectEvent((path: string) => { void pull(path) })
  useEffect(() => onFileChange((path) => synced.current.has(path), (path) => pullIfOpen(path)), [synced])

  const finishRun = useEffectEvent(async () => {
    await pullAll()
    onRunEnd?.()
  })
  const wasStreaming = useRef(streaming)
  useEffect(() => {
    if (wasStreaming.current && !streaming) void finishRun()
    wasStreaming.current = streaming
  }, [streaming])
}
