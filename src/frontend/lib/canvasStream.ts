"use client"

import { canvasMutationStreamUrl, type CanvasAcceptedBatch } from "./api"

const RECONNECT_INITIAL_MS = 250
const RECONNECT_MAX_MS = 5_000

/**
 * Subscribe to one canvas's gapless batch feed. Every (re)connect resumes after
 * `sinceRevision()`, so a dropped connection never replays or skips batches.
 */
export function subscribeToCanvasMutationBatches(
  canvasKey: string,
  sinceRevision: () => number,
  onBatch: (batch: CanvasAcceptedBatch) => void,
  onError: () => void,
  onOpen: () => void,
): () => void {
  let source: EventSource | null = null
  let timer: NodeJS.Timeout | number | undefined
  let attempts = 0

  const open = () => {
    source = new EventSource(canvasMutationStreamUrl(canvasKey, sinceRevision()))
    source.addEventListener("batch", (message) => onBatch(JSON.parse((message as MessageEvent<string>).data)))
    source.onopen = () => {
      attempts = 0
      onOpen()
    }
    source.onerror = () => {
      source?.close()
      onError()
      timer = setTimeout(open, Math.min(RECONNECT_INITIAL_MS * 2 ** attempts++, RECONNECT_MAX_MS))
    }
  }
  open()

  return () => {
    clearTimeout(timer)
    source?.close()
  }
}
