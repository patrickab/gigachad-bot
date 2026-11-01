"use client"

import { useEffect, useEffectEvent, useState } from "react"
import { prepareWorkbench, type PreparedWorkbench } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"

/**
 * Prepare calls by requested path, in flight or done, so double effects and remounts ask the backend once per item.
 *
 * - A failure is dropped, so a retry asks again
 * - A moved document's disk path counts as prepared too
 * - A stored key stays until a mounted item takes its move, so a move whose item went away is not lost
 */
const preparing = new Map<string, Promise<PreparedWorkbench>>()

/** The one prepare call for `path`. */
function prepareOnce(path: string): Promise<PreparedWorkbench> {
  const known = preparing.get(path)
  if (known) return known
  const request = prepareWorkbench(path)
  preparing.set(path, request)
  request.then(
    (result) => { if (result.path !== path) preparing.set(result.path, request) },
    () => { if (preparing.get(path) === request) preparing.delete(path) },
  )
  return request
}

/**
 * Get the workbench item at `path` ready before it mounts: a stored document moves to disk, its new path goes to `onMoved`.
 *
 * - `prepared` is null until the backend answers, then holds the disk path and the chat's directory
 * - The path `onMoved` hands back counts as prepared, so the item's new path does not prepare again
 * - `error` says why preparing `path` failed, `retry` asks again. Another path clears it
 */
export function usePreparedWorkbench(path: string, onMoved: (path: string) => void) {
  const [prepared, setPrepared] = useState<PreparedWorkbench | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [requestedPath, setRequestedPath] = useState(path)
  const [attempt, setAttempt] = useState(0)
  const ready = prepared?.path === path
  const reportMoved = useEffectEvent(onMoved)

  // An earlier path's failure must not hide this path's answer.
  if (requestedPath !== path) {
    setRequestedPath(path)
    setFailure(null)
  }

  useEffect(() => {
    if (ready) return
    let cancelled = false
    prepareOnce(path).then(
      (result) => {
        if (cancelled) return
        setPrepared(result)
        if (result.path === path) return
        // The stored key is gone, a new document of that name must prepare on its own.
        preparing.delete(path)
        reportMoved(result.path)
      },
      (cause: unknown) => { if (!cancelled) setFailure(errorMessage(cause, "Could not open this file in the workbench")) },
    )
    return () => { cancelled = true }
  }, [path, ready, attempt])

  const retry = () => {
    setFailure(null)
    setAttempt((count) => count + 1)
  }
  return { prepared: ready ? prepared : null, error: ready ? null : failure, retry }
}
