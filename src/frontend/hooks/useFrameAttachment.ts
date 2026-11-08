"use client"

import { useCallback, useState } from "react"

/**
 * Which tab a shared frame (omp, learn) is laid over: the tab the tab picker previews while it shows one, else the frontmost.
 *
 * - `attach(next)` registers a tab's slot as frontmost or, with `peek`, previewed, and returns the detach
 * - A preview of a tab without a frame leaves the frame with the frontmost tab
 */
export function useFrameAttachment<A extends { peek?: boolean }>() {
  const [attachments, setAttachments] = useState<{ active: A | null; peek: A | null }>({ active: null, peek: null })
  const attach = useCallback((next: A) => {
    const role = next.peek ? "peek" : "active"
    setAttachments((current) => ({ ...current, [role]: next }))
    return () => setAttachments((current) => (current[role] === next ? { ...current, [role]: null } : current))
  }, [])
  return { attachment: attachments.peek ?? attachments.active, peeked: attachments.peek !== null, attach }
}
