"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { loadWorkbenchSession, saveWorkbenchSession } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"

/** The session linked to one file, once loaded. */
interface Link {
  path: string
  sessionId: string | null
}

/**
 * The omp view of one workbench file's chat: the session its tab item shows, else the one linked to the file, else a new session in `cwd`.
 *
 * - The link lives in the backend, so every device reopens the same session for the file
 * - `view` is null while the link loads and the item shows nothing yet, `error` says why loading or saving the link failed
 * - `adopt` takes the session the frame reports: the item shows it from then on and the file links it
 */
export function useWorkbenchChat(path: string, cwd: string, itemView: OmpView | undefined, onViewChange: (view: OmpView) => void) {
  const [link, setLink] = useState<Link | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The session id the backend holds for `path`, so adopting it again writes nothing
  const linkedRef = useRef<string | null>(null)
  const needsLink = !itemView

  useEffect(() => {
    if (!needsLink) return
    let cancelled = false
    loadWorkbenchSession(path).then(
      (sessionId) => {
        if (cancelled) return
        linkedRef.current = sessionId
        setLink({ path, sessionId })
      },
      (cause: unknown) => { if (!cancelled) setError(errorMessage(cause, "Could not load the chat of this file")) },
    )
    return () => { cancelled = true }
  }, [path, needsLink])

  const view = useMemo<OmpView | null>(() => {
    if (itemView) return itemView
    if (link?.path !== path) return null
    return link.sessionId ? { kind: "session", sessionId: link.sessionId } : { kind: "new", cwd }
  }, [itemView, link, path, cwd])

  const adopt = (next: OmpView) => {
    if (next.kind === "session" && next.sessionId !== linkedRef.current) {
      linkedRef.current = next.sessionId
      saveWorkbenchSession(path, next.sessionId).then(
        () => setError(null),
        (cause: unknown) => setError(errorMessage(cause, "Could not link this chat to the file")),
      )
    }
    onViewChange(next)
  }
  return { view, adopt, error }
}
