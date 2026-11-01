"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { informFileSession, loadFileSessions, saveFileSession, type FileSessions } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"
import { fileLead } from "./fileLead"

/** The sessions linked to one file, once loaded. */
interface Linked extends FileSessions {
  path: string
}

/** What `OmpChatPanel` and the chat's owner share about one file's chat. */
export type FileChat = ReturnType<typeof useFileChat>

/**
 * The omp chat of one file: the session the owner's tab item shows, else the file's active session, else a new session in `cwd`.
 *
 * - The links live in the backend, so every device reopens the same session and lists the same ones
 * - `view` is null while the links load and the item shows nothing yet, `error` says why loading or saving them failed
 * - `adopt` takes the session the frame reports: the item shows it from then on and the file links it as active
 * - `reset` starts a new session, `select` reopens one of `sessionIds`, every earlier session stays listed
 * - `lead` is the line the next message starts with: the file for a new session, and for a session the file moved away from.
 *   `onLeadSent` records that the session knows the file's new path
 */
export function useFileChat(path: string, cwd: string, itemView: OmpView | undefined, onViewChange: (view: OmpView) => void) {
  const [linked, setLinked] = useState<Linked | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The active session the backend holds for `path`, so adopting it again writes nothing
  const activeRef = useRef<string | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    loadFileSessions(path).then(
      (sessions) => {
        if (cancelled) return
        activeRef.current = sessions.active
        setLinked({ path, ...sessions })
      },
      (cause: unknown) => { if (!cancelled) setError(errorMessage(cause, "Could not load the chat of this file")) },
    )
    return () => { cancelled = true }
  }, [path])

  const current = linked?.path === path ? linked : null
  const view = useMemo<OmpView | null>(() => {
    if (itemView) return itemView
    if (!current) return null
    return current.active ? { kind: "session", sessionId: current.active } : { kind: "new", cwd }
  }, [itemView, current, cwd])

  const link = (sessionId: string | null) => {
    if (sessionId === activeRef.current) return
    activeRef.current = sessionId
    saveFileSession(path, sessionId).then(
      (sessions) => {
        setLinked({ path, ...sessions })
        setError(null)
      },
      (cause: unknown) => setError(errorMessage(cause, "Could not link this chat to the file")),
    )
  }

  const adopt = (next: OmpView) => {
    if (next.kind === "session") link(next.sessionId)
    onViewChange(next)
  }
  const reset = () => {
    link(null)
    onViewChange({ kind: "new", cwd })
  }
  const select = (sessionId: string) => {
    link(sessionId)
    onViewChange({ kind: "session", sessionId })
  }

  const owed = view?.kind === "new" || (view?.kind === "session" && (current?.moved.includes(view.sessionId) ?? false))
  const onLeadSent = () => {
    if (view?.kind !== "session") return
    informFileSession(path, view.sessionId).then(
      (sessions) => setLinked({ path, ...sessions }),
      (cause: unknown) => setError(errorMessage(cause, "Could not update the chat of this file")),
    )
  }

  return { view, adopt, reset, select, sessionIds: current?.sessionIds ?? [], error, lead: owed ? fileLead(path, cwd) : undefined, onLeadSent }
}
