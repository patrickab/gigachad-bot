"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { informFileSession, loadFileSessions, saveFileSession, type FileSessions } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"
import { fileLead } from "./fileLead"

/** The sessions linked to one chat key, once loaded. */
interface Linked extends FileSessions {
  key: string
}

/** What `OmpChatPanel` and the chat's owner share about one file's chat. */
export type FileChat = ReturnType<typeof useFileChat>

/**
 * The omp chat linked to `key`: the session the owner's tab item shows, else the key's active session, else a new session in `cwd`.
 *
 * - `key` is the file's path, or the project folder for a chat all files of a LaTeX project share
 * - The links live in the backend, so every device reopens the same session and lists the same ones
 * - `view` is null while the links load and the item shows nothing yet, `error` says why loading or saving them failed
 * - `adopt` takes the session the frame reports: the item shows it from then on and the key links it as active
 * - `reset` starts a new session, `select` reopens one of `sessionIds`, every earlier session stays listed
 * - `lead` is the line the next message starts with: `lead` for a new session, and for a session `key` moved away from (a renamed file).
 *   It names the file by default. `onLeadSent` records that the session knows the new path
 */
export function useFileChat(key: string, cwd: string, itemView: OmpView | undefined, onViewChange: (view: OmpView) => void, lead: string = fileLead(key, cwd)) {
  const [linked, setLinked] = useState<Linked | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The active session the backend holds for `key`, so adopting it again writes nothing
  const activeRef = useRef<string | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    loadFileSessions(key).then(
      (sessions) => {
        if (cancelled) return
        activeRef.current = sessions.active
        setLinked({ key, ...sessions })
      },
      (cause: unknown) => { if (!cancelled) setError(errorMessage(cause, "Could not load the chat of this file")) },
    )
    return () => { cancelled = true }
  }, [key])

  const current = linked?.key === key ? linked : null
  const view = useMemo<OmpView | null>(() => {
    if (itemView) return itemView
    if (!current) return null
    return current.active ? { kind: "session", sessionId: current.active } : { kind: "new", cwd }
  }, [itemView, current, cwd])

  const link = (sessionId: string | null) => {
    if (sessionId === activeRef.current) return
    activeRef.current = sessionId
    saveFileSession(key, sessionId).then(
      (sessions) => {
        setLinked({ key, ...sessions })
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
    informFileSession(key, view.sessionId).then(
      (sessions) => setLinked({ key, ...sessions }),
      (cause: unknown) => setError(errorMessage(cause, "Could not update the chat of this file")),
    )
  }

  return { view, adopt, reset, select, sessionIds: current?.sessionIds ?? [], error, lead: owed ? lead : undefined, onLeadSent }
}
