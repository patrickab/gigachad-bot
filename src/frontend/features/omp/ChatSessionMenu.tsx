"use client"

import { useState } from "react"
import { ChevronDown, Plus, Trash2 } from "lucide-react"
import { ErrorText } from "@/components/ui/ErrorText"
import { MenuPopover } from "@/components/ui/MenuPopover"
import { deleteOmpSession } from "@/lib/backend/api"
import type { OmpSession, OmpView } from "@/lib/domain/types"
import { cn, errorMessage } from "@/lib/utils"
import { withoutFileLead } from "./fileLead"
import { useOmp } from "./OmpProvider"
import { sessionTitle } from "./sessions"

const ICON_BUTTON = "rounded-md p-1 text-ink-subtle transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

function titleOf(session: OmpSession): string {
  return sessionTitle({ name: session.name, first_message: withoutFileLead(session.first_message) })
}

/** The file's sessions omp-web still lists, newest first. */
function sessionsOf(sessionIds: readonly string[], known: readonly OmpSession[]): OmpSession[] {
  const ids = new Set(sessionIds)
  return known.filter((session) => ids.has(session.id)).sort((a, b) => b.modified.localeCompare(a.modified))
}

/**
 * Chat panel header of a file: the shown session with a menu of every session of the file, and a button starting a new one.
 *
 * - The menu lists only sessions omp-web still has, an archived or deleted one drops out
 * - The trash button deletes a session for good, deleting the shown one starts a new session
 */
export function ChatSessionMenu({ view, sessionIds, onSelect, onReset }: {
  view: OmpView | null
  sessionIds: readonly string[]
  onSelect: (sessionId: string) => void
  onReset: () => void
}) {
  const { overview, refreshOverview } = useOmp()
  const [error, setError] = useState<string | null>(null)
  const sessions = sessionsOf(sessionIds, overview?.sessions ?? [])
  const activeId = view?.kind === "session" ? view.sessionId : null
  const remove = async (sessionId: string) => {
    try {
      await deleteOmpSession(sessionId)
      setError(null)
      refreshOverview()
      if (sessionId === activeId) onReset()
    } catch (cause) {
      setError(errorMessage(cause, "Could not delete the session"))
    }
  }
  const active = sessions.find((session) => session.id === activeId)
  const label = active ? titleOf(active) : view?.kind === "new" ? "New session" : "Chat"

  return (
    <div className="flex min-w-0 flex-1 items-center gap-1">
      <MenuPopover
        className="relative min-w-0 flex-1"
        trigger={(open, toggle) => (
          <button
            type="button"
            aria-label="Sessions of this file"
            aria-expanded={open}
            onClick={toggle}
            disabled={sessions.length === 0}
            className="flex max-w-full items-center gap-1 rounded-md px-1 py-0.5 text-ink-subtle transition-colors enabled:hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
          >
            <span className="truncate">{label}</span>
            {sessions.length > 0 && <ChevronDown aria-hidden className={cn("h-3 w-3 shrink-0 transition-transform", open && "rotate-180")} />}
          </button>
        )}
      >
        {(open, close) => open && (
          <ul className="absolute left-0 top-full z-50 mt-1 flex max-h-[60vh] w-72 max-w-[80vw] flex-col overflow-y-auto rounded-2xl border border-divider/50 bg-surface p-1 text-xs shadow-[var(--shadow-lg)]">
            {sessions.map((session) => (
              <li key={session.id} className="group flex items-center rounded-xl transition-colors hover:bg-hover">
                <button
                  type="button"
                  aria-current={session.id === activeId ? "true" : undefined}
                  onClick={() => { close(); if (session.id !== activeId) onSelect(session.id) }}
                  className={cn(
                    "min-w-0 flex-1 truncate rounded-xl px-2.5 py-1.5 text-left focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted",
                    session.id === activeId ? "text-ink" : "text-ink-subtle",
                  )}
                >
                  {titleOf(session)}
                </button>
                <button
                  type="button"
                  aria-label={`Delete ${titleOf(session)}`}
                  onClick={() => void remove(session.id)}
                  className={cn(ICON_BUTTON, "mr-1 shrink-0 opacity-0 group-hover:opacity-100 focus-visible:opacity-100")}
                >
                  <Trash2 aria-hidden className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
            {error && <ErrorText className="px-2.5 py-1">{error}</ErrorText>}
          </ul>
        )}
      </MenuPopover>
      <button type="button" aria-label="New session" onClick={onReset} disabled={view?.kind === "new"} className={cn(ICON_BUTTON, "disabled:opacity-40")}>
        <Plus aria-hidden className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
