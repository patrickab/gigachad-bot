"use client"

import { useEffect, useMemo, useState } from "react"
import { useStoredJson } from "@/hooks/useStoredJson"
import { prepareLatex } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"

const SESSIONS_STORAGE_KEY = "latex_sessions"

function parseSessions(raw: unknown): Record<string, string> {
  if (typeof raw !== "object" || raw === null) return {}
  return Object.fromEntries(Object.entries(raw).filter((entry): entry is [string, string] => typeof entry[1] === "string"))
}

/** Set up omp's LaTeX workspace beside the `.tex` file at `path`. Gives the directory the chat runs in, or why that failed. */
export function useLatexWorkspace(path: string): { cwd: string | null; error: string | null } {
  const [prepared, setPrepared] = useState<{ cwd: string | null; error: string | null }>({ cwd: null, error: null })
  useEffect(() => {
    let cancelled = false
    prepareLatex(path).then(
      ({ cwd }) => { if (!cancelled) setPrepared({ cwd, error: null }) },
      (error: unknown) => { if (!cancelled) setPrepared({ cwd: null, error: errorMessage(error, "Could not prepare the LaTeX workspace") }) },
    )
    return () => { cancelled = true }
  }, [path])
  return prepared
}

/**
 * The omp view of one `.tex` file's chat: the session its tab item shows, else the one remembered on this device for the file, else a new session in `cwd`.
 *
 * - Null until `cwd` is known and the item shows nothing yet
 * - `adopt` takes the session the frame reports: the item shows it from then on and the file remembers it
 */
export function useLatexView(path: string, itemView: OmpView | undefined, cwd: string | null, onViewChange: (view: OmpView) => void) {
  const [sessions, updateSessions] = useStoredJson(SESSIONS_STORAGE_KEY, parseSessions)
  const remembered = sessions[path]
  const view = useMemo<OmpView | null>(() => {
    if (itemView) return itemView
    if (cwd === null) return null
    return remembered ? { kind: "session", sessionId: remembered } : { kind: "new", cwd }
  }, [itemView, remembered, cwd])

  const adopt = (next: OmpView) => {
    if (next.kind === "session") updateSessions((latest) => ({ ...latest, [path]: next.sessionId }))
    onViewChange(next)
  }
  return { view, adopt }
}
