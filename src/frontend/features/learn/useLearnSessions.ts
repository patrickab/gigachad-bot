import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { ApiError, assignLearnSession, deleteLearnSession, fetchLearnSessions, renameLearnSession, type LearnDeleteMode } from "@/lib/backend/api"
import type { LearnSession } from "@/lib/domain/types"

const SESSIONS_POLL_MS = 15_000

export interface LearnSessions {
  /** Every learn session with its project, empty until the first read. */
  sessions: LearnSession[]
  /** Why the last read or assignment failed. LearnHarness being down (503) stays quiet, the list keeps the last good read. */
  error: string | null
  refresh: () => Promise<void>
  /** Group session `sessionId` under `projectSlug` unless it already belongs to a project. */
  claim: (sessionId: string, projectSlug: string) => Promise<void>
  /** Read the list for a session it does not hold yet, at most once per id. */
  lookUp: (sessionId: string) => Promise<void>
  /** Sessions deleted here, or looked up and missing from a successful read (deleted in LearnHarness). */
  missing: ReadonlySet<string>
  /** Rename a session in LearnHarness, throwing its refusal. */
  rename: (sessionId: string, title: string) => Promise<void>
  /** Delete a session in LearnHarness, its Knowledge kept unless `mode` removes its graph traces, throwing its refusal (a running turn). */
  remove: (sessionId: string, mode?: LearnDeleteMode) => Promise<void>
  /** Show `title` for a listed session at once, e.g. a rename inside the frame the next read would bring. */
  retitle: (sessionId: string, title: string) => void
}

function messageOf(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause)
}

/**
 * The learn sessions, read while `enabled` (host logins with a LearnHarness): on start and on focus, and every 15 s
 * while it is useful: the page is visible, LearnHarness answered the last read, and the user is `watching` the learn
 * frame (sessions start and get renamed there) or a session is busy (its turn is still running).
 */
export function useLearnSessions(enabled: boolean, watching: boolean): LearnSessions {
  const [sessions, setSessions] = useState<LearnSession[]>([])
  const [error, setError] = useState<string | null>(null)
  const [reachable, setReachable] = useState(true)
  const [missing, setMissing] = useState<ReadonlySet<string>>(() => new Set())
  const lookedUp = useRef(new Set<string>())
  const busy = sessions.some((session) => session.busy)

  const read = useCallback(async (): Promise<LearnSession[] | null> => {
    try {
      const next = await fetchLearnSessions()
      setSessions(next)
      setError(null)
      setReachable(true)
      return next
    } catch (cause) {
      setError(cause instanceof ApiError && cause.status === 503 ? null : messageOf(cause))
      setReachable(false)
      return null
    }
  }, [])

  const refresh = useCallback(async () => { await read() }, [read])

  useEffect(() => {
    if (!enabled) return
    const first = window.setTimeout(() => void read(), 0)
    const onFocus = () => void read()
    window.addEventListener("focus", onFocus)
    return () => {
      window.clearTimeout(first)
      window.removeEventListener("focus", onFocus)
    }
  }, [enabled, read])

  // An unreachable LearnHarness is read again on focus only, and an idle list not at all until then.
  const polling = enabled && reachable && (watching || busy)
  useEffect(() => {
    if (!polling) return
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void read()
    }, SESSIONS_POLL_MS)
    return () => window.clearInterval(timer)
  }, [polling, read])

  const claim = useCallback(async (sessionId: string, projectSlug: string) => {
    // Decided on a fresh read: a stale list must never move a session out of the project it already joined.
    const session = (await read())?.find((candidate) => candidate.id === sessionId)
    if (!session || session.project_slug !== null) return
    try {
      await assignLearnSession(sessionId, projectSlug)
    } catch (cause) {
      setError(messageOf(cause))
      return
    }
    await read()
  }, [read])

  const lookUp = useCallback(async (sessionId: string) => {
    if (!enabled || lookedUp.current.has(sessionId)) return
    lookedUp.current.add(sessionId)
    const next = await read()
    if (next && !next.some((candidate) => candidate.id === sessionId)) setMissing((current) => new Set(current).add(sessionId))
  }, [enabled, read])

  const retitle = useCallback((sessionId: string, title: string) => {
    setSessions((current) => current.some((session) => session.id === sessionId && session.title !== title)
      ? current.map((session) => session.id === sessionId ? { ...session, title } : session)
      : current)
  }, [])

  const rename = useCallback(async (sessionId: string, title: string) => {
    retitle(sessionId, await renameLearnSession(sessionId, title))
  }, [retitle])

  const remove = useCallback(async (sessionId: string, mode?: LearnDeleteMode) => {
    await deleteLearnSession(sessionId, mode)
    setSessions((current) => current.filter((session) => session.id !== sessionId))
    setMissing((current) => new Set(current).add(sessionId))
  }, [])

  return useMemo(
    () => ({ sessions: enabled ? sessions : [], error: enabled ? error : null, refresh, claim, lookUp, missing, rename, remove, retitle }),
    [enabled, sessions, error, refresh, claim, lookUp, missing, rename, remove, retitle],
  )
}
