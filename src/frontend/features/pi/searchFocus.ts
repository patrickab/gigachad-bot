/**
 * The session search hit a Pi transcript is showing: the picker sets it, the transcript scrolls to and paints it,
 * next/previous and clear change it. One at a time, for the session it names.
 *
 * - A pick starts with the hits the search listed, then the session's whole ranked list replaces them
 * - `seq` is new on every jump, so the transcript scrolls once per jump and not on every render
 */
import { useSyncExternalStore } from "react"
import { fetchPiSessionMatches, type PiSessionMatch } from "@/lib/backend/api"

/** Hits read for navigation, the best of a session's. */
const NAVIGATION_HITS = 200

export interface PiSearchFocus {
  sessionId: string
  query: string
  /** Ranked hits of the session, `index` the one shown. */
  matches: PiSessionMatch[]
  index: number
  seq: number
}

let focus: PiSearchFocus | null = null
let seq = 0
const listeners = new Set<() => void>()

function publish(next: PiSearchFocus | null): void {
  focus = next
  for (const listener of listeners) listener()
}

const sameHit = (a: PiSessionMatch, b: PiSessionMatch) => a.entryId === b.entryId && a.ordinal === b.ordinal

/** Show `hit` of `sessionId` for `query`; `listed` are the hits the search already has, the whole list follows. */
export function focusPiSearchHit(sessionId: string, query: string, hit: PiSessionMatch, listed: readonly PiSessionMatch[] = [hit]): void {
  const matches = listed.some((other) => sameHit(other, hit)) ? [...listed] : [hit, ...listed]
  const mine = ++seq
  publish({ sessionId, query, matches, index: matches.findIndex((other) => sameHit(other, hit)), seq: mine })
  fetchPiSessionMatches(sessionId, query, NAVIGATION_HITS).then(({ matches: ranked }) => {
    // Another pick or a clear since: that one stands
    if (focus?.sessionId !== sessionId || focus.query !== query || !sameHit(focus.matches[focus.index], hit)) return
    const all = ranked.some((other) => sameHit(other, hit)) ? ranked : [hit, ...ranked]
    // The jump is already done: the list grows under the shown hit without a new `seq`
    publish({ ...focus, matches: all, index: all.findIndex((other) => sameHit(other, hit)) })
  }, () => undefined)
}

/** The next (`1`) or previous (`-1`) hit of the shown session, wrapping. */
export function stepPiSearchHit(delta: 1 | -1): void {
  if (!focus || focus.matches.length === 0) return
  const count = focus.matches.length
  publish({ ...focus, index: (focus.index + delta + count) % count, seq: ++seq })
}

export function clearPiSearchHit(): void {
  if (focus) publish(null)
}

/** The hit shown in `sessionId`, null when the search is not on that session. */
export function usePiSearchFocus(sessionId: string | null): PiSearchFocus | null {
  const current = useSyncExternalStore(
    (listener) => { listeners.add(listener); return () => { listeners.delete(listener) } },
    () => focus,
    () => null,
  )
  return sessionId !== null && current?.sessionId === sessionId ? current : null
}
