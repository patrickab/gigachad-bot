"use client"

/**
 * The app's Pi session store, mounted once in `AppShell` as `<PiSessionsProvider>`.
 *
 * - Lists every repo's sessions from `GET /api/pi/sessions` and filters by profile in the browser (the coding sidebar and
 *   picker ask for `coding`, G5). Lists refresh on the host frame `sessions_changed` (M25), on window focus, on socket
 *   reopen and every 30 s, while any `usePiSessions` is mounted
 * - Status per session: `running`/`waiting`/`failed`/`contextPercent`/`cost` from the live session of a session this page
 *   attached (`liveSession.ts`), which wins, else `running`/`waiting` from the listed `live`
 * - `unread`: a run settled since the user last viewed the session (`markPiSessionRead`), persisted in localStorage
 * - `cachedUntil`: the listed `cachedUntil` (pi-host reads it from the session file), lists read again once a run settles
 *   so it follows each answer
 * - `piSessionEvents`: the run events of the attached sessions (`observeLiveSessions`)
 */
import { createElement, Fragment, useEffect, useMemo, useSyncExternalStore, type ReactNode } from "react"
import { fetchPiRepositories, fetchPiSessions, searchPiSessions as fetchSearchHits, type PiLiveStatus, type PiProfile, type PiRepository, type PiSearchHit, type PiSessionInfo } from "@/lib/backend/api"
import { readStorage, writeStorage } from "@/lib/storage"
import { liveStatus, observeLiveSessions, type PiSessionEvent } from "./liveSession"
import { observePiSocket, type HostEvent } from "./piSocket"

export type { PiSearchHit, PiSessionEvent }

/** Live status of one session for sidebar rows, the picker and tab titles. */
export interface PiSessionStatus {
  running: boolean
  waiting: boolean
  unread: boolean
  /** The last run of an attached session ended in an error. */
  failed: boolean
  /** Until when the prompt cache is likely warm (epoch ms), absent when the last answer used none. */
  cachedUntil?: number
  contextPercent?: number
  cost?: number
}

type PiSessionListener = (event: PiSessionEvent) => void

interface ListState {
  sessions: PiSessionInfo[]
  loading: boolean
  error: string | null
}

interface SessionList {
  state: ListState
  /** Mounted `usePiSessions` hooks reading this list. */
  watchers: number
  reading: boolean
  /** A refresh was asked for while reading: read again once done. */
  again: boolean
}

const POLL_MS = 30_000
const UNREAD_KEY = "pi-unread-sessions"
const IDLE_STATUS: PiSessionStatus = { running: false, waiting: false, unread: false, failed: false }
const NO_REPOSITORIES: PiRepository[] = []

const eventListeners = new Set<PiSessionListener>()
const listListeners = new Set<() => void>()
const statusListeners = new Set<() => void>()
const list: SessionList = { state: { sessions: [], loading: true, error: null }, watchers: 0, reading: false, again: false }
let repositories: PiRepository[] = NO_REPOSITORIES
let repositoriesRead: Promise<void> | null = null
let stopLive: (() => void) | null = null
let listedLive = new Map<string, PiLiveStatus | null>()
/** The listed `cachedUntil` of each session that has one, epoch ms. */
let listedCache = new Map<string, number>()
let unread = readUnread()
const statuses = new Map<string, PiSessionStatus>()

function readRepositories(): Promise<void> {
  repositoriesRead ??= fetchPiRepositories().then(
    (read) => { repositories = read },
    (error: unknown) => {
      repositoriesRead = null
      throw error
    },
  )
  return repositoriesRead
}

/** Read `list` again. Asks during a read coalesce into one more read after it. */
async function readList(list: SessionList): Promise<void> {
  if (list.reading) {
    list.again = true
    return
  }
  list.reading = true
  do {
    list.again = false
    try {
      const [sessions] = await Promise.all([fetchPiSessions(), readRepositories()])
      const { state } = list
      // An unchanged answer keeps the list's identity: nothing re-renders (compared as `ProjectProvider` does)
      if (!state.loading && state.error === null && sessions.length === state.sessions.length && JSON.stringify(sessions) === JSON.stringify(state.sessions)) continue
      noteListed(sessions)
      list.state = { sessions, loading: false, error: null }
    } catch (error) {
      list.state = { ...list.state, loading: false, error: error instanceof Error ? error.message : String(error) }
    }
    for (const listener of listListeners) listener()
  } while (list.again)
  list.reading = false
}

function refreshWatched() {
  if (list.watchers > 0) void readList(list)
}

/** Every path is a factory task's file, which no list holds. No paths means an unknown change. */
function onlyFactoryFiles(paths: unknown): boolean {
  return Array.isArray(paths) && paths.length > 0
    && paths.every((path) => typeof path === "string" && (path.includes("/factory/sessions/") || path.includes("/factory/tasks/")))
}

/** Keep the watched lists fresh: host `sessions_changed`, socket reopen, focus, visibility and a poll. */
function startLive(): () => void {
  const stopObserving = observePiSocket({
    onHostEvent: (event: HostEvent) => {
      if (event.type === "sessions_changed" && !onlyFactoryFiles(event.sessionPaths)) refreshWatched()
    },
    onOpen: refreshWatched,
  }, { hold: true })
  const refreshWhenVisible = () => {
    if (document.visibilityState === "visible") refreshWatched()
  }
  const poll = setInterval(refreshWatched, POLL_MS)
  window.addEventListener("focus", refreshWatched)
  document.addEventListener("visibilitychange", refreshWhenVisible)
  return () => {
    stopObserving()
    clearInterval(poll)
    window.removeEventListener("focus", refreshWatched)
    document.removeEventListener("visibilitychange", refreshWhenVisible)
  }
}

function watchList(list: SessionList): () => void {
  list.watchers += 1
  stopLive ??= startLive()
  void readList(list)
  return () => {
    list.watchers -= 1
    if (list.watchers > 0) return
    stopLive?.()
    stopLive = null
  }
}

function subscribeLists(listener: () => void): () => void {
  listListeners.add(listener)
  return () => listListeners.delete(listener)
}

/** The profile of `session`: one written before profiles existed is a coding session (G5). */
export function piSessionProfile(session: Pick<PiSessionInfo, "profile">): PiProfile {
  return session.profile ?? "coding"
}

/** Sessions of `profile` (any when omitted), kept live. */
export function usePiSessions(filter: { profile?: PiProfile } = {}): { sessions: PiSessionInfo[]; loading: boolean; error: string | null } {
  const { profile } = filter
  const state = useSyncExternalStore(subscribeLists, () => list.state, () => list.state)
  useEffect(() => watchList(list), [])
  const sessions = useMemo(
    () => profile === undefined ? state.sessions : state.sessions.filter((session) => piSessionProfile(session) === profile),
    [state.sessions, profile],
  )
  return { sessions, loading: state.loading, error: state.error }
}

/** Read every watched list again, for changes pi-host does not announce with `sessions_changed` (a pin writes no session file). */
export function refreshPiSessions(): void {
  refreshWatched()
}

/** The checkouts under `~/git` (`/api/pi/repos`), read along with the session lists: empty until a `usePiSessions` read them. */
export function usePiRepositories(): PiRepository[] {
  return useSyncExternalStore(subscribeLists, () => repositories, () => NO_REPOSITORIES)
}

function readUnread(): Set<string> {
  try {
    const stored: unknown = JSON.parse(readStorage(UNREAD_KEY) ?? "[]")
    return new Set(Array.isArray(stored) ? stored.filter((id): id is string => typeof id === "string") : [])
  } catch {
    return new Set()
  }
}

function markUnread(sessionId: string) {
  if (unread.has(sessionId)) return
  unread.add(sessionId)
  writeStorage(UNREAD_KEY, JSON.stringify([...unread]))
}

/** Clear the unread mark of `sessionId`, e.g. while it is in view. */
export function markPiSessionRead(sessionId: string): void {
  if (!unread.delete(sessionId)) return
  writeStorage(UNREAD_KEY, JSON.stringify([...unread]))
  statusChanged()
}

function computeStatus(sessionId: string): PiSessionStatus {
  const session = liveStatus(sessionId)
  const live = listedLive.get(sessionId)
  const running = session ? session.running : live?.running ?? false
  const waiting = session ? session.waiting : live?.waiting ?? false
  const status: PiSessionStatus = { running, waiting, unread: !running && unread.has(sessionId), failed: !running && Boolean(session?.failed) }
  const cachedUntil = listedCache.get(sessionId)
  if (cachedUntil !== undefined) status.cachedUntil = cachedUntil
  if (session?.contextPercent !== undefined) status.contextPercent = session.contextPercent
  if (session?.cost !== undefined) status.cost = session.cost
  return status
}

function sameStatus(a: PiSessionStatus, b: PiSessionStatus): boolean {
  return a.running === b.running && a.waiting === b.waiting && a.unread === b.unread && a.failed === b.failed && a.cachedUntil === b.cachedUntil
    && a.contextPercent === b.contextPercent && a.cost === b.cost
}

function statusOf(sessionId: string): PiSessionStatus {
  let status = statuses.get(sessionId)
  if (!status) {
    status = computeStatus(sessionId)
    statuses.set(sessionId, status)
  }
  return status
}

/** Recompute every status read so far, keeping unchanged ones identical for `useSyncExternalStore`. */
function statusChanged() {
  for (const [sessionId, previous] of statuses) {
    const next = computeStatus(sessionId)
    if (!sameStatus(previous, next)) statuses.set(sessionId, next)
  }
  for (const listener of statusListeners) listener()
}

function subscribeStatus(listener: () => void): () => void {
  statusListeners.add(listener)
  return () => statusListeners.delete(listener)
}

/** `summarize` of the live statuses of `sessionIds`, re-read as any changes: a string, so unchanged summaries stay identical. */
export function usePiSessionsSummary(sessionIds: readonly string[], summarize: (statuses: PiSessionStatus[]) => string): string {
  const snapshot = () => summarize(sessionIds.map(statusOf))
  return useSyncExternalStore(subscribeStatus, snapshot, snapshot)
}

/** Live status of `sessionId`, idle and read for null. */
export function usePiSessionStatus(sessionId: string | null): PiSessionStatus {
  const snapshot = () => sessionId === null ? IDLE_STATUS : statusOf(sessionId)
  return useSyncExternalStore(subscribeStatus, snapshot, snapshot)
}

/** A listed session that stopped running while this page did not watch its frames settled unseen. */
function noteListed(sessions: PiSessionInfo[]) {
  const previous = listedLive
  listedLive = new Map(sessions.map((session) => [session.sessionId, session.live]))
  for (const session of sessions) {
    const until = session.cachedUntil ? Date.parse(session.cachedUntil) : NaN
    if (Number.isNaN(until)) listedCache.delete(session.sessionId)
    else listedCache.set(session.sessionId, until)
  }
  listedCache = new Map(listedCache)
  for (const [sessionId, live] of listedLive) {
    if (previous.get(sessionId)?.running && !live?.running && liveStatus(sessionId) === null) markUnread(sessionId)
  }
  statusChanged()
}

/** A run of an attached session settled: unread, and its listing carries the new cache time. Every event goes to `piSessionEvents`. */
function onEvent(event: PiSessionEvent) {
  if (event.type === "settled") {
    markUnread(event.sessionId)
    refreshWatched()
    statusChanged()
  }
  for (const listener of eventListeners) listener(event)
}

/** Mounted once in `AppShell`: follows the run state of every session the page attaches, and unread marks other tabs set. */
export function PiSessionsProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    const stopObserving = observeLiveSessions({ onStatus: statusChanged, onEvent })
    const syncUnread = (event: StorageEvent) => {
      if (event.key !== UNREAD_KEY) return
      unread = readUnread()
      statusChanged()
    }
    window.addEventListener("storage", syncUnread)
    return () => {
      stopObserving()
      window.removeEventListener("storage", syncUnread)
    }
  }, [])
  return createElement(Fragment, null, children)
}

/** Sessions of every repo (of `profile` when given) whose entries match `query`, none for a blank query. */
export async function searchPiSessions(query: string, filter: { profile?: PiProfile } = {}, signal?: AbortSignal): Promise<PiSearchHit[]> {
  const trimmed = query.trim()
  if (!trimmed) return []
  const hits = await fetchSearchHits(trimmed, signal)
  return filter.profile === undefined ? hits : hits.filter((hit) => piSessionProfile(hit.session) === filter.profile)
}

/** Run events of every session this page attached, for notifications. */
export const piSessionEvents = {
  subscribe(listener: PiSessionListener): () => void {
    eventListeners.add(listener)
    return () => eventListeners.delete(listener)
  },
}
