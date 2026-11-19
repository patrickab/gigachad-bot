"use client"

import { createElement, Fragment, type ReactNode } from "react"
import type { PiSessionInfo } from "@/lib/backend/api"

/** A session matching a search, with the matched text and the entry it sits in (pi-host `GET /search`). */
export interface PiSearchHit {
  session: PiSessionInfo
  snippet: string
  entryId: string
}

/** Live status of one session for sidebar rows, the picker and tab titles. */
export interface PiSessionStatus {
  running: boolean
  waiting: boolean
  unread: boolean
  contextPercent?: number
  cost?: number
}

/** A run of a session ended, waits on a dialog, or failed (for notifications). */
export interface PiSessionEvent {
  type: "settled" | "waiting" | "failed"
  sessionId: string
  aborted?: boolean
}

type PiSessionListener = (event: PiSessionEvent) => void

const IDLE_STATUS: PiSessionStatus = { running: false, waiting: false, unread: false }
const listeners = new Set<PiSessionListener>()

/** The app's Pi session store, mounted once in `AppShell`. Stub: holds no state yet. */
export function PiSessionsProvider({ children }: { children: ReactNode }) {
  return createElement(Fragment, null, children)
}

/** Sessions of `repo` (all repos when omitted), the archived ones with `archived`, kept live. Stub: always empty. */
export function usePiSessions(filter: { repo?: string; archived?: boolean } = {}): { sessions: PiSessionInfo[]; loading: boolean; error: string | null } {
  void filter
  return { sessions: [], loading: false, error: null }
}

/** Live status of `sessionId`. Stub: always idle and read. */
export function usePiSessionStatus(sessionId: string | null): PiSessionStatus {
  void sessionId
  return IDLE_STATUS
}

/** Clear the unread mark of `sessionId`. Stub: no-op until the store tracks unread sessions. */
export function markPiSessionRead(sessionId: string): void {
  void sessionId
}

/** Sessions whose entries match `query`. Stub: no hits. */
export async function searchPiSessions(query: string): Promise<PiSearchHit[]> {
  void query
  return []
}

/** Run events of every session, for notifications. */
export const piSessionEvents = {
  subscribe(listener: PiSessionListener): () => void {
    listeners.add(listener)
    return () => listeners.delete(listener)
  },
}
