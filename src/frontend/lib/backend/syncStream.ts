"use client"

import { getApiBase } from "./config"
import { getDeviceId } from "./deviceId"

export interface ChangeEvent {
  seq: number
  resource_kind: "document" | "asset" | "vault_root"
  resource_key: string
  version: number | null
  device_id: string | null
}

type Listener = (event: ChangeEvent) => void

const listeners = new Set<Listener>()
// One shared EventSource for all subscribers. Closed when the last one unsubscribes.
let source: EventSource | null = null
// Null until the first connection announces the head, so a fresh page replays nothing.
let lastSeq: number | null = null
let reconnectDelayMs = 1000
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let cursorReady: Promise<void> | null = null
let markCursorReady = () => {}
let cursorTimer: ReturnType<typeof setTimeout> | undefined

const MAX_RECONNECT_DELAY_MS = 30_000
// Startup reads wait at most this long for the head. Past it they run, and the next connection replays everything.
const CURSOR_TIMEOUT_MS = 5_000

function releaseCursor(): void {
  clearTimeout(cursorTimer)
  markCursorReady()
}

function closeIfIdle(): void {
  // A stream still owing the head stays open for the startup reads waiting on it.
  if (listeners.size > 0 || lastSeq === null) return
  source?.close()
  source = null
  clearTimeout(reconnectTimer ?? undefined)
  reconnectTimer = null
}

function disconnect(stream: EventSource): void {
  stream.close()
  if (source === stream) source = null
  // Reads may now run before any head is known, so the next connection replays from the start.
  if (lastSeq === null) lastSeq = 0
  releaseCursor()
  if (listeners.size === 0 || reconnectTimer) return
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null
    connect()
  }, reconnectDelayMs)
  reconnectDelayMs = Math.min(reconnectDelayMs * 2, MAX_RECONNECT_DELAY_MS)
}

function connect(): void {
  if (source || typeof window === "undefined" || typeof EventSource === "undefined") return
  const stream = new EventSource(`${getApiBase()}/sync/stream${lastSeq === null ? "" : `?since=${lastSeq}`}`)
  source = stream
  if (lastSeq === null) cursorTimer = setTimeout(() => disconnect(stream), CURSOR_TIMEOUT_MS)

  stream.addEventListener("cursor", (message) => {
    let seq: number
    try {
      seq = (JSON.parse((message as MessageEvent<string>).data) as { seq: number }).seq
    } catch {
      return
    }
    if (lastSeq === null) lastSeq = seq
    releaseCursor()
    closeIfIdle()
  })

  stream.addEventListener("change", (message) => {
    let event: ChangeEvent
    try {
      event = JSON.parse((message as MessageEvent<string>).data)
    } catch {
      return
    }
    // The server replays by sequence, so this cursor is what makes a reconnect lossless.
    if (lastSeq === null || event.seq > lastSeq) lastSeq = event.seq
    // Any event means the server subscribed first, so reads from here on miss nothing (a server without the head event replays instead).
    releaseCursor()
    reconnectDelayMs = 1000
    // A device never reacts to its own write; it already holds that state.
    if (event.device_id && event.device_id === getDeviceId()) return
    for (const listener of listeners) listener(event)
  })

  stream.onerror = () => disconnect(stream)
}

/**
 * Resolve once the change feed knows the head it starts from, opening the feed if needed.
 *
 * - Await it before reading state that change events refresh: every change after the read then arrives as an event
 * - Resolves at once without EventSource (tests, server rendering), and after a failed or slow first connection
 */
export function syncCursorReady(): Promise<void> {
  if (!cursorReady) {
    cursorReady = new Promise((resolve) => { markCursorReady = resolve })
    connect()
    if (!source) releaseCursor()
  }
  return cursorReady
}

/** Subscribe to this user's change feed. Returns an unsubscribe function. */
export function subscribeToChanges(listener: Listener): () => void {
  listeners.add(listener)
  void syncCursorReady()
  connect()
  return () => {
    listeners.delete(listener)
    closeIfIdle()
  }
}
