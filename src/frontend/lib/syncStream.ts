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
let source: EventSource | null = null
let lastSeq = 0
let reconnectDelayMs = 1000
let reconnectTimer: ReturnType<typeof setTimeout> | null = null

const MAX_RECONNECT_DELAY_MS = 30_000

function connect(): void {
  if (source || typeof window === "undefined" || typeof EventSource === "undefined") return
  const stream = new EventSource(`${getApiBase()}/sync/stream?since=${lastSeq}`)
  source = stream

  stream.addEventListener("change", (message) => {
    let event: ChangeEvent
    try {
      event = JSON.parse((message as MessageEvent<string>).data)
    } catch {
      return
    }
    // The server replays by sequence, so this cursor is what makes a reconnect lossless.
    if (event.seq > lastSeq) lastSeq = event.seq
    reconnectDelayMs = 1000
    // A device never reacts to its own write; it already holds that state.
    if (event.device_id && event.device_id === getDeviceId()) return
    for (const listener of listeners) listener(event)
  })

  stream.onerror = () => {
    stream.close()
    source = null
    if (listeners.size === 0 || reconnectTimer) return
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      connect()
    }, reconnectDelayMs)
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, MAX_RECONNECT_DELAY_MS)
  }
}

/** Subscribe to this user's change feed. Returns an unsubscribe function. */
export function subscribeToChanges(listener: Listener): () => void {
  listeners.add(listener)
  connect()
  return () => {
    listeners.delete(listener)
    if (listeners.size > 0) return
    source?.close()
    source = null
    clearTimeout(reconnectTimer ?? undefined)
    reconnectTimer = null
  }
}
