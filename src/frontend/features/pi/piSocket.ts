/**
 * The page's one WebSocket to pi-host (`WS /api/pi/ws`), shared by every open Pi session (M18).
 *
 * - Consumes host frames `{type, sessionId, streamId, sequence, ...}` and routes each to the subscribers of its session
 * - Provides `subscribePiSession`: the first subscriber of a session attaches it, the last detaches it
 * - Provides `observePiSocket`: host frames without `sessionId` (`sessions_changed`) and socket reopens, for lists
 * - Reconnects with backoff while anything is subscribed or held; per session a subscriber's `onReopen` decides how to catch up (resync or attach), else it re-attaches
 * - No ack timeouts: liveness never depends on how fast the host answers (M18)
 * - Closes shortly after the last session and hold leave, so a remount (React strict mode, a tab switch) keeps the socket
 */
import { piSocketUrl } from "@/lib/backend/urls"
import type { ClientFrame, HostFrame } from "./protocol"

/** A host frame about no single session, e.g. `{type: "sessions_changed", sessionPaths}`. */
export type HostEvent = { type: string; [field: string]: unknown }

export type SocketStatus = { kind: "connecting" } | { kind: "open" } | { kind: "closed"; reason: string }

export interface SessionSubscriber {
  onFrame: (frame: HostFrame) => void
  onStatus: (status: SocketStatus) => void
  /** The socket (re)opened: send this session's catch-up frames. Without it the socket sends `attach`. */
  onReopen?: () => void
}

/** Sees the socket's traffic without attaching anything. */
export interface SocketObserver {
  /** A host frame without `sessionId`. */
  onHostEvent?: (event: HostEvent) => void
  /** The socket (re)opened, frames sent while it was down are lost. */
  onOpen?: () => void
}

const RETRY_MIN_MS = 500
const RETRY_MAX_MS = 10_000
const IDLE_CLOSE_MS = 2_000

const subscribers = new Map<string, Set<SessionSubscriber>>()
const observers = new Set<SocketObserver>()
/** Observers that keep the socket open with no session attached. */
let holds = 0
let socket: WebSocket | null = null
let status: SocketStatus = { kind: "closed", reason: "Not connected" }
let retryMs = RETRY_MIN_MS
let retryTimer: ReturnType<typeof setTimeout> | undefined
let idleTimer: ReturnType<typeof setTimeout> | undefined

function idle(): boolean {
  return subscribers.size === 0 && holds === 0
}

function setStatus(next: SocketStatus) {
  status = next
  for (const group of subscribers.values()) for (const subscriber of group) subscriber.onStatus(next)
}

function sendRaw(frame: ClientFrame): boolean {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false
  socket.send(JSON.stringify(frame))
  return true
}

function connect() {
  if (socket) return
  setStatus({ kind: "connecting" })
  const opened = new WebSocket(piSocketUrl())
  socket = opened
  opened.onopen = () => {
    retryMs = RETRY_MIN_MS
    setStatus({ kind: "open" })
    for (const [sessionId, group] of subscribers) {
      const owner = [...group].find((subscriber) => subscriber.onReopen)
      if (owner) owner.onReopen!()
      else sendRaw({ type: "attach", sessionId })
    }
    for (const observer of observers) observer.onOpen?.()
  }
  opened.onmessage = (message) => {
    const frame = JSON.parse(String(message.data)) as HostEvent
    if (typeof frame.sessionId !== "string") {
      for (const observer of observers) observer.onHostEvent?.(frame)
      return
    }
    const group = subscribers.get(frame.sessionId)
    if (!group) return
    for (const subscriber of group) subscriber.onFrame(frame as HostFrame)
  }
  opened.onclose = (event) => {
    socket = null
    if (idle()) return setStatus({ kind: "closed", reason: event.reason || "Disconnected" })
    setStatus({ kind: "connecting" })
    retryTimer = setTimeout(() => {
      retryTimer = undefined
      connect()
    }, retryMs)
    retryMs = Math.min(retryMs * 2, RETRY_MAX_MS)
  }
}

function closeWhenIdle() {
  clearTimeout(idleTimer)
  idleTimer = setTimeout(() => {
    idleTimer = undefined
    if (!idle()) return
    clearTimeout(retryTimer)
    retryTimer = undefined
    const closing = socket
    socket = null
    if (closing) {
      closing.onclose = null
      closing.close()
    }
    status = { kind: "closed", reason: "Not connected" }
  }, IDLE_CLOSE_MS)
}

/** Watch one session: attaches it (now or once connected) and reports frames and socket status. Returns the unsubscribe. */
export function subscribePiSession(sessionId: string, subscriber: SessionSubscriber): () => void {
  clearTimeout(idleTimer)
  idleTimer = undefined
  let group = subscribers.get(sessionId)
  if (!group) {
    group = new Set()
    subscribers.set(sessionId, group)
    sendRaw({ type: "attach", sessionId })
  }
  group.add(subscriber)
  if (socket) subscriber.onStatus(status)
  else connect()
  return () => {
    const current = subscribers.get(sessionId)
    if (!current) return
    current.delete(subscriber)
    if (current.size > 0) return
    subscribers.delete(sessionId)
    sendRaw({ type: "detach", sessionId })
    if (idle()) closeWhenIdle()
  }
}

/** Watch the socket's traffic. With `hold` the socket also stays connected while nothing is attached, so host events keep arriving. Returns the unobserve. */
export function observePiSocket(observer: SocketObserver, { hold = false }: { hold?: boolean } = {}): () => void {
  observers.add(observer)
  if (hold) {
    holds += 1
    clearTimeout(idleTimer)
    idleTimer = undefined
    connect()
  }
  let observing = true
  return () => {
    if (!observing) return
    observing = false
    observers.delete(observer)
    if (!hold) return
    holds -= 1
    if (idle()) closeWhenIdle()
  }
}

/** Forward one raw Pi RPC command to the session. False while the socket is down. */
export function sendPiCommand(sessionId: string, command: Record<string, unknown>): boolean {
  return sendRaw({ type: "command", sessionId, command })
}

/** Send a session frame (`resync`, `attach`, `detach`). False while the socket is down. */
export function sendPiFrame(frame: ClientFrame): boolean {
  return sendRaw(frame)
}
