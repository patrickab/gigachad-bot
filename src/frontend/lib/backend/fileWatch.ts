"use client"

import { announceFileChange } from "./fileEvents"
import { fileWatchUrl } from "./urls"

/** What one registration watches: absolute files and folders, a folder covers every file below it. */
export interface WatchTargets {
  files: readonly string[]
  folders: readonly string[]
}

// Registrations arriving within this window go out as one `watch` frame.
const SET_DEBOUNCE_MS = 50
const FIRST_RECONNECT_DELAY_MS = 1000
const MAX_RECONNECT_DELAY_MS = 30_000

// How many registrations hold each path. A path is watched while it has a holder.
const fileHolders = new Map<string, number>()
const folderHolders = new Map<string, number>()

let socket: WebSocket | null = null
// The last `watch` frame this socket was sent, so an unchanged set sends nothing.
let sentFrame: string | null = null
// The socket dropped while paths were watched, so the next open owes every watched path a catch-up.
let dropped = false
let flushTimer: ReturnType<typeof setTimeout> | undefined
let reconnectTimer: ReturnType<typeof setTimeout> | undefined
let reconnectDelayMs = FIRST_RECONNECT_DELAY_MS

function hold(holders: Map<string, number>, paths: Iterable<string>): void {
  for (const path of paths) holders.set(path, (holders.get(path) ?? 0) + 1)
}

function drop(holders: Map<string, number>, paths: Iterable<string>): void {
  for (const path of paths) {
    const remaining = (holders.get(path) ?? 0) - 1
    if (remaining > 0) holders.set(path, remaining)
    else holders.delete(path)
  }
}

function sendSet(current: WebSocket): void {
  const frame = JSON.stringify({ type: "watch", files: [...fileHolders.keys()].sort(), folders: [...folderHolders.keys()].sort() })
  if (frame === sentFrame) return
  current.send(frame)
  sentFrame = frame
}

/** Report every watched path once: files by their own path, folders by the folder path (something below may have changed). */
function announceWatched(): void {
  for (const path of [...fileHolders.keys(), ...folderHolders.keys()]) announceFileChange(path)
}

function parseChanged(data: unknown): string | null {
  if (typeof data !== "string") return null
  try {
    const frame = JSON.parse(data) as { type?: unknown; path?: unknown }
    return frame.type === "changed" && typeof frame.path === "string" ? frame.path : null
  } catch {
    return null
  }
}

function connect(): void {
  const current = new WebSocket(fileWatchUrl())
  socket = current
  current.onopen = () => {
    reconnectDelayMs = FIRST_RECONNECT_DELAY_MS
    sendSet(current)
    // Writes during the outage were missed: the frame goes first so the readers this triggers start after the watch.
    if (dropped) {
      dropped = false
      announceWatched()
    }
  }
  current.onmessage = (event) => {
    const path = parseChanged(event.data)
    if (path !== null) announceFileChange(path)
  }
  current.onclose = () => {
    if (socket !== current) return
    socket = null
    sentFrame = null
    dropped = true
    reconnectTimer = setTimeout(() => {
      reconnectTimer = undefined
      flush()
    }, reconnectDelayMs)
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, MAX_RECONNECT_DELAY_MS)
  }
}

/** Close the socket and forget its state: nothing is watched, so the server holds nothing either. */
function stop(): void {
  clearTimeout(reconnectTimer)
  reconnectTimer = undefined
  dropped = false
  sentFrame = null
  reconnectDelayMs = FIRST_RECONNECT_DELAY_MS
  const closing = socket
  socket = null
  closing?.close()
}

function flush(): void {
  if (fileHolders.size === 0 && folderHolders.size === 0) stop()
  else if (!socket && !reconnectTimer) connect()
  else if (socket?.readyState === WebSocket.OPEN) sendSet(socket)
}

function scheduleFlush(): void {
  clearTimeout(flushTimer)
  flushTimer = setTimeout(flush, SET_DEBOUNCE_MS)
}

/**
 * Follow the disk changes of absolute files and folders until the returned function is called.
 *
 * - One lazy WebSocket per browser tab serves every registration. It opens on the first and closes after the last leaves
 * - A path held by several registrations is sent once and unwatched only when the last one leaves
 * - Each change goes to the `onFileChange` listeners, a folder watch reports the changed file inside it
 * - After a reconnect every watched path is reported once more, since writes during the outage were missed
 */
export function watchFiles({ files, folders }: WatchTargets): () => void {
  const watchedFiles = new Set(files)
  const watchedFolders = new Set(folders)
  hold(fileHolders, watchedFiles)
  hold(folderHolders, watchedFolders)
  scheduleFlush()
  let released = false
  return () => {
    if (released) return
    released = true
    drop(fileHolders, watchedFiles)
    drop(folderHolders, watchedFolders)
    scheduleFlush()
  }
}
