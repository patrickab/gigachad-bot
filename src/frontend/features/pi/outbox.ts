import { useSyncExternalStore } from "react"
import type { ImageContent } from "./types"

/**
 * Messages sent while a run streams, held in browser memory until pi-host confirms them (queue snapshot or the user message in the session).
 *
 * - `sending`: not confirmed yet. Not `dispatched` means the socket was down, it is flushed in order on reconnect
 * - `queued`: Pi accepted it as `mode`, its confirming event is still to come
 * - `failed` ("not sent"): refused, or its delivery is uncertain after a drop. Never resent except by the user's Retry
 * - Plain module state per session, it outlives a remounted tab and dies with the page
 */
export type HeldState = "sending" | "queued" | "failed"
export type HeldMode = "steer" | "followUp"

export interface HeldMessage {
  id: string
  text: string
  images: ImageContent[]
  mode: HeldMode
  state: HeldState
  /** The prompt command went out on the socket, its answer is outstanding or in. */
  dispatched: boolean
  createdAt: number
  error?: string
}

const NONE: readonly HeldMessage[] = []
const held = new Map<string, readonly HeldMessage[]>()
const listeners = new Set<() => void>()
let nextId = 0

function set(sessionId: string, entries: readonly HeldMessage[]) {
  if (entries.length === 0) held.delete(sessionId)
  else held.set(sessionId, entries)
  for (const listener of listeners) listener()
}

export function heldOf(sessionId: string): readonly HeldMessage[] {
  return held.get(sessionId) ?? NONE
}

/** Hold a message, oldest first. */
export function hold(sessionId: string, message: Pick<HeldMessage, "text" | "images" | "mode">): HeldMessage {
  const entry: HeldMessage = { ...message, id: `held-${++nextId}`, state: "sending", dispatched: false, createdAt: Date.now() }
  set(sessionId, [...heldOf(sessionId), entry])
  return entry
}

/** Change a held message, a no-op once it is gone. */
export function patchHeld(sessionId: string, id: string, patch: Partial<Omit<HeldMessage, "id">>) {
  const entries = heldOf(sessionId)
  if (!entries.some((entry) => entry.id === id)) return
  set(sessionId, entries.map((entry) => entry.id === id ? { ...entry, ...patch } : entry))
}

export function removeHeld(sessionId: string, ids: readonly string[]) {
  const entries = heldOf(sessionId)
  if (!entries.some((entry) => ids.includes(entry.id))) return
  set(sessionId, entries.filter((entry) => !ids.includes(entry.id)))
}

/** Empty the session's held messages and return them. */
export function takeHeld(sessionId: string): readonly HeldMessage[] {
  const entries = heldOf(sessionId)
  if (entries.length > 0) set(sessionId, [])
  return entries
}

/** The ids of `entries` that `queue` (Pi's queue snapshot) or `userTexts` (user messages sent at or after `since`, one text per message) confirm. Each queue item or message confirms one entry. */
export function confirmedIds(entries: readonly HeldMessage[], queue: { steering: string[]; followUp: string[] }, userTexts: (since: number) => string[]): string[] {
  const queued = [...queue.steering, ...queue.followUp]
  const seen: string[][] = []
  const ids: string[] = []
  for (const entry of entries) {
    const at = queued.indexOf(entry.text)
    if (at >= 0) {
      queued.splice(at, 1)
      ids.push(entry.id)
      continue
    }
    const texts = userTexts(entry.createdAt)
    const used = seen.flat().filter((text) => text === entry.text).length
    if (texts.filter((text) => text === entry.text).length > used) {
      seen.push([entry.text])
      ids.push(entry.id)
    }
  }
  return ids
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/** The held messages of `sessionId`, re-rendering on change. */
export function useHeld(sessionId: string): readonly HeldMessage[] {
  return useSyncExternalStore(subscribe, () => heldOf(sessionId), () => NONE)
}

/** Test reset. */
export function clearOutbox() {
  held.clear()
  for (const listener of listeners) listener()
}
