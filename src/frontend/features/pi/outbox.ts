import { useSyncExternalStore } from "react"
import { splitSkill, textOf, type Transcript } from "./transcript"
import type { ImageContent, UserMessage } from "./types"

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

/** Pi expands `/skill:name args` into a `<skill>` block before it queues or records it: map that back to what the user typed. */
function typed(text: string): string {
  const { skill, text: args } = splitSkill(text)
  return skill ? `/skill:${skill.name}${args ? ` ${args}` : ""}` : text
}

/** Drop the held messages of `sessionId` that Pi's queue snapshot or a user message sent since (one per message) confirms. Each queue item or message confirms one. */
export function confirmFromTranscript(sessionId: string, { messages, queue }: Pick<Transcript, "messages" | "queue">) {
  const queued = [...queue.steering, ...queue.followUp].map(typed)
  const byMessage: string[] = []
  const ids: string[] = []
  for (const entry of heldOf(sessionId)) {
    const at = queued.indexOf(entry.text)
    if (at >= 0) {
      queued.splice(at, 1)
      ids.push(entry.id)
      continue
    }
    const sent = messages.filter((message) => message.role === "user" && message.timestamp >= entry.createdAt - 2_000 && typed(textOf((message as UserMessage).content)) === entry.text)
    if (sent.length > byMessage.filter((text) => text === entry.text).length) {
      byMessage.push(entry.text)
      ids.push(entry.id)
    }
  }
  removeHeld(sessionId, ids)
}

/** For the draft after an abort: `queue` (Pi's queue it cleared), then the texts of `taken` (the held messages) it does not hold. */
export function takeBack(taken: readonly HeldMessage[], queue: { steering: string[]; followUp: string[] }): string[] {
  const back = [...queue.steering, ...queue.followUp]
  const rest = [...back]
  for (const { text } of taken) {
    const at = rest.indexOf(text)
    if (at >= 0) rest.splice(at, 1)
    else back.push(text)
  }
  return back
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
