import type { LearnView } from "@/lib/domain/types"
import {
  ANY_VERSION_SHELL_MESSAGE,
  IS_STATE,
  isLearnChromeKey,
  isLearnSessionStatus,
  LEARN_EMBED_CHANNEL,
  LEARN_PROTOCOL_VERSION,
  UNVERSIONED_PROTOCOL,
  type LearnActivityItem,
  type LearnEnvelope,
  type LearnEventType,
  type LearnFrameMessage,
  type LearnFrameMessageOf,
  type LearnFrameMessageType,
  type LearnModelRef,
  type LearnSessionKind,
  type LearnShellMessage,
  type LearnShellMessageType,
  type LearnStateType,
  type LearnTheme,
} from "@gigachad/chat-ui/learn-protocol"

/** A navigation waits this long for the frame to report its view before reports count as the user's own navigation again. */
export const NAVIGATION_ACK_MS = 5_000

/**
 * Whether the shell may drive the frame: `"waiting"` until the frame's `ready`, `"matched"` when it speaks this shell's
 * protocol version, `"mismatch"` when it does not (LearnHarness runs an older or newer bundle and needs a restart).
 */
export type LearnDrive = "waiting" | "matched" | "mismatch"

/** Shell messages that need drive mode, the ones features send. Handshake, theme and navigation stay the bridge's own. */
export type LearnDriveMessageType = { [Type in LearnShellMessageType]: (typeof ANY_VERSION_SHELL_MESSAGE)[Type] extends true ? never : Type }[LearnShellMessageType]
export type LearnDriveMessage = Extract<LearnShellMessage, { type: LearnDriveMessageType }>

/** The kept state of one replayed frame message type: the latest message, or the latest one per session for `"per-session"` state. */
export type LearnFrameState<Type extends LearnStateType> = (typeof IS_STATE)[Type] extends "per-session"
  ? Readonly<Record<string, LearnFrameMessageOf<Type>>>
  : LearnFrameMessageOf<Type> | null

/** What the shell does with the frame's side of the protocol. */
export interface LearnBridgeHost {
  /** Deliver `envelope` to the frame. */
  post: (envelope: LearnEnvelope<LearnShellMessage>) => void
  /** The frame (re)loaded and got `hello`. */
  onReady: () => void
  /** The user navigated inside the frame, the attached tab should now show `view`. */
  onNavigate: (view: LearnView) => void
}

export interface LearnBridge {
  /** Route the bridge's output to `host`, replacing the previous one. */
  connect: (host: LearnBridgeHost) => void
  /** Handle a message the frame window posted from the frame origin, anything else on the channel is ignored. */
  receive: (data: unknown) => void
  /** The view the attached learn tab wants, null while no learn tab shows the frame. */
  show: (view: LearnView | null) => void
  setTheme: (theme: LearnTheme) => void
  /** The frame reloads: wait for its next `ready`. */
  reset: () => void
  /** Post a drive message, true when it went out. Dropped until the frame is ready at this protocol version. */
  send: (message: LearnDriveMessage) => boolean
  drive: () => LearnDrive
  /** The kept state of `type` since the frame's last `ready`. Same reference until it changes. */
  state: <Type extends LearnStateType>(type: Type) => LearnFrameState<Type>
  /** Called after every change of `drive()` or `state()`, returns the unsubscribe. */
  subscribe: (listener: () => void) => () => void
  /** Called with every frame event of `type`, returns the unsubscribe. */
  onEvent: <Type extends LearnEventType>(type: Type, listener: (message: LearnFrameMessageOf<Type>) => void) => () => void
}

type Fields = Record<string, unknown>

const isId = (value: unknown): value is string => typeof value === "string" && value !== ""
const isOptionalId = (value: unknown): value is string | null => value === null || isId(value)
const isOptionalString = (value: unknown): value is string | null => value === null || typeof value === "string"
const isSessionKind = (value: unknown): value is LearnSessionKind => value === "teach" || value === "practice"
const isTime = (value: unknown): value is string => typeof value === "string" && !Number.isNaN(Date.parse(value))

function modelRefOf(value: unknown): LearnModelRef | null {
  if (typeof value !== "object" || value === null) return null
  const { provider, modelId } = value as Fields
  return isId(provider) && isId(modelId) ? { provider, modelId } : null
}

/** Every element of `value` through `parse`, null when `value` is no array or any element is malformed. */
function listOf<Item>(value: unknown, parse: (item: unknown) => Item | null): Item[] | null {
  if (!Array.isArray(value)) return null
  const items: Item[] = []
  for (const item of value) {
    const parsed = parse(item)
    if (parsed === null) return null
    items.push(parsed)
  }
  return items
}

function activityItemOf(value: unknown): LearnActivityItem | null {
  if (typeof value !== "object" || value === null) return null
  const { id, label, status, sessionId } = value as Fields
  return isId(id) && typeof label === "string" && typeof status === "string" && isOptionalId(sessionId)
    ? { id, label, status, sessionId }
    : null
}

/** One validating parser per frame message type, exhaustive: a new frame message fails `tsc` until it is parsed here. */
const FRAME_PARSERS: { [Type in LearnFrameMessageType]: (message: Fields) => LearnFrameMessageOf<Type> | null } = {
  ready: ({ version }) => {
    if (version === undefined) return { type: "ready", version: UNVERSIONED_PROTOCOL }
    return typeof version === "number" && Number.isInteger(version) && version >= 1 ? { type: "ready", version } : null
  },
  view: ({ sessionId }) => (isOptionalId(sessionId) ? { type: "view", sessionId } : null),
  model: ({ sessionId, model, thinkingLevel, thinkingLevels, locked, unavailable }) => {
    const ref = model === null ? null : modelRefOf(model)
    const levels = listOf(thinkingLevels, (level) => (isId(level) ? level : null))
    if (!isOptionalId(sessionId) || (model !== null && ref === null) || !isOptionalString(thinkingLevel) || levels === null) return null
    if (typeof locked !== "boolean" || typeof unavailable !== "boolean") return null
    return { type: "model", sessionId, model: ref, thinkingLevel, thinkingLevels: levels, locked, unavailable }
  },
  models: ({ served }) => {
    const refs = listOf(served, modelRefOf)
    return refs === null ? null : { type: "models", served: refs }
  },
  "model-open": ({ sessionId }) => (isOptionalId(sessionId) ? { type: "model-open", sessionId } : null),
  modal: ({ open }) => (typeof open === "boolean" ? { type: "modal", open } : null),
  "session-meta": ({ sessionId, title, kind, phase, status, updatedAt }) =>
    isId(sessionId) && typeof title === "string" && (kind === null || isSessionKind(kind)) && isOptionalString(phase)
      && (status === null || isLearnSessionStatus(status)) && isTime(updatedAt)
      ? { type: "session-meta", sessionId, title, kind, phase, status, updatedAt }
      : null,
  "session-plan": ({ sessionId, available }) =>
    isId(sessionId) && typeof available === "boolean" ? { type: "session-plan", sessionId, available } : null,
  activity: ({ items }) => {
    const parsed = listOf(items, activityItemOf)
    return parsed === null ? null : { type: "activity", items: parsed }
  },
  "chrome-key": ({ action }) => (isLearnChromeKey(action) ? { type: "chrome-key", action } : null),
}

function isFrameMessageType(type: unknown): type is LearnFrameMessageType {
  return typeof type === "string" && Object.hasOwn(FRAME_PARSERS, type)
}

/** A frame message on the learn channel with the fields its type needs (and only those), null for anything else. */
export function parseFrameMessage(data: unknown): LearnFrameMessage | null {
  if (typeof data !== "object" || data === null) return null
  const message = data as Fields
  if (message.channel !== LEARN_EMBED_CHANNEL || !isFrameMessageType(message.type)) return null
  return FRAME_PARSERS[message.type](message)
}

function isStateMessage(message: LearnFrameMessage): message is LearnFrameMessageOf<LearnStateType> {
  return IS_STATE[message.type] !== false
}

const NO_SESSIONS: Readonly<Record<string, never>> = Object.freeze({})

function navigationMessage(view: LearnView): LearnShellMessage {
  if (view.kind === "session") return { type: "open-session", sessionId: view.sessionId }
  if (view.kind === "new") return { type: "new-session", kind: view.sessionKind ?? "teach" }
  return { type: "open-home" }
}

/** Which of the views reporting a null session `view` is: a new teaching and a new practice session differ. */
function unsavedViewKey(view: LearnView): string {
  return view.kind === "new" ? `new:${view.sessionKind ?? "teach"}` : view.kind
}

/** Whether a frame reporting `sessionId` shows `view`. Home and new sessions all report null, the last navigation tells them apart. */
function frameShows(sessionId: string | null, view: LearnView, lastSent: LearnView | null): boolean {
  if (view.kind === "session") return sessionId === view.sessionId
  return sessionId === null && lastSent !== null && unsavedViewKey(lastSent) === unsavedViewKey(view)
}

/**
 * The shell's side of the learn protocol (`@gigachad/chat-ui/learn-protocol`), free of DOM and React.
 *
 * - Answers every `ready` with `hello`, the theme and this shell's protocol version, then navigates the frame to the wanted view
 * - Drive mode (`drive()`) only when the frame's `ready` carries the same version: drive messages (`send`) need it,
 *   theme and navigation work at any version
 * - Keeps the frame's replayed state (`IS_STATE`) since its last `ready`, hands its events to `onEvent` listeners.
 *   Malformed frame messages are dropped (`parseFrameMessage`). A frame at another version feeds only its `view`: what
 *   its other messages mean may have changed
 * - Switches views by message only (`open-session`, `new-session`, `open-home`), skipped when the frame already shows the view
 * - A `view` report that disagrees with the wanted view is the user's own navigation inside the frame, unless it arrives
 *   while the frame still catches up with our last navigation
 */
export function createLearnBridge(initialTheme: LearnTheme): LearnBridge {
  let host: LearnBridgeHost | null = null
  let theme = initialTheme
  let wanted: LearnView | null = null
  let ready = false
  let drive: LearnDrive = "waiting"
  /** Session id of the frame's last `view` report, undefined until it sent one since `ready`. */
  let reported: string | null | undefined
  let lastSent: LearnView | null = null
  let pendingUntil = 0
  let states = new Map<LearnStateType, unknown>()
  const listeners = new Set<() => void>()
  const eventListeners = new Map<LearnEventType, Set<(message: LearnFrameMessage) => void>>()

  const post = (message: LearnShellMessage) => host?.post({ channel: LEARN_EMBED_CHANNEL, ...message })
  const changed = () => listeners.forEach((listener) => listener())

  const keep = (message: LearnFrameMessageOf<LearnStateType>) => {
    if (IS_STATE[message.type] === "per-session") {
      const sessions = (states.get(message.type) ?? NO_SESSIONS) as Readonly<Record<string, LearnFrameMessage>>
      states.set(message.type, { ...sessions, [(message as { sessionId: string }).sessionId]: message })
    } else {
      states.set(message.type, message)
    }
    changed()
  }

  const navigate = (view: LearnView, force: boolean) => {
    if (!ready) return
    if (!force && reported !== undefined && frameShows(reported, view, lastSent)) return
    lastSent = view
    pendingUntil = Date.now() + NAVIGATION_ACK_MS
    post(navigationMessage(view))
  }

  const onView = (sessionId: string | null) => {
    reported = sessionId
    if (!wanted) return
    if (frameShows(sessionId, wanted, lastSent)) {
      pendingUntil = 0
      return
    }
    if (Date.now() < pendingUntil) return
    // Null from a session is the user going home. Null on home or a new session stays put, the two report alike.
    const next: LearnView | null = sessionId !== null ? { kind: "session", sessionId } : wanted.kind === "session" ? { kind: "home" } : null
    if (!next) return
    lastSent = next
    host?.onNavigate(next)
  }

  const onReady = (version: number) => {
    ready = true
    drive = version === LEARN_PROTOCOL_VERSION ? "matched" : "mismatch"
    reported = undefined
    lastSent = null
    states = new Map()
    changed()
    post({ type: "hello", theme, version: LEARN_PROTOCOL_VERSION })
    host?.onReady()
    if (wanted) navigate(wanted, true)
  }

  return {
    connect(next) {
      host = next
    },
    receive(data) {
      const message = parseFrameMessage(data)
      if (!message) return
      if (message.type === "ready") return onReady(message.version)
      if (message.type === "view") {
        keep(message)
        return onView(message.sessionId)
      }
      // Navigation works at any version, everything else only from a frame speaking this shell's protocol.
      if (drive !== "matched") return
      if (isStateMessage(message)) return keep(message)
      eventListeners.get(message.type as LearnEventType)?.forEach((listener) => listener(message))
    },
    show(view) {
      wanted = view
      if (view) navigate(view, false)
    },
    setTheme(next) {
      if (next === theme) return
      theme = next
      if (ready) post({ type: "set-theme", theme })
    },
    reset() {
      ready = false
      drive = "waiting"
      states = new Map()
      changed()
    },
    send(message) {
      if (!ready || drive !== "matched") return false
      post(message)
      return true
    },
    drive: () => drive,
    state<Type extends LearnStateType>(type: Type) {
      const kept = states.get(type) ?? (IS_STATE[type] === "per-session" ? NO_SESSIONS : null)
      return kept as LearnFrameState<Type>
    },
    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    onEvent(type, listener) {
      const forType = eventListeners.get(type) ?? new Set()
      eventListeners.set(type, forType)
      const untyped = listener as (message: LearnFrameMessage) => void
      forType.add(untyped)
      return () => forType.delete(untyped)
    },
  }
}
