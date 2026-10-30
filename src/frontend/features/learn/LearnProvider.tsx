"use client"

import { createContext, useCallback, useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react"
import { useSettings } from "@/contexts/SettingsContext"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { useThemeName } from "@/hooks/useThemeName"
import type { LearnSession, LearnView } from "@/lib/domain/types"
import type { LearnEventType, LearnFrameMessageOf, LearnStateType } from "@gigachad/chat-ui/learn-protocol"
import { createLearnBridge, type LearnBridge, type LearnDrive, type LearnDriveMessage, type LearnFrameState } from "./bridge"
import { learnItem, type OpenItem } from "./items"
import { useLearnSessions, type LearnSessions } from "./useLearnSessions"

const READY_TIMEOUT_MS = 20_000

export type LearnFrameStatus = "loading" | "ready" | "unreachable"

/** The learn tab currently showing the shared frame. */
export interface LearnFrameAttachment {
  slot: HTMLElement
  view: LearnView
  /** The user navigated inside the frame (opened a session, went home). */
  onViewChange: (view: LearnView) => void
}

interface LearnContextValue {
  /** LearnHarness's frame address, null for logins without host access. */
  learnUrl: string | null
  frameStatus: LearnFrameStatus
  attachFrame: (attachment: LearnFrameAttachment) => () => void
  reloadFrame: () => void
  sessions: LearnSessions
  /** Show `item` in the current tab, or a new one when `inNewTab`. */
  openItem: OpenItem
  /** Post a drive message to the frame, true when it went out: dropped unless `useLearnDrive()` is `"matched"`. */
  sendToLearnFrame: (message: LearnDriveMessage) => boolean
  /**
   * Show `message`'s upstream UI (settings, a panel) in the frame: `session` opens in the current tab first, else a learn
   * tab is needed (learn home opens in a new tab when the current one is no learn tab). Sent once the frame is driven
   * and shows that session. A frame at another version shows upstream's own chrome instead, the request is dropped.
   */
  showInFrame: (message: LearnDriveMessage, session?: Pick<LearnSession, "id" | "project_slug">) => void
}

const LearnContext = createContext<LearnContextValue | null>(null)
/** The bridge alone: it never changes, so the frame state hooks re-render only for the state they read. */
const LearnFrameContext = createContext<LearnBridge | null>(null)

export function useLearn(): LearnContextValue {
  return useRequiredContext(LearnContext, "useLearn must be used inside LearnProvider")
}

function useLearnBridge(): LearnBridge {
  return useRequiredContext(LearnFrameContext, "learn frame hooks must be used inside LearnProvider")
}

/** The frame's latest `type` state since its last load (`IS_STATE`): the message or null, per session for `session-meta`. */
export function useLearnFrameState<Type extends LearnStateType>(type: Type): LearnFrameState<Type> {
  const bridge = useLearnBridge()
  return useSyncExternalStore(bridge.subscribe, () => bridge.state(type), () => bridge.state(type))
}

/** Call `listener` with every frame event of `type` (`model-open`, `chrome-key`), always its latest render's version. */
export function useLearnFrameEvent<Type extends LearnEventType>(type: Type, listener: (message: LearnFrameMessageOf<Type>) => void): void {
  const bridge = useLearnBridge()
  const onEvent = useEffectEvent(listener)
  useEffect(() => bridge.onEvent(type, (message) => onEvent(message)), [bridge, type])
}

/** Whether the shell may drive the frame: `"matched"` only when both run the same protocol version, `"mismatch"` asks for a LearnHarness restart. */
export function useLearnDrive(): LearnDrive {
  const bridge = useLearnBridge()
  return useSyncExternalStore(bridge.subscribe, bridge.drive, bridge.drive)
}

/** A `showInFrame` request waiting for the frame: driven, attached to a learn tab, on session `sessionId` when it names one. */
interface FrameRequest {
  message: LearnDriveMessage
  sessionId: string | null
  at: number
}

/** A `showInFrame` request the frame did not get to this long is dropped: the user went elsewhere meanwhile. */
const FRAME_REQUEST_MS = 10_000

/**
 * Adopt each session title the frame reports (`session-meta`) into the session list once, so the sidebar and every tab
 * title follow a rename or a fresh title at once. A rename made in the shell stays until the frame reports another
 * title. A reported session the list lacks reads the list again. One instance for every tab.
 */
function useReportedTitles(bridge: LearnBridge, sessions: LearnSessions) {
  const metas = useSyncExternalStore(bridge.subscribe, () => bridge.state("session-meta"), () => bridge.state("session-meta"))
  const reported = useRef(new Map<string, string>())
  const { sessions: known, retitle, refresh } = sessions
  useEffect(() => {
    let unknown = false
    for (const meta of Object.values(metas)) {
      if (reported.current.get(meta.sessionId) === meta.title) continue
      reported.current.set(meta.sessionId, meta.title)
      if (known.some((session) => session.id === meta.sessionId)) retitle(meta.sessionId, meta.title)
      else unknown = true
    }
    if (unknown) void refresh()
  }, [metas, known, retitle, refresh])
}

/**
 * Lay the fixed `frame` over `slot` and keep it there, like the omp frame: moving an iframe in the DOM reloads it.
 * Without a slot it parks as a zero box, Chrome throttles an off-screen cross-origin frame until it never hydrates.
 */
function useFrameOverSlot(frame: HTMLIFrameElement | null, slot: HTMLElement | null) {
  useLayoutEffect(() => {
    if (!frame) return
    const place = () => {
      const { top, left, width, height } = slot?.getBoundingClientRect() ?? { top: 0, left: 0, width: 0, height: 0 }
      Object.assign(frame.style, { top: `${top}px`, left: `${left}px`, width: `${width}px`, height: `${height}px` })
    }
    place()
    if (!slot) return
    const observer = new ResizeObserver(place)
    observer.observe(slot)
    observer.observe(document.body)
    window.addEventListener("resize", place)
    document.addEventListener("transitionend", place, true)
    return () => {
      observer.disconnect()
      window.removeEventListener("resize", place)
      document.removeEventListener("transitionend", place, true)
    }
  }, [frame, slot])
}

/**
 * Own LearnHarness's single frame, shared by every learn tab.
 *
 * - Mounted on the first attach and laid over the active learn tab's slot, tabs switch it by message (`bridge.ts`), never by reload
 * - Host logins only: without a frame URL nothing mounts and no session is read
 * - Keeps the frame's replayed state for `useLearnFrameState`, hands its events to `useLearnFrameEvent`, drive messages go
 *   out through `sendToLearnFrame` once `useLearnDrive()` reports a version-matched frame
 */
export function LearnProvider({ onOpenItem, children }: { onOpenItem: OpenItem; children: ReactNode }) {
  const { learnharnessUrl, hostAccess } = useSettings()
  const learnUrl = hostAccess ? learnharnessUrl : null
  const theme = useThemeName()
  const [attachment, setAttachment] = useState<LearnFrameAttachment | null>(null)
  const sessions = useLearnSessions(learnUrl !== null, attachment !== null)
  const [mounted, setMounted] = useState(false)
  const [frameStatus, setFrameStatus] = useState<LearnFrameStatus>("loading")
  const [frameKey, setFrameKey] = useState(0)
  const [frame, setFrame] = useState<HTMLIFrameElement | null>(null)
  const frameOrigin = useMemo(() => (learnUrl ? new URL(learnUrl).origin : null), [learnUrl])
  useFrameOverSlot(frame, attachment?.slot ?? null)
  const [bridge] = useState(() => createLearnBridge(theme))

  useLayoutEffect(() => bridge.connect({
    post: (envelope) => {
      if (frameOrigin) frame?.contentWindow?.postMessage(envelope, frameOrigin)
    },
    onReady: () => setFrameStatus("ready"),
    onNavigate: (view) => attachment?.onViewChange(view),
  }), [bridge, frame, frameOrigin, attachment])

  useEffect(() => {
    function onMessage(event: MessageEvent) {
      if (!frame || event.source !== frame.contentWindow || event.origin !== frameOrigin) return
      bridge.receive(event.data)
    }
    window.addEventListener("message", onMessage)
    return () => window.removeEventListener("message", onMessage)
  }, [bridge, frame, frameOrigin])

  useEffect(() => bridge.setTheme(theme), [bridge, theme])

  const attachedView = attachment?.view ?? null
  useEffect(() => bridge.show(attachedView), [bridge, attachedView])

  useEffect(() => {
    if (!mounted || frameStatus !== "loading") return
    const timer = window.setTimeout(() => setFrameStatus("unreachable"), READY_TIMEOUT_MS)
    return () => window.clearTimeout(timer)
  }, [mounted, frameStatus, frameKey])

  const attachFrame = useCallback((next: LearnFrameAttachment) => {
    setAttachment(next)
    setMounted(true)
    return () => setAttachment((current) => (current === next ? null : current))
  }, [])

  const reloadFrame = useCallback(() => {
    bridge.reset()
    setFrameStatus("loading")
    setFrameKey((key) => key + 1)
  }, [bridge])

  useReportedTitles(bridge, sessions)

  // The waiting request lives in a ref, `requested` re-runs the delivery for a new one.
  const frameRequest = useRef<FrameRequest | null>(null)
  const [requested, setRequested] = useState(0)
  const attached = attachment !== null
  const showInFrame = useCallback((message: LearnDriveMessage, session?: Pick<LearnSession, "id" | "project_slug">) => {
    if (session) onOpenItem(learnItem({ kind: "session", sessionId: session.id }, session.project_slug))
    else if (!attached) onOpenItem(learnItem({ kind: "home" }), true)
    frameRequest.current = { message, sessionId: session?.id ?? null, at: Date.now() }
    setRequested((count) => count + 1)
  }, [onOpenItem, attached])
  const drive = useSyncExternalStore(bridge.subscribe, bridge.drive, bridge.drive)
  const frameView = useSyncExternalStore(bridge.subscribe, () => bridge.state("view"), () => bridge.state("view"))
  useEffect(() => {
    const request = frameRequest.current
    if (!request) return
    if (Date.now() - request.at > FRAME_REQUEST_MS) {
      frameRequest.current = null
      return
    }
    if (drive === "waiting" || frameStatus !== "ready" || !attached) return
    if (drive === "matched" && request.sessionId !== null && frameView?.sessionId !== request.sessionId) return
    if (drive === "matched") bridge.send(request.message)
    frameRequest.current = null
  }, [requested, drive, frameStatus, attached, frameView, bridge])

  const value = useMemo<LearnContextValue>(
    () => ({ learnUrl, frameStatus, attachFrame, reloadFrame, sessions, openItem: onOpenItem, sendToLearnFrame: bridge.send, showInFrame }),
    [learnUrl, frameStatus, attachFrame, reloadFrame, sessions, onOpenItem, bridge, showInFrame],
  )

  const visible = attachment !== null && frameStatus === "ready"
  useEffect(() => {
    // A hidden frame keeps focus, keystrokes would land in it.
    if (!visible && frame && document.activeElement === frame) frame.blur()
  }, [visible, frame])
  return (
    <LearnContext.Provider value={value}>
      <LearnFrameContext.Provider value={bridge}>{children}</LearnFrameContext.Provider>
      {mounted && learnUrl && (
        <iframe
          key={frameKey}
          ref={setFrame}
          src={learnUrl}
          title="Learn"
          // The microphone: LearnHarness's own dictation runs in the frame.
          allow="clipboard-read; clipboard-write; microphone"
          className="fixed z-0 border-0 bg-paper"
          // Position and size come from useFrameOverSlot. Hidden by opacity, not visibility: Chrome stops rendering
          // a visibility-hidden cross-origin frame and LearnHarness then never reports ready
          inert={!visible}
          style={{ opacity: visible ? 1 : 0, pointerEvents: visible ? undefined : "none" }}
        />
      )}
    </LearnContext.Provider>
  )
}
