"use client"

import { createContext, useCallback, useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { useSettings } from "@/contexts/SettingsContext"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { useThemeName } from "@/hooks/useThemeName"
import { fetchOmpOverview } from "@/lib/backend/api"
import type { OmpOverview, OmpView } from "@/lib/domain/types"
import type { ComposerState } from "./composer"
import { EMBED_CHANNEL, EMBED_PROTOCOL_VERSION, type BranchNode, type Envelope, type FrameMessage, type KeyChord, type ShellMessage } from "@gigachad/chat-ui/protocol"

type ComposerCommand = Extract<ShellMessage, { type: `composer-${string}` }>
/** Frame messages for one composer, not state: delivered to `onComposerEvent` listeners, never kept. */
export type ComposerEvent = Extract<FrameMessage, { type: "composer-restore" | "composer-completions" }>
type ComposerEventListener<T extends ComposerEvent["type"]> = (event: Extract<ComposerEvent, { type: T }>) => void

const OVERVIEW_POLL_MS = 15_000
/** A navigation waits this long for the frame to report its view before reports count as user navigation again. */
const NAVIGATION_ACK_MS = 5_000
const READY_TIMEOUT_MS = 20_000

/** Shell shortcuts that must keep working while focus sits inside the frame. */
const FORWARD_KEYS: KeyChord[] = ["t", "w", "d"].map((key) => ({ key, altKey: true, ctrlKey: false, metaKey: false, shiftKey: false }))

export type FrameStatus = "loading" | "ready" | "unreachable"

export interface BranchTree {
  sessionId: string
  tree: BranchNode[]
  activeLeafId: string | null
}

/** The omp tab currently showing the shared frame. */
export interface FrameAttachment {
  slot: HTMLElement
  view: OmpView
  /** The user navigated inside the frame (fork, first message of a new session, session move). */
  onViewChange: (view: OmpView) => void
  /** The user clicked a local file link inside the frame, `path` is absolute. */
  onOpenFile: (path: string) => void
}

/** Pixels of shell chrome over the frame's top and bottom edge. */
export type FrameInsets = { top: number; bottom: number }

interface OmpContextValue {
  /** Last good omp-web overview, null until one loaded. */
  overview: OmpOverview | null
  refreshOverview: () => void
  frameStatus: FrameStatus
  ompWebUrl: string | null
  /** Branch tree of the session shown in the frame. */
  branchTree: BranchTree | null
  selectLeaf: (sessionId: string, leafId: string | null) => void
  attachFrame: (attachment: FrameAttachment) => () => void
  /** Shell chrome laid over the frame, its transcript scrolls beneath it. */
  setFrameInsets: (insets: FrameInsets) => void
  reloadFrame: () => void
  /** Composer state of the session the frame shows, null until the frame reported one. */
  composer: ComposerState | null
  sendToComposer: (command: ComposerCommand) => void
  /** Listen for composer events of one `type`, returns the unsubscribe. */
  onComposerEvent: <T extends ComposerEvent["type"]>(type: T, listener: ComposerEventListener<T>) => () => void
}

const OmpContext = createContext<OmpContextValue | null>(null)

export function useOmp(): OmpContextValue {
  return useRequiredContext(OmpContext, "useOmp must be used inside OmpProvider")
}

type FrameView = Extract<FrameMessage, { type: "view" }>

function isFrameEnvelope(data: unknown): data is Envelope<FrameMessage> {
  return typeof data === "object" && data !== null && (data as { channel?: unknown }).channel === EMBED_CHANNEL
}

/** Whether the frame already shows `view`. A new-session screen matches only the cwd last sent to it. */
function frameShows(frameView: FrameView | null, view: OmpView, lastNewCwd: string | null): boolean {
  if (!frameView) return false
  if (view.kind === "settings") return frameView.settingsOpen
  if (frameView.settingsOpen) return false
  if (view.kind === "session") return frameView.sessionId === view.sessionId
  return frameView.sessionId === null && lastNewCwd === view.cwd
}

function navigationMessage(view: OmpView): ShellMessage {
  if (view.kind === "session") return { type: "open-session", sessionId: view.sessionId }
  if (view.kind === "new") return { type: "new-session", cwd: view.cwd }
  return { type: "open-settings" }
}

/** Deep-link the initial destination so omp-web's startup cannot select another workspace. */
function frameUrl(ompWebUrl: string, view: OmpView): string {
  if (view.kind === "settings") return ompWebUrl
  const url = new URL(ompWebUrl)
  if (view.kind === "session") url.searchParams.set("session", view.sessionId)
  else url.searchParams.set("cwd", view.cwd)
  return url.href
}

function useOverviewPolling(enabled: boolean): { overview: OmpOverview | null; refreshOverview: () => void } {
  const [overview, setOverview] = useState<OmpOverview | null>(null)
  const { retry: refreshOverview } = useStartupRead("OMP overview", fetchOmpOverview, setOverview, enabled)

  useEffect(() => {
    if (!enabled) return
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refreshOverview()
    }, OVERVIEW_POLL_MS)
    const onFocus = () => refreshOverview()
    window.addEventListener("focus", onFocus)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener("focus", onFocus)
    }
  }, [enabled, refreshOverview])

  return { overview, refreshOverview }
}

/**
 * Lay the fixed `frame` over `slot` and keep it there. The frame never moves in the DOM, moving an iframe reloads it.
 *
 * - Writes the frame's style directly, so it follows a resizing slot in the same paint (a render would trail it)
 * - Without a slot it parks as a zero box at the viewport origin: Chrome throttles an off-screen
 *   cross-origin frame, then omp-web never hydrates and never reports ready
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
 * Own omp-web's single shared frame and the overview every omp surface reads.
 *
 * - One `<iframe>` for all omp tabs, mounted on the first attach and laid over the active tab's slot
 * - Tabs switch it by postMessage (`protocol.ts`), never by reload
 * - Polls the overview every 15s and on window focus
 */
export function OmpProvider({ children }: { children: ReactNode }) {
  const { ompWebUrl, hostAccess } = useSettings()
  const { overview, refreshOverview } = useOverviewPolling(hostAccess)
  const theme = useThemeName()
  const [attachment, setAttachment] = useState<FrameAttachment | null>(null)
  /** View the frame (re)loaded with, null until the first omp tab attached. */
  const [loadedView, setLoadedView] = useState<OmpView | null>(null)
  const [frameStatus, setFrameStatus] = useState<FrameStatus>("loading")
  const [frameKey, setFrameKey] = useState(0)
  const [branchTree, setBranchTree] = useState<BranchTree | null>(null)
  const [composer, setComposer] = useState<ComposerState | null>(null)
  const [insets, setInsets] = useState<FrameInsets>({ top: 0, bottom: 0 })
  const frameRef = useRef<HTMLIFrameElement>(null)
  const [frame, setFrame] = useState<HTMLIFrameElement | null>(null)
  const attachFrameElement = useCallback((element: HTMLIFrameElement | null) => {
    frameRef.current = element
    setFrame(element)
  }, [])
  const frameViewRef = useRef<FrameView | null>(null)
  const lastNewCwdRef = useRef<string | null>(null)
  const pendingUntilRef = useRef(0)
  const readyRef = useRef(false)
  const composerListenersRef = useRef(new Set<(event: ComposerEvent) => void>())
  useFrameOverSlot(frame, attachment?.slot ?? null)
  const frameOrigin = useMemo(() => (ompWebUrl ? new URL(ompWebUrl).origin : null), [ompWebUrl])

  const post = useCallback((message: ShellMessage) => {
    const target = frameRef.current?.contentWindow
    if (!target || !frameOrigin) return
    const envelope: Envelope<ShellMessage> = { channel: EMBED_CHANNEL, ...message }
    target.postMessage(envelope, frameOrigin)
  }, [frameOrigin])

  const navigate = useCallback((view: OmpView, force: boolean) => {
    if (!readyRef.current) return
    if (!force && frameShows(frameViewRef.current, view, lastNewCwdRef.current)) return
    if (view.kind === "new") lastNewCwdRef.current = view.cwd
    pendingUntilRef.current = Date.now() + NAVIGATION_ACK_MS
    post(navigationMessage(view))
  }, [post])

  const onFrameView = useEffectEvent((frameView: FrameView) => {
    frameViewRef.current = frameView
    if (!attachment) return
    const view = attachment.view
    if (frameShows(frameView, view, lastNewCwdRef.current)) {
      pendingUntilRef.current = 0
      return
    }
    // Reports before the frame caught up with our last navigation are stale.
    if (Date.now() < pendingUntilRef.current) return
    if (view.kind === "settings" && !frameView.settingsOpen) {
      // Escape closed Settings inside the frame, the settings tab keeps showing them.
      navigate(view, true)
      return
    }
    if (frameView.sessionId && !frameView.settingsOpen) attachment.onViewChange({ kind: "session", sessionId: frameView.sessionId })
  })

  const onFrameReady = useEffectEvent(() => {
    readyRef.current = true
    frameViewRef.current = null
    lastNewCwdRef.current = null
    setFrameStatus("ready")
    setComposer(null)
    post({ type: "hello", version: EMBED_PROTOCOL_VERSION, forwardKeys: FORWARD_KEYS, theme })
    post({ type: "insets", ...insets })
    if (attachment) navigate(attachment.view, true)
  })

  const onOpenFile = useEffectEvent((path: string) => attachment?.onOpenFile(path))

  useEffect(() => {
    function onMessage(event: MessageEvent) {
      if (event.source !== frameRef.current?.contentWindow || event.origin !== frameOrigin) return
      if (!isFrameEnvelope(event.data)) return
      const message = event.data
      if (message.type === "ready") onFrameReady()
      else if (message.type === "view") onFrameView(message)
      else if (message.type === "branch-tree") setBranchTree({ sessionId: message.sessionId, tree: message.tree, activeLeafId: message.activeLeafId })
      else if (message.type === "composer") setComposer(message)
      else if (message.type === "composer-restore" || message.type === "composer-completions") composerListenersRef.current.forEach((listener) => listener(message))
      else if (message.type === "key") {
        const { key, altKey, ctrlKey, metaKey, shiftKey } = message
        document.dispatchEvent(new KeyboardEvent("keydown", { key, altKey, ctrlKey, metaKey, shiftKey, bubbles: true, cancelable: true }))
      }
      else if (message.type === "open-file") onOpenFile(message.path)
    }
    window.addEventListener("message", onMessage)
    return () => window.removeEventListener("message", onMessage)
  }, [frameOrigin])

  useEffect(() => {
    if (readyRef.current) post({ type: "set-theme", theme })
  }, [theme, post])

  useEffect(() => {
    if (readyRef.current) post({ type: "insets", ...insets })
  }, [insets, post])

  const setFrameInsets = useCallback((next: FrameInsets) => {
    setInsets((current) => (current.top === next.top && current.bottom === next.bottom ? current : next))
  }, [])

  const attachedView = attachment?.view
  useEffect(() => {
    if (attachedView) navigate(attachedView, false)
  }, [attachedView, navigate])

  useEffect(() => {
    if (!loadedView || frameStatus !== "loading") return
    const timer = window.setTimeout(() => setFrameStatus("unreachable"), READY_TIMEOUT_MS)
    return () => window.clearTimeout(timer)
  }, [loadedView, frameStatus, frameKey])

  const attachFrame = useCallback((next: FrameAttachment) => {
    setAttachment(next)
    setLoadedView((current) => current ?? next.view)
    return () => setAttachment((current) => (current === next ? null : current))
  }, [])

  const reloadFrame = useCallback(() => {
    readyRef.current = false
    if (attachedView) setLoadedView(attachedView)
    setFrameStatus("loading")
    setFrameKey((key) => key + 1)
  }, [attachedView])

  const selectLeaf = useCallback((sessionId: string, leafId: string | null) => {
    post({ type: "select-leaf", sessionId, leafId })
  }, [post])

  const onComposerEvent = useCallback(<T extends ComposerEvent["type"]>(type: T, listener: ComposerEventListener<T>) => {
    const listeners = composerListenersRef.current
    const typed = (event: ComposerEvent) => { if (event.type === type) listener(event as Extract<ComposerEvent, { type: T }>) }
    listeners.add(typed)
    return () => { listeners.delete(typed) }
  }, [])

  const value = useMemo<OmpContextValue>(() => ({
    overview, refreshOverview, frameStatus, ompWebUrl, branchTree, selectLeaf, attachFrame, setFrameInsets, reloadFrame, composer, sendToComposer: post, onComposerEvent,
  }), [overview, refreshOverview, frameStatus, ompWebUrl, branchTree, selectLeaf, attachFrame, setFrameInsets, reloadFrame, composer, post, onComposerEvent])

  const visible = attachment !== null && frameStatus === "ready"
  useEffect(() => {
    // A hidden frame keeps focus, keystrokes would land in it (e.g. after Alt+T from inside it opened a chat tab).
    if (!visible && document.activeElement === frameRef.current) frameRef.current?.blur()
  }, [visible])
  return (
    <OmpContext.Provider value={value}>
      {children}
      {loadedView && ompWebUrl && (
        <iframe
          key={frameKey}
          ref={attachFrameElement}
          src={frameUrl(ompWebUrl, loadedView)}
          title="omp"
          allow="clipboard-read; clipboard-write"
          className="fixed z-0 border-0 bg-paper"
          // Position and size come from useFrameOverSlot, React never writes them. Hidden by opacity, not visibility:
          // Chrome stops rendering a visibility-hidden cross-origin frame, and omp-web then never reports ready
          inert={!visible}
          style={{ opacity: visible ? 1 : 0, pointerEvents: visible ? undefined : "none" }}
        />
      )}
    </OmpContext.Provider>
  )
}
