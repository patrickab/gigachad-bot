"use client"

import { createContext, useCallback, useContext, useEffect, useEffectEvent, useMemo, useRef, useState, type ReactNode } from "react"
import { useSettings } from "@/contexts/SettingsContext"
import { useThemeName } from "@/hooks/useThemeName"
import { fetchOmpOverview } from "@/lib/backend/api"
import type { OmpOverview, OmpView } from "@/lib/domain/types"
import { EMBED_CHANNEL, EMBED_PROTOCOL_VERSION, type BranchNode, type Envelope, type FrameMessage, type KeyChord, type ShellMessage } from "./protocol"

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
}

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
  reloadFrame: () => void
}

const OmpContext = createContext<OmpContextValue | null>(null)

export function useOmp(): OmpContextValue {
  const value = useContext(OmpContext)
  if (!value) throw new Error("useOmp must be used inside OmpProvider")
  return value
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

function useOverviewPolling(): { overview: OmpOverview | null; refreshOverview: () => void } {
  const [overview, setOverview] = useState<OmpOverview | null>(null)
  const refreshOverview = useCallback(() => {
    fetchOmpOverview().then(setOverview).catch(() => {
      // omp-web down or disabled, keep the last good overview
    })
  }, [])

  useEffect(() => {
    refreshOverview()
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") refreshOverview()
    }, OVERVIEW_POLL_MS)
    const onFocus = () => refreshOverview()
    window.addEventListener("focus", onFocus)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener("focus", onFocus)
    }
  }, [refreshOverview])

  return { overview, refreshOverview }
}

/** Viewport rect of `slot`, kept current so the fixed frame stays laid over it. The frame never moves in the DOM, moving an iframe reloads it. */
function useSlotRect(slot: HTMLElement | null): DOMRect | null {
  const [rect, setRect] = useState<DOMRect | null>(null)
  useEffect(() => {
    if (!slot) return
    const measure = () => setRect(slot.getBoundingClientRect())
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(slot)
    observer.observe(document.body)
    window.addEventListener("resize", measure)
    document.addEventListener("transitionend", measure, true)
    return () => {
      observer.disconnect()
      window.removeEventListener("resize", measure)
      document.removeEventListener("transitionend", measure, true)
    }
  }, [slot])
  return slot ? rect : null
}

/**
 * Own omp-web's single shared frame and the overview every omp surface reads.
 *
 * - One `<iframe>` for all omp tabs, mounted on the first attach and laid over the active tab's slot
 * - Tabs switch it by postMessage (`protocol.ts`), never by reload
 * - Polls the overview every 15s and on window focus
 */
export function OmpProvider({ children }: { children: ReactNode }) {
  const { ompWebUrl } = useSettings()
  const { overview, refreshOverview } = useOverviewPolling()
  const theme = useThemeName()
  const [attachment, setAttachment] = useState<FrameAttachment | null>(null)
  const [mounted, setMounted] = useState(false)
  const [frameStatus, setFrameStatus] = useState<FrameStatus>("loading")
  const [frameKey, setFrameKey] = useState(0)
  const [branchTree, setBranchTree] = useState<BranchTree | null>(null)
  const frameRef = useRef<HTMLIFrameElement>(null)
  const frameViewRef = useRef<FrameView | null>(null)
  const lastNewCwdRef = useRef<string | null>(null)
  const pendingUntilRef = useRef(0)
  const readyRef = useRef(false)
  const rect = useSlotRect(attachment?.slot ?? null)
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
    post({ type: "hello", version: EMBED_PROTOCOL_VERSION, forwardKeys: FORWARD_KEYS, theme })
    if (attachment) navigate(attachment.view, true)
  })

  useEffect(() => {
    function onMessage(event: MessageEvent) {
      if (event.source !== frameRef.current?.contentWindow || event.origin !== frameOrigin) return
      if (!isFrameEnvelope(event.data)) return
      const message = event.data
      if (message.type === "ready") onFrameReady()
      else if (message.type === "view") onFrameView(message)
      else if (message.type === "branch-tree") setBranchTree({ sessionId: message.sessionId, tree: message.tree, activeLeafId: message.activeLeafId })
      else if (message.type === "key") {
        const { key, altKey, ctrlKey, metaKey, shiftKey } = message
        document.dispatchEvent(new KeyboardEvent("keydown", { key, altKey, ctrlKey, metaKey, shiftKey, bubbles: true, cancelable: true }))
      }
    }
    window.addEventListener("message", onMessage)
    return () => window.removeEventListener("message", onMessage)
  }, [frameOrigin])

  useEffect(() => {
    if (readyRef.current) post({ type: "set-theme", theme })
  }, [theme, post])

  const attachedView = attachment?.view
  useEffect(() => {
    if (attachedView) navigate(attachedView, false)
  }, [attachedView, navigate])

  useEffect(() => {
    if (!mounted || frameStatus !== "loading") return
    const timer = window.setTimeout(() => setFrameStatus("unreachable"), READY_TIMEOUT_MS)
    return () => window.clearTimeout(timer)
  }, [mounted, frameStatus, frameKey])

  const attachFrame = useCallback((next: FrameAttachment) => {
    setAttachment(next)
    setMounted(true)
    return () => setAttachment((current) => (current === next ? null : current))
  }, [])

  const reloadFrame = useCallback(() => {
    readyRef.current = false
    setFrameStatus("loading")
    setFrameKey((key) => key + 1)
  }, [])

  const selectLeaf = useCallback((sessionId: string, leafId: string | null) => {
    post({ type: "select-leaf", sessionId, leafId })
  }, [post])

  const value = useMemo<OmpContextValue>(() => ({
    overview, refreshOverview, frameStatus, ompWebUrl, branchTree, selectLeaf, attachFrame, reloadFrame,
  }), [overview, refreshOverview, frameStatus, ompWebUrl, branchTree, selectLeaf, attachFrame, reloadFrame])

  const visible = attachment !== null && rect !== null && frameStatus === "ready"
  useEffect(() => {
    // A hidden frame keeps focus, keystrokes would land in it (e.g. after Alt+T from inside it opened a chat tab).
    if (!visible && document.activeElement === frameRef.current) frameRef.current?.blur()
  }, [visible])
  return (
    <OmpContext.Provider value={value}>
      {children}
      {mounted && ompWebUrl && (
        <iframe
          key={frameKey}
          ref={frameRef}
          src={ompWebUrl}
          title="omp"
          allow="clipboard-read; clipboard-write"
          className="fixed z-0 border-0 bg-paper"
          style={{
            top: rect?.top ?? 0,
            left: rect?.left ?? 0,
            width: rect?.width ?? 0,
            height: rect?.height ?? 0,
            visibility: visible ? "visible" : "hidden",
          }}
        />
      )}
    </OmpContext.Provider>
  )
}
