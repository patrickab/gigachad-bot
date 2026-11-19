"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { useOpenFile } from "@/features/editor/FilePanel"
import { ApiError, fetchPiSessions, openPiSession } from "@/lib/backend/api"
import type { Attachment, PiView } from "@/lib/domain/types"
import { imageOf, PiComposer } from "./PiComposer"
import { PiTranscript } from "./PiTranscript"
import type { ImageContent } from "./types"
import { usePiSession } from "./usePiSession"

/** First message of a Pi tab started from a chat, sent once the tab's session is ready. */
export interface PiLaunch {
  text: string
  attachments: Attachment[]
}

/** What a session view's header shows. */
export interface PiSessionHeader {
  tone: StatusTone
  sessionName: string | undefined
  /** `provider/model · thinking`, null before the session reports its model. */
  model: string | null
}

interface FirstMessage {
  message: string
  images: ImageContent[]
}

const EMPTY_QUEUE = { steering: [], followUp: [] }

/** The message an open that failed shows. */
export function piOpenError(reason: unknown): string {
  return reason instanceof ApiError && reason.status === 409 ? "Another Pi process holds this session" : reason instanceof Error ? reason.message : String(reason)
}

/**
 * One open Pi session: its header (`renderHeader`), transcript and composer, shared by Pi tabs and the `/pi` page.
 *
 * - File links resolve against `cwd` and open in the file panel
 * - `firstMessage` is sent once the session attaches, then reported through `onFirstMessageSent`
 */
export function PiSessionView({ sessionId, cwd, renderHeader, firstMessage, onFirstMessageSent }: {
  sessionId: string
  cwd: string
  renderHeader: (header: PiSessionHeader) => ReactNode
  firstMessage?: FirstMessage | null
  onFirstMessageSent?: () => void
}) {
  const session = usePiSession(sessionId)
  const { state, status, transcript, send } = session
  const openFile = useOpenFile()
  const fileLinks = useMemo<FileLinks>(() => ({ base: cwd, open: openFile }), [cwd, openFile])
  const tone: StatusTone = status.kind === "ready" ? (transcript.running ? "active" : "ready") : status.kind === "connecting" ? "idle" : "error"
  const model = state?.model ? `${state.model.provider}/${state.model.id}${state.thinkingLevel ? ` · ${state.thinkingLevel}` : ""}` : null

  const firstSent = useRef(false)
  useEffect(() => {
    if (!firstMessage || firstSent.current || status.kind !== "ready") return
    firstSent.current = true
    void send(firstMessage.message, firstMessage.images)
    onFirstMessageSent?.()
  }, [firstMessage, status.kind, send, onFirstMessageSent])

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      {renderHeader({ tone, sessionName: state?.sessionName, model })}
      {status.kind === "closed" && <p role="alert" className="border-b border-divider/50 px-4 py-2 text-xs text-danger">{status.reason}</p>}
      <PiTranscript transcript={transcript} fileLinks={fileLinks} />
      <PiComposer
        sessionKey={sessionId}
        running={transcript.running}
        disabled={status.kind !== "ready"}
        commands={session.commands}
        dialog={session.dialogs[0]}
        queue={transcript.queue}
        notices={session.notices}
        statusLines={session.statusLines}
        editorText={session.editorText}
        onSend={(message, images) => void send(message, images)}
        onAbort={session.abort}
        onAnswerDialog={session.answerDialog}
      />
    </section>
  )
}

/** The tab's top-bar content: live dot, repo, session name and model. */
function TopBarContent({ repo, header }: { repo: string; header: PiSessionHeader | null }) {
  return (
    <div className="flex min-w-0 items-center gap-2 px-2 text-sm">
      {header && <StatusDot tone={header.tone} />}
      <span className="font-medium text-ink">{repo}</span>
      {header?.sessionName && <span className="truncate text-ink-muted">{header.sessionName}</span>}
      {header?.model && <span className="truncate font-mono text-xs text-ink-faint">{header.model}</span>}
    </div>
  )
}

/**
 * A Pi tab: the session `view` names in its repo, with its top bar rendered into `topBar` (the tab's top bar slot).
 *
 * - A tab without a session shows an empty composer, its first send creates the session and reports it through `onViewChange`
 * - A tab with a session resumes it (spawns its Pi process when none runs)
 * - `launch` (a chat's first message) starts the session like a send, then `onLaunched` clears it
 */
export function PiTab({ view, tabId, onViewChange, topBar = null, launch = null, onLaunched }: {
  view: PiView
  tabId: string
  onViewChange: (view: PiView) => void
  topBar?: HTMLElement | null
  launch?: PiLaunch | null
  onLaunched?: () => void
}) {
  const [opened, setOpened] = useState<{ sessionId: string; cwd: string } | null>(null)
  const [firstMessage, setFirstMessage] = useState<FirstMessage | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { sessionId, repo } = view

  // Resume the view's session unless this tab opened it
  useEffect(() => {
    if (!sessionId || opened?.sessionId === sessionId) return
    let cancelled = false
    fetchPiSessions(repo)
      .then((listed) => {
        const session = listed.find((candidate) => candidate.sessionId === sessionId)
        if (!session) throw new Error("This session no longer exists")
        return openPiSession(repo, { sessionPath: session.sessionPath })
      })
      .then((session) => cancelled || setOpened({ sessionId: session.sessionId, cwd: session.cwd }), (reason: unknown) => cancelled || setError(piOpenError(reason)))
    return () => {
      cancelled = true
    }
  }, [sessionId, repo, opened?.sessionId])

  const start = useCallback(async (message: string, images: ImageContent[]) => {
    setError(null)
    setStarting(true)
    try {
      const session = await openPiSession(repo, null)
      setFirstMessage({ message, images })
      setOpened({ sessionId: session.sessionId, cwd: session.cwd })
      onViewChange({ repo, sessionId: session.sessionId })
    } catch (reason) {
      setError(piOpenError(reason))
    } finally {
      setStarting(false)
    }
  }, [repo, onViewChange])

  const launched = useRef(false)
  useEffect(() => {
    if (!launch || sessionId || launched.current) return
    launched.current = true
    const images = launch.attachments.filter((attachment) => attachment.mime.startsWith("image/")).map(imageOf)
    void Promise.all(images).then((converted) => start(launch.text, converted)).finally(() => onLaunched?.())
  }, [launch, sessionId, start, onLaunched])

  const clearFirstMessage = useCallback(() => setFirstMessage(null), [])
  const inTopBar = (content: ReactNode) => topBar ? createPortal(content, topBar) : null
  const shown = opened && opened.sessionId === sessionId ? opened : null

  if (shown) {
    return (
      <PiSessionView
        key={shown.sessionId}
        sessionId={shown.sessionId}
        cwd={shown.cwd}
        renderHeader={(header) => inTopBar(<TopBarContent repo={repo} header={header} />)}
        firstMessage={firstMessage}
        onFirstMessageSent={clearFirstMessage}
      />
    )
  }

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      {inTopBar(<TopBarContent repo={repo} header={null} />)}
      <div className="flex flex-1 items-center justify-center">
        {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      </div>
      {!sessionId && (
        <PiComposer
          sessionKey={`new:${tabId}`}
          running={false}
          disabled={starting}
          commands={[]}
          dialog={undefined}
          queue={EMPTY_QUEUE}
          notices={[]}
          statusLines={{}}
          editorText={null}
          onSend={(message, images) => void start(message, images)}
          onAbort={async () => []}
          onAnswerDialog={() => {}}
        />
      )}
    </section>
  )
}
