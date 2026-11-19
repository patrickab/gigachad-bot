"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import type { StatusTone } from "@/components/ui/StatusDot"
import { useOpenFile } from "@/features/editor/FilePanel"
import { ApiError, fetchPiSessions, openPiSession, type PiSessionOptions } from "@/lib/backend/api"
import type { Attachment, PiView } from "@/lib/domain/types"
import { imageOf, PiComposer, type PiSubmit } from "./PiComposer"
import { PiTopBar, type PiSessionHeader } from "./PiTopBar"
import { PiTranscript } from "./PiTranscript"
import { usePiSession } from "./usePiSession"

/** First message of a Pi tab started from a chat, sent once the tab's session is ready. */
export interface PiLaunch {
  text: string
  attachments: Attachment[]
}

/** Where a branch starts: the session file and the last entry it keeps (D11). */
export interface PiFork {
  sessionPath: string
  entryId: string
}

/** A new session starts on Pi's default model at thinking `medium` (D10). */
export const NEW_SESSION_OPTIONS: PiSessionOptions = { thinking: "medium" }

const EMPTY_QUEUE = { steering: [], followUp: [] }

/** The message an open that failed shows. */
export function piOpenError(reason: unknown): string {
  return reason instanceof ApiError && reason.status === 409 ? "Another Pi process holds this session" : reason instanceof Error ? reason.message : String(reason)
}

/**
 * One open Pi session: its header (`renderHeader`), transcript and composer, shared by Pi tabs and the `/pi` page.
 *
 * - File links resolve against `cwd` and open in the file panel, `@` completes files of `repo`
 * - `firstSubmit` runs once the session attaches, then reported through `onFirstSubmitted`
 * - Each settled turn branches through `onBranch` (D11); a failed branch shows as a notice
 */
export function PiSessionView({ sessionId, cwd, repo, renderHeader, firstSubmit, onFirstSubmitted, onBranch }: {
  sessionId: string
  cwd: string
  repo: string
  renderHeader: (header: PiSessionHeader) => ReactNode
  firstSubmit?: PiSubmit | null
  onFirstSubmitted?: () => void
  onBranch?: (fork: PiFork) => Promise<void>
}) {
  const session = usePiSession(sessionId)
  const { state, status, transcript, send, runBash, compact, setModel, setThinking, notify } = session
  const openFile = useOpenFile()
  const fileLinks = useMemo<FileLinks>(() => ({ base: cwd, open: openFile }), [cwd, openFile])
  const tone: StatusTone = status.kind === "ready" ? (transcript.running ? "active" : "ready") : status.kind === "connecting" ? "idle" : "error"

  const submit = useCallback((request: PiSubmit) => {
    const { action } = request
    if (action.kind === "bash") void runBash(action.command, action.excludeFromContext)
    else if (action.kind === "compact") void compact(action.instructions)
    else void send(action.text, request.images, request.mode)
  }, [runBash, compact, send])

  const firstSent = useRef(false)
  useEffect(() => {
    if (!firstSubmit || firstSent.current || status.kind !== "ready") return
    firstSent.current = true
    submit(firstSubmit)
    onFirstSubmitted?.()
  }, [firstSubmit, status.kind, submit, onFirstSubmitted])

  const sessionFile = state?.sessionFile
  const branch = useMemo(() => {
    if (!onBranch || !sessionFile) return undefined
    return (entryId: string) => {
      onBranch({ sessionPath: sessionFile, entryId }).catch((reason: unknown) => notify(piOpenError(reason), "error"))
    }
  }, [onBranch, sessionFile, notify])

  const header: PiSessionHeader = {
    tone,
    sessionName: state?.sessionName,
    controls: {
      model: state?.model ?? null,
      models: session.models,
      thinkingLevel: state?.thinkingLevel ?? null,
      thinkingLevels: session.thinkingLevels,
      stats: session.stats,
      onModel: (model) => void setModel(model),
      onThinking: (level) => void setThinking(level),
    },
  }

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      {renderHeader(header)}
      {status.kind === "closed" && <p role="alert" className="border-b border-divider/50 px-4 py-2 text-xs text-danger">{status.reason}</p>}
      <PiTranscript transcript={transcript} fileLinks={fileLinks} onBranch={branch} />
      <PiComposer
        sessionKey={sessionId}
        repo={repo}
        running={transcript.running}
        disabled={status.kind !== "ready"}
        commands={session.commands}
        dialog={session.dialogs[0]}
        queue={transcript.queue}
        notices={session.notices}
        statusLines={session.statusLines}
        editorText={session.editorText}
        onSubmit={submit}
        onAbort={session.abort}
        onAnswerDialog={session.answerDialog}
      />
    </section>
  )
}

/**
 * A Pi tab: the session `view` names in its repo, with its top bar rendered into `topBar` (the tab's top bar slot).
 *
 * - A tab without a session shows an empty composer, its first send creates the session (`NEW_SESSION_OPTIONS`) and reports it through `onViewChange`
 * - A tab with a session resumes it (spawns its Pi process when none runs)
 * - A turn's branch action forks the session into a new one that opens in this tab
 * - `launch` (a chat's first message) starts the session like a send, then `onLaunched` clears it
 */
export function PiTab({ view, tabId, onViewChange, topBar = null, launch = null, onLaunched }: {
  view: PiView
  onViewChange: (view: PiView) => void
  topBar?: HTMLElement | null
  launch?: PiLaunch | null
  onLaunched?: () => void
}) {
  const [opened, setOpened] = useState<{ sessionId: string; cwd: string } | null>(null)
  const [firstSubmit, setFirstSubmit] = useState<PiSubmit | null>(null)
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

  const start = useCallback(async (submit: PiSubmit) => {
    setError(null)
    setStarting(true)
    try {
      const session = await openPiSession(repo, null, NEW_SESSION_OPTIONS)
      setFirstSubmit(submit)
      setOpened({ sessionId: session.sessionId, cwd: session.cwd })
      onViewChange({ repo, sessionId: session.sessionId })
    } catch (reason) {
      setError(piOpenError(reason))
    } finally {
      setStarting(false)
    }
  }, [repo, onViewChange])

  const branch = useCallback(async (forkOf: PiFork) => {
    const session = await openPiSession(repo, { forkOf })
    setFirstSubmit(null)
    setOpened({ sessionId: session.sessionId, cwd: session.cwd })
    onViewChange({ repo, sessionId: session.sessionId })
  }, [repo, onViewChange])

  const launched = useRef(false)
  useEffect(() => {
    if (!launch || sessionId || launched.current) return
    launched.current = true
    const images = launch.attachments.filter((attachment) => attachment.mime.startsWith("image/")).map(imageOf)
    void Promise.all(images)
      .then((converted) => start({ action: { kind: "prompt", text: launch.text }, images: converted, mode: "steer" }))
      .finally(() => onLaunched?.())
  }, [launch, sessionId, start, onLaunched])

  const clearFirstSubmit = useCallback(() => setFirstSubmit(null), [])
  const inTopBar = (content: ReactNode) => topBar ? createPortal(content, topBar) : null
  const shown = opened && opened.sessionId === sessionId ? opened : null

  if (shown) {
    return (
      <PiSessionView
        key={shown.sessionId}
        sessionId={shown.sessionId}
        cwd={shown.cwd}
        repo={repo}
        renderHeader={(header) => inTopBar(<PiTopBar repo={repo} header={header} />)}
        firstSubmit={firstSubmit}
        onFirstSubmitted={clearFirstSubmit}
        onBranch={branch}
      />
    )
  }

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      {inTopBar(<PiTopBar repo={repo} header={null} />)}
      <div className="flex flex-1 items-center justify-center">
        {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      </div>
      {!sessionId && (
        <PiComposer
          sessionKey={`new:${tabId}`}
          repo={repo}
          running={false}
          disabled={starting}
          commands={[]}
          dialog={undefined}
          queue={EMPTY_QUEUE}
          notices={[]}
          statusLines={{}}
          editorText={null}
          onSubmit={(submit) => void start(submit)}
          onAbort={async () => []}
          onAnswerDialog={() => {}}
        />
      )}
    </section>
  )
}
