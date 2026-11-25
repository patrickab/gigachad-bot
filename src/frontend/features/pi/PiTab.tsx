"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { useTabWorkingDirectory } from "@/contexts/tabWorkingDirectory"
import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import { LoadFailure } from "@/components/ui/DocumentLoadState"
import type { StatusTone } from "@/components/ui/StatusDot"
import { useOpenFile } from "@/features/editor/FilePanel"
import { ApiError, fetchPiCatalog, fetchPiSessions, openPiSession, restorePiSession, type PiFileScope, type PiSessionInfo, type PiSessionOptions } from "@/lib/backend/api"
import type { Attachment, PiCatalog, PiView } from "@/lib/domain/types"
import { ComposerDock, useComposerDock } from "@/features/chat/ComposerDock"
import { BtwPanel } from "./BtwPanel"
import { isBtwCommand } from "./btw"
import { imageOf, PiComposer, type PiInsertion, type PiSubmit } from "./PiComposer"
import { piGatewayId, PiRepoLabel, PiTopBar, type PiCatalogProps, type PiModelControls, type PiSessionHeader, type PiTopBarSlots } from "./PiTopBar"
import { PiTranscript } from "./PiTranscript"
import { usePiSearchFocus } from "./searchFocus"
import { SessionWavesPanel } from "./SessionWaves"
import { SubagentRoster, useSubagentRoster } from "./SubagentRoster"
import { refreshPiSessions } from "./sessionStore"
import type { SubagentRow } from "./subagents"
import type { PiCommand, PiModel } from "./types"
import { usePiSession, type Notice } from "./usePiSession"

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
const NEW_SESSION_OPTIONS: PiSessionOptions = { thinking: "medium" }

const EMPTY_QUEUE = { steering: [], followUp: [] }
const NO_COMMANDS: PiCommand[] = []
const NO_NOTICES: Notice[] = []
const NO_SLOTS: PiTopBarSlots = { thinking: null, model: null, usage: null }

/** The message an open in `repo` that failed shows. */
function piOpenError(reason: unknown, repo?: string): string {
  if (reason instanceof ApiError && reason.status === 409) return "Another Pi process holds this session"
  if (reason instanceof ApiError && reason.status === 404 && repo !== undefined && reason.message.startsWith("Unknown repository")) return `${repo} is not a folder`
  return reason instanceof Error ? reason.message : String(reason)
}

/** Session `sessionId` of `repo` to resume: a listed one, or an archived one restored first. */
async function resumableSession(repo: string, sessionId: string): Promise<PiSessionInfo> {
  const listed = await fetchPiSessions(repo)
  const session = listed.find((candidate) => candidate.sessionId === sessionId)
  if (session) return session
  const archived = await fetchPiSessions(repo, { archived: true })
  if (!archived.some((candidate) => candidate.sessionId === sessionId)) throw new Error("This session no longer exists")
  const restored = await restorePiSession(sessionId)
  refreshPiSessions()
  return restored
}

/**
 * Models a new session may start on (`/api/models/pi`), as Pi names them, and each one's thinking ladder by `<provider>/<id>`. Empty until loaded or
 * when pi-host is offline.
 *
 * `defaultLevels` (for Pi's default model, whose ladder the catalog does not name) is the longest ladder: Pi clamps a level its model lacks.
 */
function useNewSessionModels(enabled: boolean): { models: PiModel[]; levels: Map<string, string[]>; defaultLevels: string[] } {
  const [catalog, setCatalog] = useState<PiCatalog | null>(null)
  useEffect(() => {
    if (!enabled || catalog) return
    let cancelled = false
    fetchPiCatalog().then((loaded) => cancelled || setCatalog(loaded), () => {})
    return () => {
      cancelled = true
    }
  }, [enabled, catalog])
  return useMemo(() => {
    const models: PiModel[] = []
    const levels = new Map<string, string[]>()
    for (const provider of catalog?.providers ?? []) {
      for (const model of provider.models) {
        const piModel = { provider: provider.id, id: model.id.startsWith(`${provider.id}/`) ? model.id.slice(provider.id.length + 1) : model.id }
        models.push(piModel)
        levels.set(piGatewayId(piModel), model.thinking_levels)
      }
    }
    const defaultLevels = [...levels.values()].reduce<string[]>((longest, ladder) => (ladder.length > longest.length ? ladder : longest), [])
    return { models, levels, defaultLevels }
  }, [catalog])
}

/** `level` if `levels` offers it, else medium, else the highest offered. Any level for a model without reasoning (Pi turns thinking off). */
function clampThinking(level: string, levels: string[]): string {
  if (levels.length === 0 || levels.includes(level)) return level
  return levels.includes("medium") ? "medium" : levels[levels.length - 1]
}

/** `text` led by `lead` and a blank line, unless there is no lead or `text` is a Pi command (`/name`), which must stay first. */
export function withLead(text: string, lead: string | undefined): string {
  if (!lead || text.trimStart().startsWith("/")) return text
  return `${lead}\n\n${text}`
}

/**
 * One open Pi session: its header (`renderHeader`), transcript and composer, shared by Pi tabs, the `/pi` page and workbench chats.
 *
 * - Laid out like a chat: the composer floats over the transcript's column, a session without turns lifts it under the greeting
 * - The owner opens the session: a coding tab through `/api/pi/sessions`, a workbench chat through `/api/workbench/chat`
 * - File links resolve against `cwd` and open in the file panel, `@` completes the files of `files` (a repo, or a chat's folder)
 * - `firstSubmit` runs once the session attaches, then reported through `onFirstSubmitted`
 * - `insertion` goes into the composer once it is enabled (a quote, a canvas screenshot), `onInserted` follows
 * - `lead` leads the next prompt once (a PDF chat's `Text: <path>`, a renamed file's new path), `onLeadSent` follows. Commands never carry it
 * - Each settled turn branches through `onBranch` (D11), Edit with its prompt for the branch's composer; a failed branch shows as a notice
 * - A closed session's reason shows with the composer's notices
 * - `docked` keeps the composer at the bottom of an empty session, without the greeting (a side panel)
 * - `composerLeading` renders the composer's controls after its "+" menu from the header
 * - `waves` sits above the subagent roster (a Pi tab's factory waves)
 */
export function PiSessionView({ sessionId, cwd, files, renderHeader, composerLeading, firstSubmit, onFirstSubmitted, onBranch, insertion, onInserted, lead, onLeadSent, docked = false, waves }: {
  sessionId: string
  cwd: string
  files: PiFileScope
  renderHeader: (header: PiSessionHeader) => ReactNode
  composerLeading?: (header: PiSessionHeader) => ReactNode
  firstSubmit?: PiSubmit | null
  onFirstSubmitted?: () => void
  onBranch?: (fork: PiFork, prompt?: string) => Promise<void>
  insertion?: PiInsertion | null
  onInserted?: () => void
  lead?: string
  onLeadSent?: () => void
  docked?: boolean
  waves?: ReactNode
}) {
  const openFile = useOpenFile()
  const session = usePiSession(sessionId, openFile)
  const search = usePiSearchFocus(sessionId)
  const { state, status, transcript, send, runBash, compact, setModel, setThinking, notify } = session
  const fileLinks = useMemo<FileLinks>(() => ({ base: cwd, open: openFile }), [cwd, openFile])
  const tone: StatusTone = status.kind === "ready" ? (transcript.running ? "active" : "ready") : status.kind === "connecting" ? "idle" : "error"
  const [column, setColumn] = useState<HTMLDivElement | null>(null)
  const roster = useSubagentRoster(transcript.runs, session.widgets, transcript.running)
  const { inspectSubagent } = session
  const inspectRow = useCallback((row: SubagentRow) => inspectSubagent(row.inspect!), [inspectSubagent])

  const { messages } = transcript
  // Known to be empty only once attached: a connecting session keeps the composer at the bottom, like a chat whose history loads
  const empty = status.kind === "ready" && !firstSubmit && messages.length === 0 && transcript.unconfirmed.length === 0 && !transcript.optimistic && !transcript.running
  const dock = useComposerDock(empty && !docked)
  const { mountColumn } = dock
  const notices = useMemo(() => status.kind === "closed" ? [...session.notices, { id: "closed", text: status.reason, tone: "error" as const }] : session.notices, [status, session.notices])

  // The lead already sent: one the owner has not cleared yet never leads a second prompt
  const sentLead = useRef<string | null>(null)
  useEffect(() => {
    if (!lead) sentLead.current = null
  }, [lead])

  // `/compact` sent with Alt+Enter while a run is active waits for the run to settle: compacting now would abort it
  const pendingCompact = useRef<{ instructions?: string } | null>(null)
  useEffect(() => {
    if (transcript.running || !pendingCompact.current) return
    const { instructions } = pendingCompact.current
    pendingCompact.current = null
    void compact(instructions)
  }, [transcript.running, compact])
  const submit = useCallback(async (request: PiSubmit) => {
    const { action } = request
    if (action.kind === "prompt" && isBtwCommand(action.text)) void send(action.text, [])
    else if (action.kind === "bash") void runBash(action.command, action.excludeFromContext)
    else if (action.kind === "compact") {
      if (request.mode === "followUp" && transcript.running) pendingCompact.current = { instructions: action.instructions }
      else void compact(action.instructions)
    }
    else {
      const text = withLead(action.text, lead !== sentLead.current ? lead : undefined)
      void send(text, request.images, request.mode)
      if (text !== action.text) {
        sentLead.current = lead ?? null
        onLeadSent?.()
      }
    }
  }, [runBash, compact, send, lead, onLeadSent, transcript.running])

  const firstSent = useRef(false)
  useEffect(() => {
    if (!firstSubmit || firstSent.current || status.kind !== "ready") return
    firstSent.current = true
    void submit(firstSubmit)
    onFirstSubmitted?.()
  }, [firstSubmit, status.kind, submit, onFirstSubmitted])

  const sessionFile = state?.sessionFile
  const branch = useMemo(() => {
    if (!onBranch || !sessionFile) return undefined
    return (entryId: string, prompt?: string) => {
      onBranch({ sessionPath: sessionFile, entryId }, prompt).catch((reason: unknown) => notify(piOpenError(reason), "error"))
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
    column,
  }

  return (
    <section ref={mountColumn} className="relative flex min-h-0 min-w-0 flex-1 flex-col">
      {renderHeader(header)}
      <PiTranscript transcript={transcript} held={session.held} onRetryHeld={session.retryHeld} onDeleteHeld={session.deleteHeld} fileLinks={fileLinks} search={search} onBranch={branch} paddingBottom={dock.paddingBottom} onColumnChange={setColumn} />
      <ComposerDock dock={dock}>
        <BtwPanel threads={session.btw.threads} actions={session.btw} fileLinks={fileLinks} />
        <PiComposer
          sessionKey={sessionId}
          files={files}
          running={transcript.running}
          disabled={status.kind !== "ready" && !(status.kind === "connecting" && transcript.running)}
          commands={session.commands}
          dialog={session.dialogs[0]}
          queue={transcript.queue}
          notices={notices}
          statusLines={session.statusLines}
          roster={<>{waves}<SubagentRoster rows={roster.rows} onDismiss={roster.dismiss} onClear={roster.clear} inspect={inspectRow} /></>}
          editorText={session.editorText}
          insertion={insertion}
          onInserted={onInserted}
          leading={composerLeading?.(header)}
          onSubmit={(request) => void submit(request)}
          onAbort={session.abort}
          onAnswerDialog={session.answerDialog}
        />
      </ComposerDock>
    </section>
  )
}

/**
 * A Pi tab: the session `view` names in its repo, laid out like a chat, with its controls in the tab's top bar `slots` where a chat has its own.
 *
 * - A tab without a session shows the chat's empty state (greeting over the lifted composer) and the model and thinking level to start on
 *   (Pi's default model at `medium`, the models from `/api/models/pi`), its first send creates the session and reports it through
 *   `onViewChange`. Its draft is kept per repo, a failed start shows with the composer's notices
 * - The top bar picks the model from the user's catalog (`modelProps`, the chats' dropdown with Manage models) among those Pi offers, the thinking
 *   level from the model's ladder
 * - A tab with a session resumes it (spawns its Pi process when none runs), an archived one is restored first
 * - A turn's branch action forks the session into a new one that opens in this tab, Edit puts the turn's prompt into its composer
 * - The repo and live dot sit in the composer's action row, where an empty chat picks its project
 * - `launch` (a chat's first message) starts the session like a send, then `onLaunched` clears it
 * - With `onOpenSession`, the unfinished factory waves the session declared sit above the composer, a wave task's session opens through it.
 *   They are read for the session's cwd (absolute, the core maps it to its checkout), never the view's repo, which may be a checkout name
 */
export function PiTab({ view, onViewChange, modelProps, slots = NO_SLOTS, launch = null, onLaunched, onOpenSession }: {
  view: PiView
  onViewChange: (view: PiView) => void
  modelProps: PiCatalogProps
  slots?: PiTopBarSlots
  launch?: PiLaunch | null
  onLaunched?: () => void
  onOpenSession?: (sessionId: string) => void
}) {
  const [opened, setOpened] = useState<{ sessionId: string; cwd: string } | null>(null)
  const [firstSubmit, setFirstSubmit] = useState<PiSubmit | null>(null)
  const [insertion, setInsertion] = useState<PiInsertion | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Bumped by Retry after a resume failed, so the resume runs again
  const [resumeAttempt, setResumeAttempt] = useState(0)
  const { sessionId, repo } = view
  const newSession = useNewSessionModels(!sessionId)
  const [choice, setChoice] = useState<{ model: PiModel | null; thinking: string }>({ model: null, thinking: NEW_SESSION_OPTIONS.thinking ?? "medium" })
  const choiceLevels = choice.model ? newSession.levels.get(piGatewayId(choice.model)) ?? [] : newSession.defaultLevels

  // Resume the view's session unless this tab opened it
  useEffect(() => {
    if (!sessionId || opened?.sessionId === sessionId) return
    let cancelled = false
    resumableSession(repo, sessionId)
      .then((session) => openPiSession(repo, { sessionPath: session.sessionPath }))
      .then((session) => cancelled || setOpened({ sessionId: session.sessionId, cwd: session.cwd }), (reason: unknown) => cancelled || setError(piOpenError(reason, repo)))
    return () => {
      cancelled = true
    }
  }, [sessionId, repo, opened?.sessionId, resumeAttempt])

  const { model: chosenModel, thinking: chosenThinking } = choice
  const start = useCallback(async (submit: PiSubmit) => {
    setError(null)
    setStarting(true)
    try {
      // No model chosen keeps Pi's default model
      const options: PiSessionOptions = chosenModel ? { model: { provider: chosenModel.provider, id: chosenModel.id }, thinking: chosenThinking } : { thinking: chosenThinking }
      const session = await openPiSession(repo, null, options)
      setFirstSubmit(submit)
      setOpened({ sessionId: session.sessionId, cwd: session.cwd })
      onViewChange({ repo, sessionId: session.sessionId })
    } catch (reason) {
      setError(piOpenError(reason, repo))
    } finally {
      setStarting(false)
    }
  }, [repo, onViewChange, chosenModel, chosenThinking])

  const branch = useCallback(async (forkOf: PiFork, prompt?: string) => {
    const session = await openPiSession(repo, { forkOf })
    setFirstSubmit(null)
    setInsertion(prompt ? { id: Date.now(), text: prompt } : null)
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
  const clearInsertion = useCallback(() => setInsertion(null), [])
  const shown = opened && opened.sessionId === sessionId ? opened : null
  useTabWorkingDirectory(shown?.cwd)
  const dock = useComposerDock(!sessionId)
  const { mountColumn } = dock

  if (shown) {
    return (
      <PiSessionView
        key={shown.sessionId}
        sessionId={shown.sessionId}
        cwd={shown.cwd}
        files={{ repo }}
        renderHeader={(header) => <PiTopBar slots={slots} header={header} modelProps={modelProps} />}
        composerLeading={(header) => <PiRepoLabel repo={repo} tone={header.tone} />}
        firstSubmit={firstSubmit}
        onFirstSubmitted={clearFirstSubmit}
        onBranch={branch}
        insertion={insertion}
        onInserted={clearInsertion}
        waves={onOpenSession && <SessionWavesPanel repo={shown.cwd} sessionId={shown.sessionId} onOpenSession={onOpenSession} />}
      />
    )
  }

  const newControls: PiModelControls = {
    model: choice.model,
    models: newSession.models,
    thinkingLevel: choice.thinking,
    thinkingLevels: choiceLevels,
    onModel: (model) => setChoice((current) => ({ model, thinking: clampThinking(current.thinking, newSession.levels.get(piGatewayId(model)) ?? []) })),
    onThinking: (thinking) => setChoice((current) => ({ ...current, thinking })),
  }

  if (sessionId) {
    return (
      <section className="flex min-h-0 min-w-0 flex-1 flex-col items-center justify-center">
        <PiTopBar slots={slots} header={null} modelProps={modelProps} />
        {error && <LoadFailure message={error} onRetry={() => { setError(null); setResumeAttempt((value) => value + 1) }} />}
      </section>
    )
  }

  return (
    <section ref={mountColumn} className="relative flex min-h-0 min-w-0 flex-1 flex-col">
      <PiTopBar slots={slots} header={null} controls={newControls} modelProps={modelProps} />
      <ComposerDock dock={dock}>
        <PiComposer
          sessionKey={`new:${repo}`}
          files={{ repo }}
          running={false}
          disabled={starting}
          commands={NO_COMMANDS}
          dialog={undefined}
          queue={EMPTY_QUEUE}
          notices={error ? [{ id: "start", text: error, tone: "error" }] : NO_NOTICES}
          statusLines={{}}
          editorText={null}
          leading={<PiRepoLabel repo={repo} />}
          onSubmit={(submit) => void start(submit)}
          onAbort={async () => []}
          onAnswerDialog={() => {}}
        />
      </ComposerDock>
    </section>
  )
}
