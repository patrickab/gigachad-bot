"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ComponentProps, type ReactNode } from "react"
import { useTabWorkingDirectory } from "@/contexts/tabWorkingDirectory"
import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import { LoadFailure } from "@/components/ui/DocumentLoadState"
import type { StatusTone } from "@/components/ui/StatusDot"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { useOpenFile } from "@/features/editor/FilePanel"
import { ApiError, deletePiTurn, fetchPiCatalog, fetchPiSessions, openPiSession, restorePiSession, type PiFileScope, type PiSessionInfo, type PiSessionOptions } from "@/lib/backend/api"
import type { Attachment, PiCatalog, PiView } from "@/lib/domain/types"
import { ComposerDock, useComposerDock } from "@/features/chat/ComposerDock"
import { BtwPanel } from "./BtwPanel"
import { isBtwCommand } from "./btw"
import { imageOf, PiComposer, type PiInsertion, type PiSubmit } from "./PiComposer"
import { piGatewayId, PiRepoLabel, PiTopBar, type PiCatalogProps, type PiModelControls, type PiSessionHeader, type PiTopBarSlots } from "./PiTopBar"
import { PiTranscript } from "./PiTranscript"
import type { Transcript } from "./transcript"
import { usePiSearchFocus } from "./searchFocus"
import { factoryRulings } from "./factoryCard"
import { SessionWavesPanel, useSessionWaves, type SessionCards } from "./SessionWaves"
import { SubagentRoster, useSubagentRoster } from "./SubagentRoster"
import { refreshPiSessions } from "./sessionStore"
import { PiSessionDot } from "./usePiSidebar"
import { isActive, rowsFromToolDetails, type SubagentRow } from "./subagents"
import type { ImageContent, PiCommand, PiModel, ToolRun } from "./types"
import { liveSession, type Notice } from "./liveSession"
import { removeHeld, useHeld } from "./outbox"
import { usePiSession, usePiTranscript } from "./usePiSession"

/** First message of a Pi tab started from a chat, sent once the tab's session is ready. */
export interface PiLaunch {
  text: string
  attachments: Attachment[]
  /** The Pi model and thinking level picked on the start screen, the session starts on them. */
  model?: PiModel | null
  thinking?: string
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

/** The sentence a refused or failed turn delete shows: a 409's reason in words, any other failure's reason as it came. */
export function piTurnError(reason: unknown): string {
  if (reason instanceof ApiError && reason.status === 409) {
    if (reason.code === "running") return "Wait for the run to finish"
    if (reason.code === "compacted") return "A compaction already summarized this turn"
    if (reason.code === "locked" || reason.code === "changed") return "The session file is in use"
  }
  return reason instanceof Error ? reason.message : String(reason)
}

/** The notice of a turn delete whose cleanup could not remove `paths` (pastes, media hashes). */
export function piLeftOnDisk(paths: string[]): string {
  const files = paths.length === 1 ? "1 file" : `${paths.length} files`
  return `The turn is deleted, but ${files} of it could not be removed from disk: ${paths.join(", ")}`
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
export function useNewSessionModels(enabled: boolean): { models: PiModel[]; levels: Map<string, string[]>; defaultLevels: string[]; settled: boolean } {
  const [catalog, setCatalog] = useState<PiCatalog | null>(null)
  const [failed, setFailed] = useState(false)
  useLoadWhile(enabled && !catalog, fetchPiCatalog, setCatalog, () => setFailed(true))
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
    return { models, levels, defaultLevels, settled: catalog !== null || failed }
  }, [catalog, failed])
}

/** `level` if `levels` offers it, else medium, else the highest offered. Any level for a model without reasoning (Pi turns thinking off). */
export function clampThinking(level: string, levels: string[]): string {
  if (levels.length === 0 || levels.includes(level)) return level
  return levels.includes("medium") ? "medium" : levels[levels.length - 1]
}

/** `text` led by `lead` and a blank line, unless there is no lead or `text` is a Pi command (`/name`), which must stay first. */
export function withLead(text: string, lead: string | undefined): string {
  if (!lead || text.trimStart().startsWith("/")) return text
  return `${lead}\n\n${text}`
}

/**
 * What the session's subagents and the factory waves it declared cost, and how many of them work now.
 *
 * A subagent's cost is kept at its highest seen, so dismissing a finished child or the next run's clearing of the roster does not lower the total.
 */
function useSubSessions(sessionId: string, rows: readonly SubagentRow[]): { subCost: number; subActive: number } {
  const [seen, setSeen] = useState<Record<string, number>>({})
  const { waves } = useSessionWaves(sessionId)
  let highest = seen
  for (const row of rows) {
    if ((row.cost ?? 0) > (highest[row.id] ?? 0)) highest = { ...highest, [row.id]: row.cost ?? 0 }
  }
  // Adjusted while rendering: a child's cost rose
  if (highest !== seen) setSeen(highest)
  let total = waves.reduce((sum, wave) => sum + (wave.usage?.cost ?? 0), 0)
  for (const cost of Object.values(highest)) total += cost
  const activeTasks = waves.flatMap((wave) => wave.tasks).filter((task) => task.state === "running" || task.state === "merging").length
  return { subCost: total, subActive: rows.filter((row) => isActive(row.status)).length + activeTasks }
}

/** A transcript without turns: nothing confirmed, live, sent or running. */
const isEmptyTranscript = (transcript: Transcript) =>
  transcript.messages.length === 0 && transcript.unconfirmed.length === 0 && !transcript.optimistic && !transcript.running

/** A selector of the runs the subagent roster reads (those with rows), the previous object while they are the same: a bash call's output frame changes nothing. */
function rosterRunsSelector() {
  let last: Record<string, ToolRun> = {}
  return ({ runs }: Transcript) => {
    const next = Object.fromEntries(Object.entries(runs).filter(([id, run]) => rowsFromToolDetails(id, run.result?.details, run.running).length > 0))
    const ids = Object.keys(next)
    if (ids.length !== Object.keys(last).length || ids.some((id) => next[id] !== last[id])) last = next
    return last
  }
}

/** The session's transcript and held messages: the only part of the view that follows a streamed frame. */
function LiveTranscript({ sessionId, ...props }: { sessionId: string } & Omit<ComponentProps<typeof PiTranscript>, "transcript" | "held" | "onRetryHeld" | "onDeleteHeld">) {
  const transcript = usePiTranscript(sessionId)
  const held = useHeld(sessionId)
  return <PiTranscript transcript={transcript} held={held} onRetryHeld={liveSession(sessionId).retryHeld} onDeleteHeld={(id) => removeHeld(sessionId, [id])} {...props} />
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
 * - Each settled turn branches through `onBranch` (D11); a failed branch shows as a notice
 * - A turn's delete and edit rewrite the session file (`deletePiTurn`), the transcript follows the host's resync. An edit then
 *   sends its text with the turn's images; a refusal shows as a notice, a failed send puts the text into the composer
 * - A closed session's reason shows with the composer's notices
 * - `docked` keeps the composer at the bottom of an empty session, without the greeting (a side panel)
 * - `composerLeading` renders the composer's controls after its "+" menu from the header
 * - With `onOpenTask` the session's factory waves and decisions sit above the subagent roster, given the transcript's factory cards and its jump to a notice
 */
export function PiSessionView({ sessionId, cwd, files, renderHeader, composerLeading, firstSubmit, onFirstSubmitted, onBranch, insertion, onInserted, lead, onLeadSent, docked = false, onOpenTask }: {
  sessionId: string
  cwd: string
  files: PiFileScope
  renderHeader: (header: PiSessionHeader) => ReactNode
  composerLeading?: (header: PiSessionHeader) => ReactNode
  firstSubmit?: PiSubmit | null
  onFirstSubmitted?: () => void
  onBranch?: (fork: PiFork) => Promise<void>
  insertion?: PiInsertion | null
  onInserted?: () => void
  lead?: string
  onLeadSent?: () => void
  docked?: boolean
  onOpenTask?: (repo: string, taskId: string) => void
}) {
  const openFile = useOpenFile()
  const session = usePiSession(sessionId, openFile)
  const search = usePiSearchFocus(sessionId)
  const { state, status, send, runBash, compact, setModel, setThinking, notify } = session
  const running = usePiTranscript(sessionId, (transcript) => transcript.running)
  const queue = usePiTranscript(sessionId, (transcript) => transcript.queue)
  const messages = usePiTranscript(sessionId, (transcript) => transcript.messages)
  const runs = usePiTranscript(sessionId, useMemo(() => rosterRunsSelector(), []))
  const fileLinks = useMemo<FileLinks>(() => ({ base: cwd, open: openFile }), [cwd, openFile])
  const tone: StatusTone = status.kind === "ready" ? (running ? "active" : "ready") : status.kind === "connecting" ? "idle" : "error"
  const [column, setColumn] = useState<HTMLDivElement | null>(null)
  // The strip's view of the transcript's factory cards: a jump goes through this tab's transcript, never another
  const revealNoticeRef = useRef<((key: string) => void) | null>(null)
  const revealNotice = useCallback((key: string) => revealNoticeRef.current?.(key), [])
  const rulings = useMemo(() => factoryRulings(messages), [messages])
  const cards = useMemo<SessionCards>(() => ({ ...rulings, revealNotice }), [rulings, revealNotice])
  const roster = useSubagentRoster(runs, session.widgets, running)
  const { inspectSubagent } = session
  const { subCost, subActive } = useSubSessions(sessionId, roster.rows)
  const inspectRow = useCallback((row: SubagentRow) => inspectSubagent(row.inspect!), [inspectSubagent])

  // Known to be empty only once attached: a connecting session keeps the composer at the bottom, like a chat whose history loads
  const empty = usePiTranscript(sessionId, isEmptyTranscript) && status.kind === "ready" && !firstSubmit
  const dock = useComposerDock(empty && !docked)
  const { mountColumn } = dock
  const notices = useMemo(() => status.kind === "closed" ? [...session.notices, { id: "closed", text: status.reason, tone: "error" as const }] : session.notices, [status, session.notices])

  // An edit's text whose send failed goes into the composer, before the owner's insertion
  const [editInsertion, setEditInsertion] = useState<PiInsertion | null>(null)
  const editInserted = useCallback(() => setEditInsertion(null), [])
  // A delete whose cleanup left files on disk says so once: delete means deleted entirely
  const deleteRange = useCallback(async (entryId: string, through: "turn" | "end") => {
    const { notRemoved } = await deletePiTurn(sessionId, entryId, through)
    if (notRemoved.length > 0) notify(piLeftOnDisk(notRemoved), "warning")
  }, [sessionId, notify])
  const deleteTurn = useCallback((entryId: string) => {
    deleteRange(entryId, "turn").catch((reason: unknown) => notify(piTurnError(reason), "error"))
  }, [deleteRange, notify])
  const editTurn = useCallback(async (entryId: string, text: string, images: ImageContent[]) => {
    try {
      await deleteRange(entryId, "end")
    } catch (reason) {
      return notify(piTurnError(reason), "error")
    }
    if (!(await send(text, images))) setEditInsertion({ id: Date.now(), text })
  }, [deleteRange, notify, send])

  // The lead already sent: one the owner has not cleared yet never leads a second prompt
  const sentLead = useRef<string | null>(null)
  useEffect(() => {
    if (!lead) sentLead.current = null
  }, [lead])

  // `/compact` sent with Alt+Enter while a run is active waits for the run to settle: compacting now would abort it
  const pendingCompact = useRef<{ instructions?: string } | null>(null)
  useEffect(() => {
    if (running || !pendingCompact.current) return
    const { instructions } = pendingCompact.current
    pendingCompact.current = null
    void compact(instructions)
  }, [running, compact])
  const submit = useCallback(async (request: PiSubmit) => {
    const { action } = request
    if (action.kind === "prompt" && isBtwCommand(action.text)) void send(action.text, [])
    else if (action.kind === "bash") void runBash(action.command, action.excludeFromContext)
    else if (action.kind === "compact") {
      if (request.mode === "followUp" && running) pendingCompact.current = { instructions: action.instructions }
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
  }, [runBash, compact, send, lead, onLeadSent, running])

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
      stats: session.stats && { ...session.stats, subCost, subActive },
      onModel: (model) => void setModel(model),
      onThinking: (level) => void setThinking(level),
    },
    column,
  }

  return (
    <section ref={mountColumn} className="relative flex min-h-0 min-w-0 flex-1 flex-col">
      {renderHeader(header)}
      <LiveTranscript sessionId={sessionId} fileLinks={fileLinks} search={search} onBranch={branch} onDeleteTurn={deleteTurn} onEditTurn={editTurn} paddingBottom={dock.paddingBottom} onColumnChange={setColumn} revealNoticeRef={revealNoticeRef} />
      <ComposerDock dock={dock}>
        <BtwPanel threads={session.btw.threads} actions={session.btw} fileLinks={fileLinks} />
        <PiComposer
          sessionKey={sessionId}
          files={files}
          running={running}
          disabled={status.kind !== "ready" && !(status.kind === "connecting" && running)}
          commands={session.commands}
          dialog={session.dialogs[0]}
          queue={queue}
          notices={notices}
          statusLines={session.statusLines}
          roster={<>{onOpenTask && <SessionWavesPanel sessionId={sessionId} cards={cards} onOpenTask={onOpenTask} />}<SubagentRoster rows={roster.rows} onDismiss={roster.dismiss} onClear={roster.clear} inspect={inspectRow} /></>}
          editorText={session.editorText}
          insertion={editInsertion ?? insertion}
          onInserted={editInsertion ? editInserted : onInserted}
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
 * - A turn's branch action forks the session into a new one that opens in this tab
 * - The repo and live dot sit in the composer's action row, where an empty chat picks its project
 * - `launch` (a chat's first message) starts the session like a send, then `onLaunched` clears it
 * - With `onOpenTask`, the factory waves the session declared sit above the composer, a wave's task opens through it (`repo`, task id).
 *   They are read for the session's cwd (absolute, the core maps it to its checkout), never the view's repo, which may be a checkout name
 */
export function PiTab({ view, onViewChange, modelProps, slots = NO_SLOTS, launch = null, onLaunched, onOpenTask }: {
  view: PiView
  onViewChange: (view: PiView) => void
  modelProps: PiCatalogProps
  slots?: PiTopBarSlots
  launch?: PiLaunch | null
  onLaunched?: () => void
  onOpenTask?: (repo: string, taskId: string) => void
}) {
  const [opened, setOpened] = useState<{ sessionId: string; cwd: string } | null>(null)
  const [firstSubmit, setFirstSubmit] = useState<PiSubmit | null>(null)
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
  const start = useCallback(async (submit: PiSubmit, picked?: { model: PiModel | null; thinking: string }): Promise<boolean> => {
    const { model: startModel, thinking: startThinking } = picked ?? { model: chosenModel, thinking: chosenThinking }
    setError(null)
    setStarting(true)
    try {
      // No model chosen keeps Pi's default model
      const options: PiSessionOptions = startModel ? { model: { provider: startModel.provider, id: startModel.id }, thinking: startThinking } : { thinking: startThinking }
      const session = await openPiSession(repo, null, options)
      setFirstSubmit(submit)
      setOpened({ sessionId: session.sessionId, cwd: session.cwd })
      onViewChange({ repo, sessionId: session.sessionId })
      return true
    } catch (reason) {
      setError(piOpenError(reason, repo))
      return false
    } finally {
      setStarting(false)
    }
  }, [repo, onViewChange, chosenModel, chosenThinking])

  const branch = useCallback(async (forkOf: PiFork) => {
    const session = await openPiSession(repo, { forkOf })
    setFirstSubmit(null)
    setOpened({ sessionId: session.sessionId, cwd: session.cwd })
    onViewChange({ repo, sessionId: session.sessionId })
  }, [repo, onViewChange])

  const launched = useRef(false)
  const { settled: catalogSettled, levels: catalogLevels } = newSession
  useEffect(() => {
    if (!launch || sessionId || launched.current || !catalogSettled) return
    launched.current = true
    const model = launch.model ?? null
    const thinking = clampThinking(launch.thinking || chosenThinking, model ? catalogLevels.get(piGatewayId(model)) ?? [] : [])
    const images = launch.attachments.filter((attachment) => attachment.mime.startsWith("image/")).map(imageOf)
    void Promise.all(images)
      .then((converted) => start({ action: { kind: "prompt", text: launch.text }, images: converted, mode: "steer" }, { model, thinking }))
      .finally(() => onLaunched?.())
  }, [launch, sessionId, start, onLaunched, catalogSettled, catalogLevels, chosenThinking])

  const clearFirstSubmit = useCallback(() => setFirstSubmit(null), [])
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
        composerLeading={(header) => <PiRepoLabel repo={repo} tone={header.tone}
          // Connected, the session's own state shows; connecting or closed, the connection's
          mark={header.tone === "active" || header.tone === "ready" ? <PiSessionDot sessionId={shown.sessionId} /> : undefined} />}
        firstSubmit={firstSubmit}
        onFirstSubmitted={clearFirstSubmit}
        onBranch={branch}
        onOpenTask={onOpenTask}
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
          onSubmit={start}
          onAbort={async () => []}
          onAnswerDialog={() => {}}
        />
      </ComposerDock>
    </section>
  )
}
