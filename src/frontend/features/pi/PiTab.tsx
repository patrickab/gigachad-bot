"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import type { StatusTone } from "@/components/ui/StatusDot"
import { useOpenFile } from "@/features/editor/FilePanel"
import { ApiError, fetchPiCatalog, fetchPiSessions, openPiSession, restorePiSession, type PiSessionInfo, type PiSessionOptions } from "@/lib/backend/api"
import type { Attachment, Message, PiCatalog, PiView } from "@/lib/domain/types"
import { imageOf, PiComposer, type PiSubmit } from "./PiComposer"
import { PiTopBar, type PiSessionControls, type PiSessionHeader } from "./PiTopBar"
import { PiTranscript } from "./PiTranscript"
import { refreshPiSessions } from "./sessionStore"
import { textOf } from "./transcript"
import type { AgentMessage, PiCommand, PiModel } from "./types"
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
const NO_COMMANDS: PiCommand[] = []

/** The message an open in `repo` that failed shows. */
export function piOpenError(reason: unknown, repo?: string): string {
  if (reason instanceof ApiError && reason.status === 409) return "Another Pi process holds this session"
  if (reason instanceof ApiError && reason.status === 404 && repo !== undefined && reason.message.startsWith("Unknown repository")) return `${repo} is not a Git checkout under ~/git, Pi sessions start only there`
  return reason instanceof Error ? reason.message : String(reason)
}

/** Commands the tab's shell runs itself (`/memorize`), offered in the composer's `/` menu. */
export interface PiShellCommands {
  /** Command names without the `/`. */
  names: string[]
  /** Run `text` if it is one of them and report whether it was. */
  run: (text: string) => Promise<boolean>
}

/** Confirmed messages of each open session view, by session id, for `/memorize`. */
const openTranscripts = new Map<string, AgentMessage[]>()

/** The prompts and replies of the open session `sessionId` as a memorization reads them, null while no view shows it. */
export function piSessionHistory(sessionId: string): Pick<Message, "role" | "content">[] | null {
  const messages = openTranscripts.get(sessionId)
  if (!messages) return null
  return messages.flatMap((message) => {
    if (message.role !== "user" && message.role !== "assistant") return []
    const content = textOf(message.content).trim()
    return content ? [{ role: message.role, content }] : []
  })
}

/** Run `submit` as one of the tab's shell commands (`/memorize`) when it is one, reporting whether it was. */
async function runShellCommand(submit: PiSubmit, shellCommands: PiShellCommands | undefined): Promise<boolean> {
  return submit.action.kind === "prompt" && !!shellCommands && shellCommands.run(submit.action.text.trim())
}

/** `shellCommands` listed with Pi's commands in the composer's `/` menu. */
function useComposerCommands(commands: PiCommand[], shellCommands: PiShellCommands | undefined): PiCommand[] {
  return useMemo(() => [...commands, ...(shellCommands?.names ?? []).map((name): PiCommand => ({ name, source: "extension" }))], [commands, shellCommands])
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

type CatalogModel = PiCatalog["providers"][number]["models"][number]

/**
 * Models a new session may start on (`/api/models/pi`), as Pi names them, with each one's thinking ladder. Empty until loaded or when pi-host is offline.
 *
 * `defaultLevels` (for Pi's default model, whose ladder the catalog does not name) is the longest ladder: Pi clamps a level its model lacks.
 */
function useNewSessionModels(enabled: boolean): { models: PiModel[]; levels: Map<PiModel, string[]>; defaultLevels: string[] } {
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
    const levels = new Map<PiModel, string[]>()
    for (const provider of catalog?.providers ?? []) {
      for (const model of provider.models) levels.set(piModelOf(provider.id, model), model.thinking_levels)
    }
    const defaultLevels = [...levels.values()].reduce<string[]>((longest, ladder) => (ladder.length > longest.length ? ladder : longest), [])
    return { models: [...levels.keys()], levels, defaultLevels }
  }, [catalog])
}

/** A catalog model (`anthropic/claude-haiku-4-5`) as Pi names it: provider plus its own id. */
function piModelOf(provider: string, model: CatalogModel): PiModel {
  const id = model.id.startsWith(`${provider}/`) ? model.id.slice(provider.length + 1) : model.id
  return { provider, id, name: model.name, reasoning: model.thinking_levels.length > 0 }
}

/** `level` if `levels` offers it, else medium, else the highest offered. Any level for a model without reasoning (Pi turns thinking off). */
function clampThinking(level: string, levels: string[]): string {
  if (levels.length === 0 || levels.includes(level)) return level
  return levels.includes("medium") ? "medium" : levels[levels.length - 1]
}

/**
 * One open Pi session: its header (`renderHeader`), transcript and composer, shared by Pi tabs and the `/pi` page.
 *
 * - File links resolve against `cwd` and open in the file panel, `@` completes files of `repo`
 * - `firstSubmit` runs once the session attaches, then reported through `onFirstSubmitted`
 * - Each settled turn branches through `onBranch` (D11); a failed branch shows as a notice
 */
export function PiSessionView({ sessionId, cwd, repo, renderHeader, firstSubmit, onFirstSubmitted, onBranch, shellCommands }: {
  sessionId: string
  cwd: string
  repo: string
  renderHeader: (header: PiSessionHeader) => ReactNode
  firstSubmit?: PiSubmit | null
  onFirstSubmitted?: () => void
  onBranch?: (fork: PiFork) => Promise<void>
  shellCommands?: PiShellCommands
}) {
  const session = usePiSession(sessionId)
  const { state, status, transcript, send, runBash, compact, setModel, setThinking, notify } = session
  const openFile = useOpenFile()
  const fileLinks = useMemo<FileLinks>(() => ({ base: cwd, open: openFile }), [cwd, openFile])
  const tone: StatusTone = status.kind === "ready" ? (transcript.running ? "active" : "ready") : status.kind === "connecting" ? "idle" : "error"

  const { messages } = transcript
  useEffect(() => {
    openTranscripts.set(sessionId, messages)
    return () => {
      if (openTranscripts.get(sessionId) === messages) openTranscripts.delete(sessionId)
    }
  }, [sessionId, messages])

  const commands = useComposerCommands(session.commands, shellCommands)
  const submit = useCallback(async (request: PiSubmit) => {
    if (await runShellCommand(request, shellCommands)) return
    const { action } = request
    if (action.kind === "bash") void runBash(action.command, action.excludeFromContext)
    else if (action.kind === "compact") void compact(action.instructions)
    else void send(action.text, request.images, request.mode)
  }, [runBash, compact, send, shellCommands])

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
        commands={commands}
        dialog={session.dialogs[0]}
        queue={transcript.queue}
        notices={session.notices}
        statusLines={session.statusLines}
        editorText={session.editorText}
        onSubmit={(request) => void submit(request)}
        onAbort={session.abort}
        onAnswerDialog={session.answerDialog}
      />
    </section>
  )
}

/**
 * A Pi tab: the session `view` names in its repo, with its top bar rendered into `topBar` (the tab's top bar slot).
 *
 * - A tab without a session shows an empty composer and the model and thinking level to start on (Pi's default model at `medium`, from `/api/models/pi`),
 *   its first send creates the session and reports it through `onViewChange`. Its draft is kept per repo
 * - A tab with a session resumes it (spawns its Pi process when none runs), an archived one is restored first
 * - `shellCommands` (`/memorize`) share the composer's `/` menu, the shell runs them instead of sending them
 * - A turn's branch action forks the session into a new one that opens in this tab
 * - `launch` (a chat's first message) starts the session like a send, then `onLaunched` clears it
 */
export function PiTab({ view, onViewChange, topBar = null, launch = null, onLaunched, shellCommands }: {
  view: PiView
  onViewChange: (view: PiView) => void
  topBar?: HTMLElement | null
  launch?: PiLaunch | null
  onLaunched?: () => void
  shellCommands?: PiShellCommands
}) {
  const [opened, setOpened] = useState<{ sessionId: string; cwd: string } | null>(null)
  const [firstSubmit, setFirstSubmit] = useState<PiSubmit | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const { sessionId, repo } = view
  const newSession = useNewSessionModels(!sessionId)
  const [choice, setChoice] = useState<{ model: PiModel | null; thinking: string }>({ model: null, thinking: NEW_SESSION_OPTIONS.thinking ?? "medium" })
  const choiceLevels = choice.model ? newSession.levels.get(choice.model) ?? [] : newSession.defaultLevels

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
  }, [sessionId, repo, opened?.sessionId])

  const { model: chosenModel, thinking: chosenThinking } = choice
  const start = useCallback(async (submit: PiSubmit) => {
    if (await runShellCommand(submit, shellCommands)) return
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
  }, [repo, onViewChange, shellCommands, chosenModel, chosenThinking])

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
  const newCommands = useComposerCommands(NO_COMMANDS, shellCommands)
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
        shellCommands={shellCommands}
      />
    )
  }

  const newControls: PiSessionControls = {
    model: choice.model,
    models: newSession.models,
    thinkingLevel: choice.thinking,
    thinkingLevels: choiceLevels,
    stats: null,
    onModel: (model) => setChoice((current) => ({ model, thinking: clampThinking(current.thinking, newSession.levels.get(model) ?? []) })),
    onThinking: (thinking) => setChoice((current) => ({ ...current, thinking })),
  }

  return (
    <section className="flex min-h-0 min-w-0 flex-1 flex-col">
      {inTopBar(<PiTopBar repo={repo} header={null} controls={sessionId ? undefined : newControls} />)}
      <div className="flex flex-1 items-center justify-center">
        {error && <p role="alert" className="text-sm text-danger">{error}</p>}
      </div>
      {!sessionId && (
        <PiComposer
          sessionKey={`new:${repo}`}
          repo={repo}
          running={false}
          disabled={starting}
          commands={newCommands}
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
