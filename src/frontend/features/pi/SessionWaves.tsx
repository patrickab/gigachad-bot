"use client"

import { useEffect, useMemo, useState } from "react"
import { Bot, Check, Circle, CircleAlert, LoaderCircle, X, type LucideIcon } from "lucide-react"
import { Disclosure } from "@/components/ui/Disclosure"
import { IconButton } from "@/components/ui/IconButton"
import { callFactory } from "@/lib/backend/api"
import { concernsFactory, isUnknownAction, waveNeedsUser, waveStageLabel, type FactoryTask, type FactoryWave, type FactoryWaveTask } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { observePiSocket } from "./piSocket"
import { formatCost, formatTokens } from "./subagents"
import type { FactoryCard } from "./factoryCard"

/** A ruling the supervisor gave in this session, from `decisions.list`. */
export interface FactoryDecision { at: string; key?: string; answers?: string; card: FactoryCard }

const RESTING = new Set(["merged", "cancelled"])
const FINISHED_TASK = new Set(["done", "cancelled"])
const ATTENTION_TASK = new Set(["waiting_answer", "waiting_approve", "blocked"])

const newestFirst = (a: FactoryWave, b: FactoryWave) => b.createdAt.localeCompare(a.createdAt)
const totalCost = (waves: readonly FactoryWave[]) => waves.reduce((sum, wave) => sum + (wave.usage?.cost ?? 0), 0)

const storeKey = (sessionId: string) => `gigachad:session-waves:${sessionId}`

/** The session's waves as last read in this browser: shown until the first read answers (`waves.list` takes seconds). */
function storedWaves(sessionId: string): FactoryWave[] {
  try {
    const stored: unknown = JSON.parse(localStorage.getItem(storeKey(sessionId)) ?? "[]")
    return Array.isArray(stored) ? (stored as FactoryWave[]) : []
  } catch {
    // No store (server render, a blocked store) or a broken entry: nothing to show early.
    return []
  }
}

function storeWaves(sessionId: string, waves: readonly FactoryWave[]): void {
  try {
    if (waves.length === 0) localStorage.removeItem(storeKey(sessionId))
    else localStorage.setItem(storeKey(sessionId), JSON.stringify(waves))
  } catch {
    // A full or unavailable store only costs the early display.
  }
}

/** The coder model of a task by task id, without its provider. */
export type TaskModels = Record<string, string>

/** Each task's coder model (`provider/model` shown as `model`); a task without one is left out. */
export function taskModels(tasks: readonly FactoryTask[]): TaskModels {
  const models: TaskModels = {}
  for (const task of tasks) {
    const model = task.settings?.models.coder ?? task.settings?.models.big
    if (model) models[task.id] = model.slice(model.lastIndexOf("/") + 1)
  }
  return models
}

/** The waves session `sessionId` declared (`origin`). */
export function sessionWaves(waves: readonly FactoryWave[], sessionId: string): FactoryWave[] {
  return waves.filter((wave) => wave.origin === sessionId)
}

/** `waves` split into active ones (those needing the user first, then newest first) and finished ones (`merged`, `cancelled`, newest first). */
export function splitWaves(waves: readonly FactoryWave[]): { active: FactoryWave[]; finished: FactoryWave[] } {
  const active = waves.filter((wave) => !RESTING.has(wave.stage)).sort(newestFirst)
  return {
    active: [...active.filter(waveNeedsUser), ...active.filter((wave) => !waveNeedsUser(wave))],
    finished: waves.filter((wave) => RESTING.has(wave.stage)).sort(newestFirst),
  }
}

/**
 * The waves session `sessionId` declared, in whatever repository, from the core's `waves.list` without a repo.
 *
 * - Read on mount, and again on pi-host's `sessions_changed` notice of a factory path and on a reopened socket, as the board does
 * - Until the first read answers, the session's waves as this browser last read them (`localStorage`)
 * - A factory without waves (`unknown action`) has none, another failed read keeps the last list and is reported with `console.error`
 * - `models`: the coder model of each task by task id, from `tasks.list` of each repository the session has waves in
 */
export function useSessionWaves(sessionId: string): { waves: FactoryWave[]; models: TaskModels; decisions: FactoryDecision[] | null } {
  const [decisions, setDecisions] = useState<FactoryDecision[] | null>(null)
  const [answered, setAnswered] = useState<{ sessionId: string; waves: FactoryWave[] } | null>(null)
  const [models, setModels] = useState<TaskModels>({})
  useEffect(() => {
    let cancelled = false
    let latest = 0
    const read = () => {
      const call = ++latest
      const current = () => !cancelled && call === latest
      callFactory<{ waves: FactoryWave[] }>("waves.list", {}).then(
        (answer) => {
          if (!current()) return
          const own = sessionWaves(answer.waves, sessionId)
          setAnswered({ sessionId, waves: own })
          storeWaves(sessionId, own)
          const repos = [...new Set(own.map((wave) => wave.repo))]
          Promise.all(repos.map((repo) => callFactory<{ tasks: FactoryTask[] }>("tasks.list", { repo }))).then(
            (listed) => { if (current()) setModels(taskModels(listed.flatMap((answer) => answer.tasks ?? []))) },
            (cause: unknown) => console.error("tasks.list failed", cause),
          )
        },
        (cause: unknown) => {
          if (!current()) return
          if (isUnknownAction(cause)) setAnswered({ sessionId, waves: [] })
          else console.error("waves.list failed", cause)
        },
      )
      callFactory<{ decisions: FactoryDecision[] }>("decisions.list", { origin: sessionId }).then(
        (answer) => { if (current()) setDecisions(answer.decisions ?? []) },
        (cause: unknown) => {
          if (!current()) return
          if (isUnknownAction(cause)) setDecisions(null)
          else console.error("decisions.list failed", cause)
        },
      )
    }
    read()
    const stop = observePiSocket({ onHostEvent: (event) => { if (concernsFactory(event)) read() }, onOpen: read }, { hold: true })
    return () => {
      cancelled = true
      stop()
    }
  }, [sessionId])
  return useMemo(() => ({ waves: answered?.sessionId === sessionId ? answered.waves : storedWaves(sessionId), models, decisions }), [answered, sessionId, models, decisions])
}

function waveIcon(wave: FactoryWave): { icon: LucideIcon; className: string } {
  if (wave.stage === "merged") return { icon: Check, className: "text-ink-subtle" }
  if (wave.stage === "cancelled") return { icon: X, className: "text-ink-faint" }
  if (waveNeedsUser(wave)) return { icon: CircleAlert, className: "text-warning" }
  return { icon: LoaderCircle, className: wave.running ? "animate-spin text-ink-muted" : "text-ink-muted" }
}

function taskIcon(state: string): { icon: LucideIcon; className: string } {
  if (ATTENTION_TASK.has(state)) return { icon: CircleAlert, className: "text-warning" }
  if (state === "done") return { icon: Check, className: "text-ink-subtle" }
  if (state === "cancelled") return { icon: X, className: "text-ink-faint" }
  if (state === "pending") return { icon: Circle, className: "text-ink-faint" }
  return { icon: LoaderCircle, className: "text-ink-muted" }
}

/** A task's row: status, title, the coder's model, tokens in (with cache reads) and out, state; a started one opens its task view. */
function TaskRow({ task, model, onOpenTask }: { task: FactoryWaveTask; model: string | undefined; onOpenTask: (taskId: string) => void }) {
  const { icon: Icon, className } = taskIcon(task.state)
  const { taskId, usage } = task
  const state = task.state.replace(/_/g, " ")
  return (
    <li className="ml-5 flex items-center gap-2 py-0.5">
      <Icon size={12} aria-hidden="true" className={cn("shrink-0", className)} />
      <span className="min-w-0 flex-1 truncate text-ink-muted">{task.title}</span>
      <span className="shrink-0 tabular-nums text-ink-faint">
        {model && `${model} · `}
        {usage && `↑${formatTokens(usage.input + (usage.cacheRead ?? 0))} ↓${formatTokens(usage.output)} · `}
        <span className={cn(ATTENTION_TASK.has(task.state) && "text-warning")}>{state}</span>
      </span>
      <span className="flex h-5 w-5 shrink-0 items-center justify-center">
        {taskId && <IconButton icon={Bot} label="Open task" onClick={() => onOpenTask(taskId)} className="h-5 w-5" />}
      </span>
    </li>
  )
}

/** A wave's line: status icon, tag, title, `done/total`, stage, cost. */
function WaveSummary({ wave }: { wave: FactoryWave }) {
  const { icon: Icon, className } = waveIcon(wave)
  return (
    <>
      <Icon size={12} aria-hidden="true" className={cn("shrink-0", className)} />
      <span className="shrink-0 text-ink">{wave.tag}</span>
      <span className="min-w-0 flex-1 truncate text-ink-muted">{wave.title}</span>
      <span className="shrink-0 tabular-nums text-ink-faint">
        {wave.counts.done}/{wave.counts.total}
        {" · "}
        <span className={cn(waveNeedsUser(wave) && "text-warning")}>{waveStageLabel(wave)}</span>
        {wave.usage && ` · ${formatCost(wave.usage.cost)}`}
      </span>
      <span className="w-5 shrink-0" />
    </>
  )
}

const ROW = "h-auto py-0.5"
const NO_MODELS: TaskModels = {}

/**
 * What the tab's transcript knows of its factory cards (`factoryRulings`), and its jump to a notice's card.
 * `revealNotice` reveals the notice's turn in this tab's transcript, collapsed or not yet mounted.
 */
export interface SessionCards {
  notices: ReadonlySet<string>
  waiting: readonly string[]
  revealNotice: (key: string) => void
}

const NO_CARDS: SessionCards = { notices: new Set(), waiting: [], revealNotice: () => {} }

/** The session's decisions, each jumping to the card it answers when that card is in this transcript. */
function DecisionsEntry({ decisions, cards }: { decisions: readonly FactoryDecision[]; cards: SessionCards }) {
  return (
    <li>
      <Disclosure summary={<span className="min-w-0 flex-1">Decisions · {decisions.length}</span>} summaryClassName={ROW}>
        <ul aria-label="Decisions" className="ml-5 flex flex-col">
          {decisions.map((decision, index) => {
            const target = decision.answers ?? decision.key
            return (
              <li key={`${decision.at}-${index}`} className="flex items-center py-0.5">
                {target && cards.notices.has(target) ? (
                  <button type="button" onClick={() => cards.revealNotice(target)} className="min-w-0 flex-1 truncate text-left text-ink-muted hover:text-ink focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ink-faint">{decision.card.title}</button>
                ) : (
                  <span className="min-w-0 flex-1 truncate text-ink-muted">{decision.card.title}</span>
                )}
              </li>
            )
          })}
        </ul>
      </Disclosure>
    </li>
  )
}

/** A wave's row, opening to its tasks: the unfinished ones first, then the finished, each started one with its task view. */
function WaveRow({ wave, models, onOpenTask }: { wave: FactoryWave; models: TaskModels; onOpenTask: (repo: string, taskId: string) => void }) {
  const tasks = [...wave.tasks.filter((task) => !FINISHED_TASK.has(task.state)), ...wave.tasks.filter((task) => FINISHED_TASK.has(task.state))]
  return (
    <li>
      <Disclosure summary={<WaveSummary wave={wave} />} summaryClassName={ROW}>
        {tasks.length > 0 && (
          <ul aria-label={`${wave.tag} tasks`} className="flex flex-col">
            {tasks.map((task) => <TaskRow key={task.key} task={task} model={task.taskId ? models[task.taskId] : undefined} onOpenTask={(taskId) => onOpenTask(wave.repo, taskId)} />)}
          </ul>
        )}
      </Disclosure>
    </li>
  )
}

/**
 * The factory waves a Pi session declared, pinned above the composer beside the subagent roster: one element that collapses to its header.
 *
 * - Header: `Factory`, the active waves (`<n> active`, `<m> needs you` in the warning colour when any), their summed cost; `<n> finished` without active waves.
 *   Open while a wave is active, collapsed otherwise
 * - Active waves (not `merged` or `cancelled`), those needing the user first, then newest first
 * - Finished waves behind one collapsed `<n> finished` row with their summed cost, newest first
 * - Every wave's row opens to its tasks, unfinished ones first: the coder's model, tokens in and out, state. A started one opens its task view
 *   (the coder's session, brief and handoff) through `onOpenTask`
 * - `<m> needs you` counts the session's cards waiting for a ruling: the transcript's attention notices no decision answers
 *   (one in the transcript or from `decisions.list`)
 * - A `Decisions` entry lists `decisions`, each jumping to the card it answers through `cards.revealNotice`
 * - Renders nothing without waves, decisions or waiting cards
 */
export function SessionWaves({ waves, models = NO_MODELS, decisions = null, cards = NO_CARDS, onOpenTask }: {
  waves: FactoryWave[]
  models?: TaskModels
  decisions?: readonly FactoryDecision[] | null
  cards?: SessionCards
  onOpenTask: (repo: string, taskId: string) => void
}) {
  const { active, finished } = useMemo(() => splitWaves(waves), [waves])
  const needsUser = useMemo(() => {
    const answered = new Set(decisions?.map((decision) => decision.answers ?? decision.card.answers))
    return cards.waiting.filter((key) => !answered.has(key)).length
  }, [decisions, cards.waiting])
  const hasDecisions = !!decisions && decisions.length > 0
  if (waves.length === 0 && !hasDecisions && needsUser === 0) return null
  const header = (
    <>
      <span className="shrink-0">
        Factory{active.length > 0 ? ` · ${active.length} active` : finished.length > 0 ? ` · ${finished.length} finished` : ""}
        {needsUser > 0 && <> · <span className="text-warning">{needsUser} needs you</span></>}
      </span>
      <span className="flex-1" />
      {active.length > 0 && <span className="shrink-0 tabular-nums text-ink-faint">{formatCost(totalCost(active))}</span>}
      <span className="w-5 shrink-0" />
    </>
  )
  return (
    <section aria-label="Waves" className="mx-auto mb-2 flex max-h-[40vh] w-full max-w-2xl flex-col overflow-y-auto rounded-xl bg-paper px-3 py-1.5 text-xs">
      <Disclosure key={active.length > 0 ? "active" : "finished"} summary={header} summaryClassName={ROW} defaultOpen={active.length > 0 || (waves.length === 0 && needsUser > 0)}>
        <ul className="flex flex-col">
          {hasDecisions && <DecisionsEntry decisions={decisions} cards={cards} />}
          {active.map((wave) => <WaveRow key={wave.id} wave={wave} models={models} onOpenTask={onOpenTask} />)}
          {finished.length > 0 && (
            <li>
              <Disclosure
                summary={(
                  <>
                    <span className="min-w-0 flex-1">{finished.length} finished</span>
                    <span className="shrink-0 tabular-nums text-ink-faint">{formatCost(totalCost(finished))}</span>
                    <span className="w-5 shrink-0" />
                  </>
                )}
                summaryClassName={ROW}
              >
                <ul aria-label="Finished waves" className="ml-5 flex flex-col">
                  {finished.map((wave) => <WaveRow key={wave.id} wave={wave} models={models} onOpenTask={onOpenTask} />)}
                </ul>
              </Disclosure>
            </li>
          )}
        </ul>
      </Disclosure>
    </section>
  )
}

/** The waves of session `sessionId`, read and shown. */
export function SessionWavesPanel({ sessionId, cards, onOpenTask }: { sessionId: string; cards?: SessionCards; onOpenTask: (repo: string, taskId: string) => void }) {
  const { waves, models, decisions } = useSessionWaves(sessionId)
  return <SessionWaves waves={waves} models={models} decisions={decisions} cards={cards} onOpenTask={onOpenTask} />
}
