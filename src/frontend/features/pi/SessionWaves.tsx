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

const RESTING = new Set(["merged", "cancelled"])
const FINISHED_TASK = new Set(["done", "cancelled"])
const ATTENTION_TASK = new Set(["waiting_answer", "waiting_approve", "blocked"])

const newestFirst = (a: FactoryWave, b: FactoryWave) => b.createdAt.localeCompare(a.createdAt)
const totalCost = (waves: readonly FactoryWave[]) => waves.reduce((sum, wave) => sum + (wave.usage?.cost ?? 0), 0)

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
 * The waves session `sessionId` declared in `repo` (an absolute path), from the core's `waves.list`.
 *
 * - Read on mount, and again on pi-host's `sessions_changed` notice of a factory path and on a reopened socket, as the board does
 * - A factory without waves (`unknown action`) has none, another failed read keeps the last list and is reported with `console.error`
 * - `models`: the coder model of each task by task id, from `tasks.list`, read only when the session has waves
 */
export function useSessionWaves(repo: string, sessionId: string): { waves: FactoryWave[]; models: TaskModels } {
  const [waves, setWaves] = useState<FactoryWave[]>([])
  const [models, setModels] = useState<TaskModels>({})
  useEffect(() => {
    let cancelled = false
    let latest = 0
    const read = () => {
      const call = ++latest
      const current = () => !cancelled && call === latest
      callFactory<{ waves: FactoryWave[] }>("waves.list", { repo }).then(
        (answer) => {
          if (!current()) return
          setWaves(answer.waves)
          if (sessionWaves(answer.waves, sessionId).length === 0) return
          callFactory<{ tasks: FactoryTask[] }>("tasks.list", { repo }).then(
            (listed) => { if (current()) setModels(taskModels(listed.tasks ?? [])) },
            (cause: unknown) => console.error("tasks.list failed", cause),
          )
        },
        (cause: unknown) => {
          if (!current()) return
          if (isUnknownAction(cause)) setWaves([])
          else console.error("waves.list failed", cause)
        },
      )
    }
    read()
    const stop = observePiSocket({ onHostEvent: (event) => { if (concernsFactory(event)) read() }, onOpen: read }, { hold: true })
    return () => {
      cancelled = true
      stop()
    }
  }, [repo, sessionId])
  return useMemo(() => ({ waves: sessionWaves(waves, sessionId), models }), [waves, sessionId, models])
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

/** A wave's row, opening to its tasks: the unfinished ones first, then the finished, each started one with its task view. */
function WaveRow({ wave, models, onOpenTask }: { wave: FactoryWave; models: TaskModels; onOpenTask: (taskId: string) => void }) {
  const tasks = [...wave.tasks.filter((task) => !FINISHED_TASK.has(task.state)), ...wave.tasks.filter((task) => FINISHED_TASK.has(task.state))]
  return (
    <li>
      <Disclosure summary={<WaveSummary wave={wave} />} summaryClassName={ROW}>
        {tasks.length > 0 && (
          <ul aria-label={`${wave.tag} tasks`} className="flex flex-col">
            {tasks.map((task) => <TaskRow key={task.key} task={task} model={task.taskId ? models[task.taskId] : undefined} onOpenTask={onOpenTask} />)}
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
 * - Renders nothing without waves
 */
export function SessionWaves({ waves, models = NO_MODELS, onOpenTask }: { waves: FactoryWave[]; models?: TaskModels; onOpenTask: (taskId: string) => void }) {
  const { active, finished } = useMemo(() => splitWaves(waves), [waves])
  if (waves.length === 0) return null
  const needsUser = active.filter(waveNeedsUser).length
  const header = (
    <>
      <span className="shrink-0">
        Factory · {active.length > 0 ? `${active.length} active` : `${finished.length} finished`}
        {needsUser > 0 && <> · <span className="text-warning">{needsUser} needs you</span></>}
      </span>
      <span className="flex-1" />
      {active.length > 0 && <span className="shrink-0 tabular-nums text-ink-faint">{formatCost(totalCost(active))}</span>}
      <span className="w-5 shrink-0" />
    </>
  )
  return (
    <section aria-label="Waves" className="mx-auto mb-2 flex w-full max-w-2xl flex-col rounded-xl bg-paper px-3 py-1.5 text-xs">
      <Disclosure key={active.length > 0 ? "active" : "finished"} summary={header} summaryClassName={ROW} defaultOpen={active.length > 0}>
        <ul className="flex flex-col">
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

/** The waves of session `sessionId` in `repo`, read and shown. */
export function SessionWavesPanel({ repo, sessionId, onOpenTask }: { repo: string; sessionId: string; onOpenTask: (taskId: string) => void }) {
  const { waves, models } = useSessionWaves(repo, sessionId)
  return <SessionWaves waves={waves} models={models} onOpenTask={onOpenTask} />
}
