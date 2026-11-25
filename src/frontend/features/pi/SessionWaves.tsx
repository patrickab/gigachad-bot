"use client"

import { useEffect, useMemo, useState } from "react"
import { Bot, ChevronDown, ChevronRight, Circle, CircleAlert, LoaderCircle, type LucideIcon } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { callFactory } from "@/lib/backend/api"
import { concernsFactory, isUnknownAction, waveNeedsUser, waveStageLabel, type FactoryWave, type FactoryWaveTask } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { observePiSocket } from "./piSocket"
import { formatCost } from "./subagents"

const RESTING = new Set(["merged", "cancelled"])
const FINISHED_TASK = new Set(["done", "cancelled"])
const ATTENTION_TASK = new Set(["waiting_answer", "waiting_approve", "blocked"])

/** The unfinished waves session `sessionId` declared (`origin`), oldest first. */
export function sessionWaves(waves: readonly FactoryWave[], sessionId: string): FactoryWave[] {
  return waves.filter((wave) => wave.origin === sessionId && !RESTING.has(wave.stage))
}

/**
 * The unfinished waves session `sessionId` declared in `repo`, from the core's `waves.list`.
 *
 * - Read on mount, and again on pi-host's `sessions_changed` notice of a factory path and on a reopened socket, as the board does
 * - A factory without waves (`unknown action`) has none, another failed read keeps the last list
 */
export function useSessionWaves(repo: string, sessionId: string): FactoryWave[] {
  const [waves, setWaves] = useState<FactoryWave[]>([])
  useEffect(() => {
    let cancelled = false
    let latest = 0
    const read = () => {
      const call = ++latest
      const current = () => !cancelled && call === latest
      callFactory<{ waves: FactoryWave[] }>("waves.list", { repo }).then(
        (answer) => { if (current()) setWaves(answer.waves) },
        (cause: unknown) => { if (current() && isUnknownAction(cause)) setWaves([]) },
      )
    }
    read()
    const stop = observePiSocket({ onHostEvent: (event) => { if (concernsFactory(event)) read() }, onOpen: read }, { hold: true })
    return () => {
      cancelled = true
      stop()
    }
  }, [repo])
  return useMemo(() => sessionWaves(waves, sessionId), [waves, sessionId])
}

function waveIcon(wave: FactoryWave): { icon: LucideIcon; className: string } {
  if (waveNeedsUser(wave)) return { icon: CircleAlert, className: "text-warning" }
  return { icon: LoaderCircle, className: wave.running ? "animate-spin text-ink-muted" : "text-ink-muted" }
}

function taskIcon(state: string): { icon: LucideIcon; className: string } {
  if (ATTENTION_TASK.has(state)) return { icon: CircleAlert, className: "text-warning" }
  if (state === "pending") return { icon: Circle, className: "text-ink-faint" }
  return { icon: LoaderCircle, className: "text-ink-muted" }
}

function TaskRow({ task, onOpenSession }: { task: FactoryWaveTask; onOpenSession: (sessionId: string) => void }) {
  const { icon: Icon, className } = taskIcon(task.state)
  const { sessionId } = task
  const state = task.state.replace(/_/g, " ")
  return (
    <li className="ml-5 flex items-center gap-2 py-0.5">
      <Icon size={12} aria-hidden="true" className={cn("shrink-0", className)} />
      <span className="min-w-0 flex-1 truncate text-ink-muted">{task.title}</span>
      <span className={cn("shrink-0", ATTENTION_TASK.has(task.state) ? "text-warning" : "text-ink-faint")}>{state}</span>
      <span className="flex h-5 w-5 shrink-0 items-center justify-center">
        {sessionId && <IconButton icon={Bot} label="Open session" onClick={() => onOpenSession(sessionId)} className="h-5 w-5" />}
      </span>
    </li>
  )
}

function WaveRow({ wave, open, onToggle, onOpenSession }: { wave: FactoryWave; open: boolean; onToggle: () => void; onOpenSession: (sessionId: string) => void }) {
  const { icon: Icon, className } = waveIcon(wave)
  const Chevron = open ? ChevronDown : ChevronRight
  const tasks = wave.tasks.filter((task) => !FINISHED_TASK.has(task.state))
  return (
    <li>
      <button type="button" aria-expanded={open} onClick={onToggle} className="flex w-full min-w-0 items-center gap-2 rounded py-0.5 text-left hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
        <Chevron size={12} aria-hidden="true" className="-mr-1 shrink-0 text-ink-faint" />
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
      </button>
      {open && tasks.length > 0 && (
        <ul aria-label={`${wave.tag} tasks`} className="flex flex-col">
          {tasks.map((task) => <TaskRow key={task.key} task={task} onOpenSession={onOpenSession} />)}
        </ul>
      )}
    </li>
  )
}

/**
 * The unfinished waves a Pi session declared, pinned above the composer beside the subagent roster and in its shape.
 *
 * - One collapsed row per wave: status icon, tag, title, `done/total`, stage, cost. A stage that needs the user shows in the warning colour
 * - A row opens to its tasks that are not done or cancelled, each with its state; a started one opens its session through `onOpenSession`
 * - Renders nothing without waves
 */
export function SessionWaves({ waves, onOpenSession }: { waves: FactoryWave[]; onOpenSession: (sessionId: string) => void }) {
  const [openId, setOpenId] = useState<string | null>(null)
  if (waves.length === 0) return null
  return (
    <section aria-label="Waves" className="mx-auto mb-2 flex w-full max-w-2xl flex-col rounded-xl bg-paper px-3 py-1.5 text-xs">
      <ul className="flex flex-col">
        {waves.map((wave) => (
          <WaveRow
            key={wave.id}
            wave={wave}
            open={openId === wave.id}
            onToggle={() => setOpenId((current) => current === wave.id ? null : wave.id)}
            onOpenSession={onOpenSession}
          />
        ))}
      </ul>
    </section>
  )
}

/** The waves of session `sessionId` in `repo`, read and shown. */
export function SessionWavesPanel({ repo, sessionId, onOpenSession }: { repo: string; sessionId: string; onOpenSession: (sessionId: string) => void }) {
  const waves = useSessionWaves(repo, sessionId)
  return <SessionWaves waves={waves} onOpenSession={onOpenSession} />
}
