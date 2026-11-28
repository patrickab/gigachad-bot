"use client"

import { useEffect, useMemo, useState } from "react"
import { Disclosure } from "@/components/ui/Disclosure"
import { outlineOfWaves, type OutlineFact, type OutlineMark, type OutlineRow, type OutlineTone } from "./waveOutline"
import { useNowMs } from "@/hooks/useNowMs"
import { callFactory } from "@/lib/backend/api"
import { concernsFactory, type FactoryTask, type FactoryWave } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { observePiSocket } from "./piSocket"
import { factoryCard, type FactoryCard } from "./factoryCard"

/** A ruling the supervisor gave in this session, from `decisions.list`. */
export interface FactoryDecision { at: string; key?: string; answers?: string; card: FactoryCard }

/** The notice key a decision answers: the one place it is read (the decision's own `key` is never a notice key). */
function decisionAnswers(decision: FactoryDecision): string | undefined {
  return decision.answers ?? decision.card.answers
}

/** The decisions of a `decisions.list` answer, each card validated by `factoryCard`; malformed entries are dropped. */
function validDecisions(listed: unknown): FactoryDecision[] {
  if (!Array.isArray(listed)) return []
  return listed.flatMap((entry: unknown) => {
    if (typeof entry !== "object" || entry === null) return []
    const card = factoryCard({ card: (entry as { card?: unknown }).card })
    return card ? [{ ...(entry as FactoryDecision), card }] : []
  })
}

const RESTING = new Set(["merged", "cancelled"])

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
    const model = task.settings?.models.big
    if (model) models[task.id] = model.slice(model.lastIndexOf("/") + 1)
  }
  return models
}

/** The waves session `sessionId` declared (`origin`). */
export function sessionWaves(waves: readonly FactoryWave[], sessionId: string): FactoryWave[] {
  return waves.filter((wave) => wave.origin === sessionId)
}

/**
 * The waves session `sessionId` declared, in whatever repository, from the core's `waves.list` without a repo.
 *
 * - Read on mount, and again on pi-host's `sessions_changed` notice of a factory path and on a reopened socket, as the board does
 * - Until the first read answers, the session's waves as this browser last read them (`localStorage`)
 * - A failed read keeps the last list and is reported with `console.error`
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
        (cause: unknown) => { if (current()) console.error("waves.list failed", cause) },
      )
      callFactory<{ decisions?: unknown }>("decisions.list", { origin: sessionId }).then(
        (answer) => { if (current()) setDecisions(validDecisions(answer.decisions)) },
        (cause: unknown) => { if (current()) console.error("decisions.list failed", cause) },
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

const TONE_CLASS: Record<OutlineTone, string> = { ink: "text-ink", muted: "text-ink-muted", faint: "text-ink-faint", warning: "text-warning", danger: "text-danger" }

const MARKS: Record<Exclude<OutlineMark, "working">, { glyph: string; className: string; label: string }> = {
  done: { glyph: "✓", className: "text-diff-added", label: "Done" },
  cancelled: { glyph: "✗", className: "text-ink-faint", label: "Cancelled" },
  attention: { glyph: "!", className: "text-warning", label: "Needs you" },
  failed: { glyph: "!", className: "text-danger", label: "Failed" },
  waiting: { glyph: "○", className: "text-ink-faint", label: "Waiting" },
}

/** A row's mark in a cell of one width: a turning ring while something works, else its glyph. */
function Mark({ mark }: { mark: OutlineMark }) {
  if (mark === "working") {
    return (
      <span role="img" aria-label="Working" className="flex w-3 shrink-0 justify-center">
        <span className="h-2 w-2 animate-spin rounded-full border border-ink-faint border-t-ink-muted motion-reduce:animate-none" />
      </span>
    )
  }
  const { glyph, className, label } = MARKS[mark]
  return <span role="img" aria-label={label} className={cn("w-3 shrink-0 text-center", className)}>{glyph}</span>
}

/** Facts on one quiet line, each after a dot. */
function Facts({ facts }: { facts: readonly OutlineFact[] }) {
  return (
    <span className="shrink-0 whitespace-pre tabular-nums">
      {facts.map((item, index) => (
        <span key={index}>
          <span className="text-ink-faint"> · </span>
          <span className={TONE_CLASS[item.tone]}>{item.text}</span>
        </span>
      ))}
    </span>
  )
}

const OUTLINE_ROW = "flex h-5 w-full min-w-0 items-center rounded text-left"
const INTERACTIVE = "hover:bg-hover focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** One line of the tree: tree lines, mark, label, title, facts. A row with rows below opens and closes them, a started task opens its task view. */
function OutlineLine({ row, onToggle, onOpenTask }: { row: OutlineRow; onToggle: (id: string) => void; onOpenTask: (repo: string, taskId: string) => void }) {
  const quiet = row.mark === "done" || row.mark === "cancelled" || row.mark === "waiting"
  const content = (
    <>
      <span aria-hidden="true" className="shrink-0 whitespace-pre text-ink-faint">{row.lead}{row.open === null ? "" : row.open ? "▾ " : "▸ "}</span>
      {row.mark && <Mark mark={row.mark} />}
      <span className={cn("shrink-0", row.mark && "ml-1.5", row.group ? "font-semibold text-ink" : quiet || row.mark === null ? "text-ink-muted" : "text-ink")}>{row.label}</span>
      {row.title && <span className={cn("ml-3 min-w-0 truncate", quiet ? "text-ink-faint" : "text-ink-muted")}>{row.title}</span>}
      <Facts facts={row.facts} />
    </>
  )
  const { toggle, task } = row
  if (toggle !== null) return <li><button type="button" aria-expanded={row.open ?? false} onClick={() => onToggle(toggle)} className={cn(OUTLINE_ROW, INTERACTIVE)}>{content}</button></li>
  if (task !== null) return <li><button type="button" onClick={() => onOpenTask(task.repo, task.taskId)} className={cn(OUTLINE_ROW, INTERACTIVE)}>{content}</button></li>
  return <li className={OUTLINE_ROW}>{content}</li>
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
            const target = decisionAnswers(decision)
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

/**
 * The factory waves a Pi session declared, pinned above the composer beside the subagent roster: one element that collapses to its header.
 *
 * - Header: `Factory`, the spend of all its waves, the coders and reviewers at work, `<n> active` (`<n> finished` without active waves),
 *   `<m> needs you` in the warning colour when any. Open while a wave is active, collapsed otherwise
 * - Below, the waves as one tree (`outlineOfWaves`): tracks and milestones from the tags' prefixes, waves, their tasks, a line each with
 *   its mark and facts. What needs the user comes first, what is finished stands behind one closed `<n> finished` row per group
 * - A click on a row opens or closes what is below it; one on a started task opens its task view (the coder's session, brief and
 *   handoff) through `onOpenTask`
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
  const [toggled, setToggled] = useState<ReadonlySet<string>>(() => new Set())
  const now = useNowMs()
  const outline = useMemo(() => outlineOfWaves(waves, toggled, now, models), [waves, toggled, now, models])
  const active = waves.some((wave) => !RESTING.has(wave.stage))
  const needsUser = useMemo(() => {
    const answered = new Set(decisions?.map(decisionAnswers))
    return cards.waiting.filter((key) => !answered.has(key)).length
  }, [decisions, cards.waiting])
  const hasDecisions = !!decisions && decisions.length > 0
  if (waves.length === 0 && !hasDecisions && needsUser === 0) return null
  const toggle = (id: string) => setToggled((ids) => {
    const next = new Set(ids)
    if (!next.delete(id)) next.add(id)
    return next
  })
  const header = (
    <span className="min-w-0 flex-1 truncate">
      <span className="text-ink">Factory</span>
      <Facts facts={needsUser > 0 ? [...outline.summary, { text: `${needsUser} needs you`, tone: "warning" }] : outline.summary} />
    </span>
  )
  return (
    <section aria-label="Waves" className="mx-auto mb-2 flex max-h-[40vh] w-full max-w-2xl flex-col overflow-y-auto rounded-xl bg-paper px-3 py-1.5 text-xs">
      <Disclosure key={active ? "active" : "finished"} summary={header} summaryClassName={ROW} defaultOpen={active || (waves.length === 0 && needsUser > 0)}>
        <ul className="flex flex-col">
          {hasDecisions && <DecisionsEntry decisions={decisions} cards={cards} />}
        </ul>
        <ul aria-label="Waves" className="flex flex-col font-mono">
          {outline.rows.map((row) => <OutlineLine key={row.key} row={row} onToggle={toggle} onOpenTask={onOpenTask} />)}
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
