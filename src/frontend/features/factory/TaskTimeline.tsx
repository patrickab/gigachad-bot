"use client"

import { useMemo } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { SidePanelScroll } from "@/components/ui/SidePanel"
import { StatusDot } from "@/components/ui/StatusDot"
import { formatDuration } from "@/features/pi/tools/format"
import type { FactoryRoundTrigger, FactoryTimelineData, FactoryTimelineEntry, FactoryUsage } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { STATE_LABELS, statusTone } from "./states"
import { formatCost, formatUsage } from "./usage"
import { groupTimeline, type TimelineMessage, type TimelineRound, type TimelineStep } from "./useTaskTimeline"

const TRIGGER_LABELS: Record<FactoryRoundTrigger, string> = {
  start: "start",
  hooks: "after failed hooks",
  gates: "after failed gates",
  review: "after review findings",
  protected: "after a protected file changed",
  message: "after a message",
  retry: "retry",
  conflict: "after a merge conflict",
}

const PHASE_LABELS: Record<FactoryTimelineData["gate"]["phase"], string> = { hooks: "hook", gates: "gate", merge_gates: "merge gate", after_merge: "after merge" }

const REVIEW_FAILURES: Record<Exclude<FactoryTimelineData["review"]["status"], "ok">, string> = {
  malformed: "malformed verdict",
  timeout: "timed out",
  aborted: "aborted",
  failed: "failed",
  no_diff: "no diff",
}

const ROW_CLASS = "flex items-baseline gap-2"
const KIND_CLASS = "w-16 shrink-0 text-ink-subtle"
const METER_CLASS = "shrink-0 tabular-nums text-ink-faint"
const SUMMARY_CLASS = "cursor-pointer text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
const OUTPUT_CLASS = "mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-surface-elevated p-2 font-mono text-[10px] text-ink-muted scrollbar-thin"
const SAID_CLASS = "whitespace-pre-wrap break-words text-ink-muted"

function Verdict({ ok, children }: { ok: boolean; children: string }) {
  return <span className={cn("shrink-0", ok ? "text-diff-added" : "text-diff-removed")}>{children}</span>
}

/** Consecutive coder messages of a round are one turn, a user message stands alone. */
type Said = { kind: "user"; text: string } | { kind: "coder"; texts: string[] }

function conversation(messages: readonly TimelineMessage[]): Said[] {
  const said: Said[] = []
  for (const message of messages) {
    const text = message.data.text.trim()
    if (!text) continue
    const last = said.at(-1)
    if (message.kind === "user") said.push({ kind: "user", text })
    else if (last?.kind === "coder") last.texts.push(text)
    else said.push({ kind: "coder", texts: [text] })
  }
  return said
}

/** A message of the user to the task's coder. */
function UserStep({ text }: { text: string }) {
  return (
    <li className={ROW_CLASS}>
      <span className={KIND_CLASS}>you</span>
      <p className={cn(SAID_CLASS, "min-w-0 flex-1")}>{text}</p>
    </li>
  )
}

/**
 * A coder turn: what it said last (what a Pi session shows when the turn ends), the earlier messages of the turn behind
 * a disclosure. Tool calls and thinking are not part of the timeline.
 */
function CoderStep({ texts }: { texts: readonly string[] }) {
  const earlier = texts.length - 1
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>coder</span>
      </div>
      <p className={cn(SAID_CLASS, "mt-1 max-h-60 overflow-auto scrollbar-thin")}>{texts[earlier]}</p>
      {earlier > 0 && (
        <details className="mt-1">
          <summary className={SUMMARY_CLASS}>{earlier} earlier {earlier === 1 ? "message" : "messages"}</summary>
          <div className="mt-1 flex flex-col gap-2">
            {texts.slice(0, earlier).map((text, index) => <p key={index} className={SAID_CLASS}>{text}</p>)}
          </div>
        </details>
      )}
    </li>
  )
}

/** A hook or gate: its command, duration and result, the output tail behind a disclosure when it failed. */
function GateStep({ gate }: { gate: FactoryTimelineData["gate"] }) {
  const exitCode = gate.exitCode
  const passed = exitCode === 0 && !gate.killed
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>{PHASE_LABELS[gate.phase]}</span>
        <code className="min-w-0 flex-1 truncate font-mono text-ink-muted">{gate.argv.join(" ")}</code>
        <span className={METER_CLASS}>{formatDuration(gate.durationMs)}</span>
        <Verdict ok={passed}>{passed ? "passed" : gate.killed ? "killed" : exitCode === null ? "failed" : `exit ${exitCode}`}</Verdict>
      </div>
      {!passed && gate.outputTail && (
        <details className="mt-1">
          <summary className={SUMMARY_CLASS}>Output</summary>
          <pre className={OUTPUT_CLASS}>{gate.outputTail}</pre>
        </details>
      )}
    </li>
  )
}

/** The review: verdict, cost and duration, its findings open while the round is the current one. */
function ReviewStep({ review, current }: { review: FactoryTimelineData["review"]; current: boolean }) {
  const count = review.findings.length
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>review</span>
        <span className="min-w-0 flex-1 truncate text-ink-muted">{review.model}</span>
        <span className={METER_CLASS}>{formatUsage(review.usage)} · {formatCost(review.usage.cost)} · {formatDuration(review.wallMs)}</span>
        <Verdict ok={review.approved}>{review.status !== "ok" ? REVIEW_FAILURES[review.status] : review.approved ? "approved" : "rejected"}</Verdict>
      </div>
      {count > 0 && (
        <details className="mt-1" open={current}>
          <summary className={SUMMARY_CLASS}>{count} {count === 1 ? "finding" : "findings"}</summary>
          <ul className="mt-1 flex flex-col gap-1.5">
            {review.findings.map((finding, index) => (
              <li key={index}>
                <div className={ROW_CLASS}>
                  <code className="min-w-0 flex-1 truncate font-mono text-ink-muted">{finding.file}:{finding.line}</code>
                  <span className={cn("shrink-0", finding.severity === "minor" ? "text-ink-subtle" : "text-diff-removed")}>{finding.severity}</span>
                </div>
                <p className="whitespace-pre-wrap break-words text-ink-muted">{finding.text}</p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </li>
  )
}

/** The merge: the commit it made or its error behind a disclosure. */
function MergeStep({ merge }: { merge: FactoryTimelineData["merge"] }) {
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>merge</span>
        <code className="min-w-0 flex-1 truncate font-mono text-ink-muted">{merge.ok ? merge.commit.slice(0, 8) : ""}</code>
        <Verdict ok={merge.ok}>{merge.ok ? "done" : "failed"}</Verdict>
      </div>
      {!merge.ok && merge.error && (
        <details className="mt-1">
          <summary className={SUMMARY_CLASS}>Error</summary>
          <pre className={OUTPUT_CLASS}>{merge.error}</pre>
        </details>
      )}
    </li>
  )
}

/** A task the coder started: its title, which opens it. */
function SpawnStep({ spawn, onOpenTask }: { spawn: FactoryTimelineData["spawn"]; onOpenTask: (taskId: string) => void }) {
  return (
    <li className={ROW_CLASS}>
      <span className={KIND_CLASS}>new task</span>
      <button type="button" onClick={() => onOpenTask(spawn.id)} className="min-w-0 flex-1 truncate text-left text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">{spawn.title}</button>
    </li>
  )
}

function Step({ step, current, onOpenTask }: { step: TimelineStep; current: boolean; onOpenTask: (taskId: string) => void }) {
  if (step.kind === "gate") return <GateStep gate={step.data} />
  if (step.kind === "review") return <ReviewStep review={step.data} current={current} />
  if (step.kind === "spawn") return <SpawnStep spawn={step.data} onOpenTask={onOpenTask} />
  return <MergeStep merge={step.data} />
}

/**
 * One round: number, trigger and status, then its steps.
 *
 * - The status is the round's newest state. An earlier round still recorded as running was ended by the next one, whose
 *   trigger says how: it shows no status
 */
function Round({ round, current, onOpenTask }: { round: TimelineRound; current: boolean; onOpenTask: (taskId: string) => void }) {
  const state = round.state && (current || round.state.state !== "running") ? round.state : null
  const said = conversation(round.messages)
  return (
    <li aria-label={`Round ${round.round}`}>
      <div className="flex items-baseline gap-2 text-xs">
        <span className="font-medium text-ink">Round {round.round}</span>
        <span className="min-w-0 flex-1 truncate text-[11px] text-ink-subtle">{round.trigger ? TRIGGER_LABELS[round.trigger] : ""}</span>
        {state && (
          <span className="flex shrink-0 items-center gap-1.5 text-[11px] text-ink-subtle">
            <StatusDot tone={statusTone(state.state)} />
            {STATE_LABELS[state.state]}{state.state === "blocked" && state.reason ? ` · ${state.reason}` : ""}
          </span>
        )}
      </div>
      {(said.length > 0 || round.steps.length > 0) && (
        <ul className="mt-1.5 flex flex-col gap-1.5 text-[11px]">
          {said.map((one, index) => one.kind === "user" ? <UserStep key={`said-${index}`} text={one.text} /> : <CoderStep key={`said-${index}`} texts={one.texts} />)}
          {round.steps.map((step, index) => <Step key={`step-${index}`} step={step} current={current} onOpenTask={onOpenTask} />)}
        </ul>
      )}
    </li>
  )
}

/**
 * A task's pipeline, compact: totals, then each round with its trigger, status, what the user and the coder wrote,
 * gates, review, merge and the tasks the coder started.
 *
 * - `entries` are the task's timeline in append order (`useTaskTimeline`), `usage` the core's total of the task
 * - A started task's title opens it (`onOpenTask`)
 * - Details stay behind disclosures: a failed command's output, a merge error, the findings of earlier rounds
 */
export function TaskTimeline({ entries, usage, error, onOpenTask }: {
  entries: readonly FactoryTimelineEntry[]
  usage: FactoryUsage | null
  error: string
  onOpenTask: (taskId: string) => void
}) {
  const { rounds, review } = useMemo(() => groupTimeline(entries), [entries])
  return (
    <SidePanelScroll className="px-4 pb-4 pt-1">
      <div className="flex flex-col gap-4">
        <ErrorText className="text-[11px]">{error}</ErrorText>
        {rounds.length > 0 && (
          <p aria-label="Totals" className="text-[11px] tabular-nums text-ink-subtle">
            {rounds.length} {rounds.length === 1 ? "round" : "rounds"}
            {usage && ` · ${formatUsage(usage)} · ${formatCost(usage.cost)}`}
            {review.cost > 0 && ` (review ${formatCost(review.cost)})`}
          </p>
        )}
        <ol aria-label="Rounds" className="flex flex-col gap-4">
          {rounds.map((round, index) => <Round key={round.round} round={round} current={index === rounds.length - 1} onOpenTask={onOpenTask} />)}
        </ol>
      </div>
    </SidePanelScroll>
  )
}
