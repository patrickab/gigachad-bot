"use client"

import { useMemo } from "react"
import { ErrorText } from "@/components/ui/ErrorText"
import { SidePanelScroll } from "@/components/ui/SidePanel"
import { StatusDot } from "@/components/ui/StatusDot"
import { formatDuration } from "@/features/pi/tools/format"
import type { FactoryEntry, FactoryEntryData, FactoryRoundTrigger } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { STATE_LABELS, statusTone } from "./states"
import { groupTimeline, type TimelineRound, type TimelineStep } from "./useTaskTimeline"

const TRIGGER_LABELS: Record<FactoryRoundTrigger, string> = {
  start: "start",
  hooks: "after failed hooks",
  gates: "after failed gates",
  review: "after review findings",
  protected: "after a protected file changed",
  message: "after a message",
  retry: "retry",
}

/** By gate phase. The extension also writes `after_merge`, which the domain type does not name yet. */
const PHASE_LABELS: Record<string, string> = { hooks: "hook", gates: "gate", merge_gates: "merge gate", after_merge: "after merge" }

const REVIEW_FAILURES: Record<Exclude<FactoryEntryData["factory-review"]["status"], "ok">, string> = {
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

/** Dollars, to the tenth of a cent below ten cents: a review on a small model costs less than one cent. */
function formatCost(cost: number): string {
  return `$${cost.toFixed(cost < 0.1 ? 3 : 2)}`
}

function Verdict({ ok, children }: { ok: boolean; children: string }) {
  return <span className={cn("shrink-0", ok ? "text-diff-added" : "text-diff-removed")}>{children}</span>
}

/** A hook or gate: its command, duration and result, the output tail behind a disclosure when it failed. */
function GateStep({ gate }: { gate: FactoryEntryData["factory-gate"] }) {
  // The extension writes a null `exitCode` for a command that did not exit by itself, which the domain type does not name yet
  const exitCode: number | null = gate.exitCode
  const passed = exitCode === 0 && !gate.killed
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>{PHASE_LABELS[gate.phase] ?? gate.phase}</span>
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
function ReviewStep({ review, current }: { review: FactoryEntryData["factory-review"]; current: boolean }) {
  const count = review.findings.length
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>review</span>
        <span className="min-w-0 flex-1 truncate text-ink-muted">{review.model}</span>
        <span className={METER_CLASS}>{formatCost(review.usage.cost)} · {formatDuration(review.wallMs)}</span>
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

/** The merge or commit: the commit it made, or its error behind a disclosure. */
function MergeStep({ merge }: { merge: FactoryEntryData["factory-merge"] }) {
  return (
    <li>
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>{merge.kind}</span>
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

function Step({ step, current }: { step: TimelineStep; current: boolean }) {
  if (step.customType === "factory-gate") return <GateStep gate={step.data} />
  if (step.customType === "factory-review") return <ReviewStep review={step.data} current={current} />
  return <MergeStep merge={step.data} />
}

/**
 * One round: number, trigger and status, then its steps.
 *
 * - The status is the round's newest state. An earlier round still recorded as running was ended by the next one, whose
 *   trigger says how: it shows no status
 */
function Round({ round, current }: { round: TimelineRound; current: boolean }) {
  const state = round.state && (current || round.state.state !== "running") ? round.state : null
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
      {round.steps.length > 0 && <ul className="mt-1.5 flex flex-col gap-1.5 text-[11px]">{round.steps.map((step) => <Step key={step.id} step={step} current={current} />)}</ul>}
    </li>
  )
}

/**
 * A task's pipeline, compact: totals, then each round with its trigger, status, gates, review and merge.
 *
 * - `entries` are the factory entries of the task's session in append order (`useTaskTimeline`)
 * - Details stay behind disclosures: a failed command's output, a merge error, the findings of earlier rounds
 */
export function TaskTimeline({ entries, error }: { entries: readonly FactoryEntry[]; error: string }) {
  const { rounds, reviewCost } = useMemo(() => groupTimeline(entries), [entries])
  return (
    <SidePanelScroll className="px-4 pb-4 pt-1">
      <div className="flex flex-col gap-4">
        <ErrorText className="text-[11px]">{error}</ErrorText>
        {rounds.length > 0 && (
          <p aria-label="Totals" className="text-[11px] tabular-nums text-ink-subtle">
            {rounds.length} {rounds.length === 1 ? "round" : "rounds"} · review {formatCost(reviewCost)}
          </p>
        )}
        <ol aria-label="Rounds" className="flex flex-col gap-4">
          {rounds.map((round, index) => <Round key={round.round} round={round} current={index === rounds.length - 1} />)}
        </ol>
      </div>
    </SidePanelScroll>
  )
}
