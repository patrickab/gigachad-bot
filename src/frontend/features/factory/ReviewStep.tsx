"use client"

import type { ReactNode } from "react"
import { formatDuration } from "@/features/pi/tools/format"
import { cn } from "@/lib/utils"
import { formatCost, formatUsage } from "./usage"

// The rows of a factory timeline shared by a task's timeline and a wave's outcome: their classes, the verdict and the review.

export const ROW_CLASS = "flex items-baseline gap-2"
export const KIND_CLASS = "w-14 shrink-0 text-ink-subtle"
export const METER_CLASS = "shrink-0 tabular-nums text-ink-faint"
export const SUMMARY_CLASS = "cursor-pointer text-ink-subtle hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
export const OUTPUT_CLASS = "mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-surface-elevated p-2 font-mono text-[10px] text-ink-muted scrollbar-thin"

/** The words of a review that ended without a verdict, by its status. */
const REVIEW_FAILURES: Record<string, string> = {
  malformed: "malformed verdict",
  timeout: "timed out",
  aborted: "aborted",
  failed: "failed",
  no_diff: "no diff",
}

/** What a failed review's status means; an unknown status as it is. */
export function reviewFailure(status: string): string {
  return REVIEW_FAILURES[status] ?? (status || "failed")
}

export function Verdict({ ok, children }: { ok: boolean; children: string }) {
  return <span className={cn("shrink-0", ok ? "text-diff-added" : "text-diff-removed")}>{children}</span>
}

/** One review as a timeline row shows it, whether of a task's round or of a wave. */
export interface ShownReview {
  model: string | null
  usage: { input: number; output: number; cost: number } | null
  wallMs: number
  /** Approved. */
  ok: boolean
  /** The verdict's word: approved, rejected, or why the review failed. */
  outcome: string
  findings: readonly { severity: "blocker" | "major" | "minor"; file: string; line: number | null; text: string }[]
}

/** A review: model, usage, duration and verdict, then `action` (a link to its session), its findings open when `open`. */
export function ReviewStep({ review, open, action }: { review: ShownReview; open: boolean; action?: ReactNode }) {
  const count = review.findings.length
  const usage = review.usage
  return (
    <li className="group">
      <div className={ROW_CLASS}>
        <span className={KIND_CLASS}>review</span>
        <span className="min-w-0 flex-1 truncate text-ink-muted">{review.model ?? ""}</span>
        <span className={METER_CLASS}>
          {usage && `${formatUsage(usage)} · ${formatCost(usage.cost)} · `}{formatDuration(review.wallMs)}
        </span>
        <Verdict ok={review.ok}>{review.outcome}</Verdict>
        {action}
      </div>
      {count > 0 && (
        <details className="mt-1" open={open}>
          <summary className={SUMMARY_CLASS}>{count} {count === 1 ? "finding" : "findings"}</summary>
          <ul className="mt-1 flex flex-col gap-1.5">
            {review.findings.map((finding, index) => (
              <li key={index}>
                <div className={ROW_CLASS}>
                  <code className="min-w-0 flex-1 truncate font-mono text-ink-muted">{finding.line === null ? finding.file : `${finding.file}:${finding.line}`}</code>
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
