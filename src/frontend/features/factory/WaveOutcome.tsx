"use client"

import { Bot } from "lucide-react"
import { formatDuration } from "@/features/pi/tools/format"
import type { FactoryWave, FactoryWaveReview } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { KIND_CLASS, METER_CLASS, OUTPUT_CLASS, ReviewStep, reviewFailure, ROW_CLASS, SUMMARY_CLASS, Verdict, type ShownReview } from "./ReviewStep"
import { ICON_BUTTON_CLASS } from "./styles"

/** A link to a Pi session: quiet until its row is hovered or it has focus. */
export function SessionButton({ label, sessionId, onOpenSession }: { label: string; sessionId: string; onOpenSession: (sessionId: string) => void }) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={() => onOpenSession(sessionId)}
      className={cn(ICON_BUTTON_CLASS, "self-center p-0.5 opacity-0 focus-visible:opacity-100 group-hover:opacity-100")}
    >
      <Bot size={12} />
    </button>
  )
}

function shownReview(review: FactoryWaveReview): ShownReview {
  const ok = review.verdict === "approved"
  return { ...review, ok, outcome: review.verdict === "failed" ? reviewFailure(review.status) : review.verdict }
}

const newestFirst = (left: FactoryWaveReview, right: FactoryWaveReview) => right.round - left.round || right.at.localeCompare(left.at)

/** The short form of a commit. */
const short = (sha: string) => sha.slice(0, 8)

/**
 * What happened to a wave as a whole, below its tasks: its reviews newest first (the newest one's findings open, a link
 * to the reviewer's session when its id is known), the gate runs that failed, the commit range that reached the target
 * and the remote request. Nothing when none of them happened yet.
 */
export function WaveOutcome({ wave, onOpenSession }: { wave: FactoryWave; onOpenSession: (sessionId: string) => void }) {
  const reviews = [...wave.reviews].sort(newestFirst)
  const failedGates = wave.gates.filter((gate) => !gate.passed)
  const remoteUrl = wave.remote?.url ?? null
  if (reviews.length === 0 && !wave.merge && failedGates.length === 0 && !remoteUrl) return null
  return (
    <ul aria-label="Outcome" className="flex flex-col gap-1.5 py-1 text-[11px]">
      {reviews.map((review, index) => (
        <ReviewStep
          key={`${review.round}-${review.at}`}
          review={shownReview(review)}
          open={index === 0}
          action={review.sessionId ? <SessionButton label="Open reviewer session" sessionId={review.sessionId} onOpenSession={onOpenSession} /> : null}
        />
      ))}
      {failedGates.map((gate) => (
        <li key={`${gate.phase}-${gate.at}`}>
          <div className={ROW_CLASS}>
            <span className={KIND_CLASS}>{gate.phase} gate</span>
            <span className="min-w-0 flex-1" />
            <span className={METER_CLASS}>{formatDuration(gate.wallMs)}</span>
            <Verdict ok={false}>failed</Verdict>
          </div>
          {gate.log && (
            <details className="mt-1">
              <summary className={SUMMARY_CLASS}>Output</summary>
              <pre className={OUTPUT_CLASS}>{gate.log}</pre>
            </details>
          )}
        </li>
      ))}
      {wave.merge && (
        <li className={ROW_CLASS}>
          <span className={KIND_CLASS}>merge</span>
          <code aria-label="Merged range" className="min-w-0 flex-1 truncate font-mono text-ink-muted">{short(wave.merge.from)}..{short(wave.merge.to)}</code>
        </li>
      )}
      {remoteUrl && (
        <li className={ROW_CLASS}>
          <span className={KIND_CLASS}>remote</span>
          <a
            href={remoteUrl}
            target="_blank"
            rel="noreferrer"
            className="min-w-0 flex-1 truncate text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
          >
            {wave.remote?.number != null ? `#${wave.remote.number}` : remoteUrl}
          </a>
        </li>
      )}
    </ul>
  )
}
