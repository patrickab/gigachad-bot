/*
 * Ported from omp-web `lib/message-update-coalescer.ts` (https://github.com/ompweb-dev/ompweb, MIT, Copyright (c) 2026 agegr).
 * Pi streams deltas, not whole messages: every buffered record is kept and applied in one `events` action instead of only the latest.
 */
import type { TranscriptAction } from "./transcript"

/** Schedules `flush` and returns a cancel function. */
export type FlushScheduler = (flush: () => void) => () => void

/** Least time between two flushes of one coalescer: a 120 or 144 Hz display would else render a streamed reply twice as often as it reads any smoother. */
const MIN_FLUSH_MS = 12
const lastFlushes = new WeakMap<() => void, number>()

/**
 * Next animation frame, the one after when `flush` last ran under `MIN_FLUSH_MS` ago;
 * a hidden tab stalls `requestAnimationFrame`, so it (and a non-browser) gets a trailing 50 ms timer.
 */
export function scheduleAtDisplayRate(flush: () => void): () => void {
  if (typeof document !== "undefined" && !document.hidden && typeof requestAnimationFrame === "function") {
    const tick = (now: number) => {
      if (now - (lastFlushes.get(flush) ?? -Infinity) < MIN_FLUSH_MS) {
        id = requestAnimationFrame(tick)
        return
      }
      lastFlushes.set(flush, now)
      flush()
    }
    let id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }
  const id = setTimeout(flush, 50)
  return () => clearTimeout(id)
}

export interface TranscriptCoalescer {
  /** Buffer a streaming record until the next flush, or flush the buffer and dispatch any other action right away. */
  push(action: TranscriptAction): void
  /** Drop the buffer and cancel the scheduled flush (unmount). */
  reset(): void
}

/**
 * Batches streaming records into one `events` action per animation frame, so a burst of deltas renders once.
 *
 * Every other action (`message_end`, snapshot, resync, ...) flushes the buffer before it is dispatched, so actions apply in arrival order.
 */
export function createTranscriptCoalescer(dispatch: (action: TranscriptAction) => void, schedule: FlushScheduler = scheduleAtDisplayRate): TranscriptCoalescer {
  let buffered: Extract<TranscriptAction, { kind: "event" }>["event"][] = []
  let cancelScheduled: (() => void) | null = null

  const flush = () => {
    cancelScheduled?.()
    cancelScheduled = null
    if (buffered.length === 0) return
    const events = buffered
    buffered = []
    dispatch({ kind: "events", events })
  }

  return {
    push(action) {
      // Streamed reply deltas and partial tool output arrive far above display rate
      if (action.kind === "event" && (action.event.type === "message_update" || action.event.type === "tool_execution_update")) {
        buffered.push(action.event)
        cancelScheduled ??= schedule(flush)
        return
      }
      flush()
      dispatch(action)
    },
    reset() {
      cancelScheduled?.()
      cancelScheduled = null
      buffered = []
    },
  }
}
