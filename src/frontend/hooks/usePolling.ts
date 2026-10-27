"use client"

import { useEffect, useEffectEvent, useSyncExternalStore } from "react"

interface PollingOptions {
  /** Milliseconds between runs; null stops polling. */
  intervalMs: number | null
  /** Polls only while true. */
  active?: boolean
  /** Also run when the window regains focus. */
  refreshOnFocus?: boolean
  /** Keep polling while the page is hidden; otherwise polling pauses and resumes with a run once the page shows again. */
  runWhileHidden?: boolean
  /** A change restarts polling with an immediate run. */
  restartKey?: string | number
}

function subscribeVisibility(onChange: () => void): () => void {
  document.addEventListener("visibilitychange", onChange)
  return () => document.removeEventListener("visibilitychange", onChange)
}

const pageVisible = () => document.visibilityState === "visible"

/**
 * Run `load` at once and then every `intervalMs` while polling, never twice at the same time.
 *
 * - Each start (activation, `intervalMs` or `restartKey` change, the page showing again) begins with a run and a fresh in-flight guard
 * - `load` gets a signal that aborts when that polling run stops, so a late result can be dropped
 * - The latest `load` is always called, so it may be an inline closure
 */
export function usePolling(load: (signal: AbortSignal) => Promise<unknown> | void, { intervalMs, active = true, refreshOnFocus = false, runWhileHidden = false, restartKey }: PollingOptions): void {
  const visible = useSyncExternalStore(subscribeVisibility, pageVisible, () => true)
  const onLoad = useEffectEvent(load)
  const polling = active && (runWhileHidden || visible)

  useEffect(() => {
    if (!polling || intervalMs === null) return
    const controller = new AbortController()
    let inFlight = false
    const run = async () => {
      if (inFlight || (!runWhileHidden && document.hidden)) return
      inFlight = true
      try {
        await onLoad(controller.signal)
      } finally {
        inFlight = false
      }
    }
    void run()
    const timer = window.setInterval(() => void run(), intervalMs)
    const onFocus = () => void run()
    if (refreshOnFocus) window.addEventListener("focus", onFocus)
    return () => {
      controller.abort()
      window.clearInterval(timer)
      window.removeEventListener("focus", onFocus)
    }
  }, [polling, intervalMs, refreshOnFocus, runWhileHidden, restartKey])
}
