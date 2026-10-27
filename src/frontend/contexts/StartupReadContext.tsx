"use client"

import { createContext, useCallback, useContext, useEffect, useEffectEvent, useId, useMemo, useRef, useState, type ReactNode } from "react"
import { errorMessage } from "@/lib/utils"

const STARTUP_TIMEOUT_MS = 15_000

type ReadFailure = { label: string; error: string; loading: boolean; retry: () => void }
type ReportFailure = (id: string, failure: ReadFailure | null) => void
const StartupReadContext = createContext<ReportFailure>(() => {})

/** Surface failed startup reads without replacing or remounting the workspace. */
export function StartupReadProvider({ children }: { children: ReactNode }) {
  const [failures, setFailures] = useState<Record<string, ReadFailure>>({})
  const report = useCallback<ReportFailure>((id, failure) => {
    setFailures((current) => {
      if (!failure && !(id in current)) return current
      const next = { ...current }
      if (failure) next[id] = failure
      else delete next[id]
      return next
    })
  }, [])
  const entries = Object.values(failures)
  const labels = [...new Set(entries.map((entry) => entry.label))].join(", ")
  const retrying = entries.some((entry) => entry.loading)

  return <StartupReadContext.Provider value={report}>
    {children}
    {entries.length > 0 && <section aria-label="Startup connection problems" className="fixed inset-x-4 top-14 z-[100] mx-auto flex max-w-2xl items-start gap-3 rounded-lg border border-divider bg-surface px-4 py-3 text-xs text-ink shadow-[var(--shadow-lg)]">
      <div role="status" className="min-w-0 flex-1 space-y-1">
        <p>{retrying ? "Retrying failed startup reads…" : "Some startup data could not be loaded."}</p>
        <p className="text-ink-muted">{labels}</p>
        {!retrying && <p className="text-ink-subtle">{[...new Set(entries.map((entry) => entry.error))].join(" ")}</p>}
      </div>
      <button type="button" disabled={retrying} onClick={() => entries.forEach((entry) => { if (!entry.loading) entry.retry() })}
        className="shrink-0 rounded px-2 py-1 text-ink-muted hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted disabled:opacity-50">
        {retrying ? "Retrying…" : "Retry"}
      </button>
    </section>}
  </StartupReadContext.Provider>
}

/** Bound one startup read and retry it without resetting its consumer's state. Keep `load` stable. */
export function useStartupRead<T>(label: string, load: (signal: AbortSignal) => Promise<T>, apply: (value: T) => void, enabled = true) {
  const id = useId()
  const report = useContext(StartupReadContext)
  const [state, setState] = useState<{ loading: boolean; error: string | null }>({ loading: enabled, error: null })
  const retryRef = useRef<() => void>(() => {})
  const retry = useCallback(() => retryRef.current(), [])
  const applyValue = useEffectEvent(apply)

  useEffect(() => {
    if (!enabled) return
    let disposed = false
    let controller: AbortController | null = null
    let failed = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const run = async () => {
      if (controller) return
      const attempt = new AbortController()
      controller = attempt
      setState({ loading: true, error: null })
      if (failed) report(id, { label, error: "", loading: true, retry })
      try {
        const timeout = new Promise<never>((_, reject) => {
          timer = setTimeout(() => {
            reject(new Error("The backend did not respond within 15 seconds."))
            attempt.abort()
          }, STARTUP_TIMEOUT_MS)
        })
        const value = await Promise.race([load(attempt.signal), timeout])
        if (disposed) return
        applyValue(value)
        failed = false
        setState({ loading: false, error: null })
        report(id, null)
      } catch (cause) {
        if (disposed) return
        attempt.abort()
        failed = true
        const error = errorMessage(cause, "The backend request failed.")
        setState({ loading: false, error })
        report(id, { label, error, loading: false, retry })
      } finally {
        clearTimeout(timer)
        controller = null
      }
    }

    retryRef.current = () => { void run() }
    void run()
    return () => {
      disposed = true
      clearTimeout(timer)
      controller?.abort()
      retryRef.current = () => {}
      report(id, null)
    }
  }, [enabled, id, label, load, report, retry])

  return useMemo(() => ({ ...state, loading: enabled && state.loading, retry }), [enabled, state, retry])
}
