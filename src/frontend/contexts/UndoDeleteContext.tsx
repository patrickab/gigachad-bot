"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"

const UNDO_MS = 6000

type PendingDelete = { key: string, label: string, run: () => Promise<void>, onUndo?: () => void }
type UndoDeleteContextValue = {
  /**
   * Starts a deletion the user can undo for six seconds; `run` commits it when that window closes.
   * Only one deletion is undoable at a time: scheduling another commits the current one at once.
   * A caller either hides the row while `pending(key)` holds, or removes it optimistically and
   * passes `onUndo` to put it back. `onUndo` runs on Undo and when `run` rejects; a rejection's
   * message is shown in the notice.
   */
  schedule: (key: string, label: string, run: () => Promise<void>, onUndo?: () => void) => void
  /** Whether `key` is undoable or its commit is still in flight; hidden rows stay hidden until it settles. */
  pending: (key: string) => boolean
}

const UndoDeleteContext = createContext<UndoDeleteContextValue | null>(null)

/** Owns the single undoable deletion and renders its notice (Undo, progress, or the refusal). */
export function UndoDeleteProvider({ children }: { children: ReactNode }) {
  const [item, setItem] = useState<PendingDelete | null>(null)
  const [committing, setCommitting] = useState<Set<string>>(() => new Set())
  const [error, setError] = useState("")
  // The undoable deletion, read synchronously by `schedule` and `undo`; `item` is what the notice shows.
  const current = useRef<PendingDelete | null>(null)
  const timer = useRef<number | undefined>(undefined)

  const commit = useCallback((candidate: PendingDelete) => {
    if (current.current !== candidate) return
    current.current = null
    window.clearTimeout(timer.current)
    setCommitting((keys) => new Set(keys).add(candidate.key))
    void Promise.resolve().then(candidate.run).then(() => {
      setItem((active) => active === candidate ? null : active)
    }).catch((cause: unknown) => {
      candidate.onUndo?.()
      setItem((active) => active === candidate ? null : active)
      setError(cause instanceof Error ? cause.message : `Could not delete ${candidate.label}`)
    }).finally(() => setCommitting((keys) => {
      const remaining = new Set(keys)
      remaining.delete(candidate.key)
      return remaining
    }))
  }, [])

  const schedule = useCallback((key: string, label: string, run: () => Promise<void>, onUndo?: () => void) => {
    if (current.current) commit(current.current)
    const candidate = { key, label, run, onUndo }
    current.current = candidate
    setItem(candidate)
    setError("")
    timer.current = window.setTimeout(() => commit(candidate), UNDO_MS)
  }, [commit])

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const undo = () => {
    current.current?.onUndo?.()
    current.current = null
    window.clearTimeout(timer.current)
    setItem(null)
  }

  const pending = useCallback((key: string) => item?.key === key || committing.has(key), [item, committing])
  const value = useMemo(() => ({ schedule, pending }), [schedule, pending])
  const itemCommitting = item !== null && committing.has(item.key)

  return (
    <UndoDeleteContext.Provider value={value}>
      {children}
      {(item || error) && <div role="status" className="fixed bottom-5 left-1/2 z-[100] flex max-w-[90vw] -translate-x-1/2 items-center gap-3 rounded-lg border border-divider bg-surface-elevated px-3 py-2 text-sm text-ink shadow-lg">
        <span>{error || (itemCommitting ? `Deleting ${item.label}…` : `Deleted ${item?.label}`)}</span>
        {item && !itemCommitting ? <button type="button" onClick={undo} className="font-medium text-ink hover:underline">Undo</button>
          : error ? <button type="button" onClick={() => setError("")} aria-label="Dismiss deletion error" className="text-ink-muted hover:text-ink">×</button> : null}
      </div>}
    </UndoDeleteContext.Provider>
  )
}

/** The app's undoable-deletion API; must be rendered under `UndoDeleteProvider`. */
export function useUndoDelete(): UndoDeleteContextValue {
  const context = useContext(UndoDeleteContext)
  if (!context) throw new Error("useUndoDelete needs UndoDeleteProvider")
  return context
}
