"use client"

import { useCallback, useEffect, useEffectEvent, useRef } from "react"

/**
 * Run the latest scheduled action once `delayMs` passed without another `schedule`, for sources that change outside React state.
 *
 * - With `maxWaitMs`, a source that never goes quiet still runs once that long after its first unrun `schedule`
 * - `flush()` runs a pending action now, `cancel()` drops it, both do nothing when none is pending
 * - Nothing is dropped on unmount, the owner cancels or flushes in its own cleanup (a cleanup here would run first and drop what the owner flushes)
 */
export function useDebouncedCall(delayMs: number, maxWaitMs?: number): { schedule: (action: () => void) => void; flush: () => void; cancel: () => void } {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pending = useRef<(() => void) | null>(null)
  const firstAt = useRef(0)

  const cancel = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current)
    timer.current = null
    pending.current = null
  }, [])
  const flush = useCallback(() => {
    const action = pending.current
    cancel()
    action?.()
  }, [cancel])
  const schedule = useCallback((action: () => void) => {
    const now = Date.now()
    if (timer.current !== null) clearTimeout(timer.current)
    else firstAt.current = now
    pending.current = action
    const untilMax = maxWaitMs === undefined ? delayMs : Math.max(0, firstAt.current + maxWaitMs - now)
    timer.current = setTimeout(flush, Math.min(delayMs, untilMax))
  }, [flush, delayMs, maxWaitMs])

  return { schedule, flush, cancel }
}

/**
 * Autosave `value` once it has been quiet for `delayMs`, and once more on unmount.
 *
 * - `save(value, force)` runs with `force` false for the autosave and the unmount flush; the debounce cleanup alone would drop the last edits
 * - Returns `flush(force)` for explicit saves such as a shortcut
 */
export function useDebouncedSave<T>(value: T | null, save: (value: T, force: boolean) => void, delayMs: number): (force?: boolean) => void {
  const onSave = useEffectEvent(save)
  const { schedule, cancel } = useDebouncedCall(delayMs)

  useEffect(() => {
    if (value === null) return
    schedule(() => onSave(value, false))
    return cancel
  }, [value, schedule, cancel])

  const flush = (force = false) => {
    if (value !== null) save(value, force)
  }
  const flushOnUnmount = useEffectEvent(() => flush())
  useEffect(() => () => flushOnUnmount(), [])

  return flush
}
