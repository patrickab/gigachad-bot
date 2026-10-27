"use client"

import { useEffect, useEffectEvent } from "react"

/**
 * Autosave `value` once it has been quiet for `delayMs`, and once more on unmount.
 *
 * - `save(value, force)` runs with `force` false for the autosave and the unmount flush; the debounce cleanup alone would drop the last edits
 * - Returns `flush(force)` for explicit saves such as a shortcut
 */
export function useDebouncedSave<T>(value: T | null, save: (value: T, force: boolean) => void, delayMs: number): (force?: boolean) => void {
  const onSave = useEffectEvent(save)

  useEffect(() => {
    if (value === null) return
    const timer = setTimeout(() => onSave(value, false), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])

  const flush = (force = false) => {
    if (value !== null) save(value, force)
  }
  const flushOnUnmount = useEffectEvent(() => flush())
  useEffect(() => () => flushOnUnmount(), [])

  return flush
}
