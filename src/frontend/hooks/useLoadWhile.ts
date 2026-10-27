"use client"

import { useEffect, useEffectEvent } from "react"

/**
 * Run `load` each time `active` turns true (or `load` changes while active) and hand its result to `onValue`, or its failure to `onError`.
 *
 * - A result arriving after `active` turned false, `load` changed or the component unmounted is dropped
 * - Pass a stable `load` (a module function or a `useCallback` over its inputs); `onValue` and `onError` may be inline
 * - Without `onError` a failure is ignored
 */
export function useLoadWhile<T>(active: boolean, load: () => Promise<T>, onValue: (value: T) => void, onError?: (cause: unknown) => void): void {
  const deliver = useEffectEvent(onValue)
  const fail = useEffectEvent((cause: unknown) => onError?.(cause))

  useEffect(() => {
    if (!active) return
    let cancelled = false
    load().then(
      (value) => { if (!cancelled) deliver(value) },
      (cause: unknown) => { if (!cancelled) fail(cause) },
    )
    return () => { cancelled = true }
  }, [active, load])
}
