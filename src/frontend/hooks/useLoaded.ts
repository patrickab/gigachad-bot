"use client"

import { useCallback, useState } from "react"
import { useLoadWhile } from "./useLoadWhile"

/** What `load(key)` resolved for the current `key` and `load`. `value` is undefined while it loads, after it failed and for a null key.
 *  `error` is the failure of the current read, else null. A result for an earlier key or load is dropped. Pass a stable `load`. */
export function useLoaded<K, T>(key: K | null, load: (key: K) => Promise<T>): { value: T | undefined; error: unknown } {
  const [held, setHeld] = useState<{ key: K | null; load: (key: K) => Promise<T>; value?: T; error: unknown } | null>(null)
  // Each read carries its own key and load, so an answer delivered just after they changed never passes for the next read's.
  const bound = useCallback(() => load(key as K).then((value) => ({ key, load, value, error: null }), (error: unknown) => ({ key, load, error })), [key, load])
  useLoadWhile(key !== null, bound, setHeld)
  return held !== null && Object.is(held.key, key) && held.load === load ? { value: held.value, error: held.error } : { value: undefined, error: null }
}
