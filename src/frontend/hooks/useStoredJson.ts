"use client"

import { useCallback, useMemo } from "react"
import { useStoredString } from "@/hooks/useStoredString"
import { readStorage } from "@/lib/storage"

/** `stored` decoded as JSON, undefined when missing or malformed. */
function decode(stored: string | null): unknown {
  if (stored === null) return undefined
  try {
    return JSON.parse(stored)
  } catch {
    return undefined
  }
}

/**
 * JSON value under `key` in `localStorage`, kept in sync across components and tabs.
 *
 * - `parse` validates the decoded JSON (undefined when missing or malformed) into a `T`; pass a stable function
 * - `update` re-reads storage first, so a write from another tab since the last render is not lost
 * - `T` must survive `JSON.stringify`
 */
export function useStoredJson<T>(key: string, parse: (raw: unknown) => T): [T, (update: (latest: T) => T) => void] {
  const [stored, store] = useStoredString(key)
  const value = useMemo(() => parse(decode(stored)), [stored, parse])
  const update = useCallback((change: (latest: T) => T) => {
    store(JSON.stringify(change(parse(decode(readStorage(key))))))
  }, [key, parse, store])
  return [value, update]
}
