"use client"

import { useEffect, useState } from "react"
import { errorMessage } from "@/lib/utils"

/**
 * Hits of `query` (used as given), asked once typing pauses `delayMs`; a blank query asks nothing.
 * `hits` is null until the answer for this exact query is in, a superseded request is aborted. `search` must be stable.
 */
export function useDebouncedSearch<T>(query: string, search: (query: string, signal: AbortSignal) => Promise<T[]>, delayMs = 200): { hits: T[] | null; error: string } {
  const [found, setFound] = useState<{ query: string; hits: T[]; error: string } | null>(null)
  useEffect(() => {
    if (!query) return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      search(query, controller.signal).then(
        (hits) => { if (!controller.signal.aborted) setFound({ query, hits, error: "" }) },
        (cause: unknown) => { if (!controller.signal.aborted) setFound({ query, hits: [], error: errorMessage(cause, "Search failed") }) },
      )
    }, delayMs)
    return () => { clearTimeout(timer); controller.abort() }
  }, [query, search, delayMs])
  return found?.query === query ? { hits: found.hits, error: found.error } : { hits: null, error: "" }
}
