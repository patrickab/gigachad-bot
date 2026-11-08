"use client"

import { useCallback, useRef } from "react"

/**
 * Only the latest of a series of requests may act: `run(request, onResult, onFailure?)` drops the answer of a request a later `run` or `cancel()` overtook.
 *
 * - `onResult` gets the answer, `onFailure` the failure of the latest request, neither runs for an overtaken one
 * - `run` resolves whether `onResult` ran. Without `onFailure` it rejects with the request's failure, overtaken or not
 */
export function useLatestRequest() {
  const latest = useRef(0)

  const run = useCallback(async <T>(request: Promise<T>, onResult: (value: T) => void, onFailure?: (cause: unknown) => void): Promise<boolean> => {
    const id = ++latest.current
    let value: T
    try {
      value = await request
    } catch (cause) {
      if (!onFailure) throw cause
      if (id === latest.current) onFailure(cause)
      return false
    }
    if (id !== latest.current) return false
    onResult(value)
    return true
  }, [])

  const cancel = useCallback(() => { latest.current++ }, [])

  return { run, cancel }
}
