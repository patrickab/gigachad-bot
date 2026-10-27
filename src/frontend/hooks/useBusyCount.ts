"use client"

import { useCallback, useState } from "react"

/** Count of in-flight `track`ed tasks: `busy` while at least one runs, whether it resolves or throws. */
export function useBusyCount(): { busy: boolean; track: <T>(work: () => Promise<T>) => Promise<T> } {
  const [count, setCount] = useState(0)
  const track = useCallback(async <T,>(work: () => Promise<T>): Promise<T> => {
    setCount((current) => current + 1)
    try {
      return await work()
    } finally {
      setCount((current) => current - 1)
    }
  }, [])
  return { busy: count > 0, track }
}
