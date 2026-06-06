"use client"

import { useCallback, useState } from "react"

export function useSafeAction() {
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(async (label: string, fn: () => Promise<unknown>) => {
    setError(null)
    try {
      await fn()
    } catch (e) {
      setError((e as Error).message || `Failed to ${label}`)
    }
  }, [])

  return { run, error, setError }
}
