"use client"

import { useCallback, useState } from "react"
import { errorMessage } from "@/lib/utils"

/** Run `action` with pending and error state; `run` resolves false when it threw, `error` is "" when none. A function `fallback` builds the message from the call's arguments. */
export function useAsyncAction<A extends unknown[]>(action: (...args: A) => Promise<unknown>, fallback: string | ((...args: A) => string)): {
  run: (...args: A) => Promise<boolean>
  pending: boolean
  error: string
  setError: (message: string) => void
} {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState("")

  const run = useCallback(async (...args: A) => {
    setError("")
    setPending(true)
    try {
      await action(...args)
      return true
    } catch (cause) {
      setError(errorMessage(cause, typeof fallback === "string" ? fallback : fallback(...args)))
      return false
    } finally {
      setPending(false)
    }
  }, [action, fallback])

  return { run, pending, error, setError }
}
