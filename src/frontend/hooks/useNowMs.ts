"use client"

import { useEffect, useState } from "react"

/** Current time in ms, renewed every `intervalMs` while `active` (by default every minute, so rows age without a refetch). */
export function useNowMs(intervalMs = 60_000, active = true): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(timer)
  }, [intervalMs, active])
  return now
}
