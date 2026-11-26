"use client"

import { useEffect, useState } from "react"

const TICK_MS = 60_000

/** Current time in ms, renewed every minute so rows age without a refetch. */
export function useNowMs(): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), TICK_MS)
    return () => clearInterval(timer)
  }, [])
  return now
}
