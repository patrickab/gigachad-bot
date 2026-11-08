"use client"

import { useEffect, useRef, useState } from "react"
import { openPresentWindow, type PresentParams } from "./presentParams"

// How often an open audience window is checked for having been closed
const CLOSED_POLL_MS = 1000

export type AudienceStatus = "closed" | "open" | "blocked"

export interface AudienceWindow {
  status: AudienceStatus
  /** Opens the audience window, or focuses it when it is already open. Runs from a click or key handler, browsers block popups otherwise. */
  open: () => void
}

/** The audience window the presenter opens. Its status falls back to closed once the audience closes it. */
export function useAudienceWindow(params: PresentParams): AudienceWindow {
  const windowRef = useRef<Window | null>(null)
  const [status, setStatus] = useState<AudienceStatus>("closed")

  useEffect(() => {
    if (status !== "open") return
    const timer = setInterval(() => {
      if (windowRef.current?.closed) setStatus("closed")
    }, CLOSED_POLL_MS)
    return () => clearInterval(timer)
  }, [status])

  const open = () => {
    if (windowRef.current && !windowRef.current.closed) {
      windowRef.current.focus()
      return
    }
    windowRef.current = openPresentWindow(params)
    setStatus(windowRef.current ? "open" : "blocked")
  }

  return { status, open }
}
