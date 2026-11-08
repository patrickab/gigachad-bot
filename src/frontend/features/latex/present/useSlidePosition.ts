"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"

/** `page`: a window moved to that slide. `ask`: a window joined and wants the current slide. */
type PositionMessage = { type: "page"; page: number } | { type: "ask" }

/** `page` within 1..pageCount, 1 while the page count is unknown. */
function clampPage(page: number, pageCount: number): number {
  return Math.min(Math.max(1, page), Math.max(1, pageCount))
}

/** The shown slide (1-based, clamped to `pageCount`) and `go` to move to one, shared both ways with every window on `channelName`. A joining window takes the others' slide. */
export function useSlidePosition(channelName: string, pageCount: number) {
  const [requested, setRequested] = useState(1)
  const requestedRef = useRef(requested)
  const channelRef = useRef<BroadcastChannel | null>(null)

  useLayoutEffect(() => {
    requestedRef.current = requested
  })

  useEffect(() => {
    const channel = new BroadcastChannel(channelName)
    channelRef.current = channel
    channel.onmessage = ({ data }: MessageEvent<PositionMessage>) => {
      if (data.type === "page") setRequested(data.page)
      else channel.postMessage({ type: "page", page: requestedRef.current } satisfies PositionMessage)
    }
    channel.postMessage({ type: "ask" } satisfies PositionMessage)
    return () => {
      channel.close()
      channelRef.current = null
    }
  }, [channelName])

  const go = useCallback((target: number) => {
    const page = clampPage(target, pageCount)
    setRequested(page)
    channelRef.current?.postMessage({ type: "page", page } satisfies PositionMessage)
  }, [pageCount])

  return { page: clampPage(requested, pageCount), go }
}
