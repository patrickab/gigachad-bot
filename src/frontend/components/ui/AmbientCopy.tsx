"use client"

import { useEffect, useRef, useSyncExternalStore } from "react"
import { getAmbientCanvas, subscribeAmbientCanvas } from "@/lib/ambientCanvas"

const NO_CANVAS = () => null

/**
 * The app background's picture as a window's own background, lined up with the background around the window.
 *
 * - A video of the background canvas's stream: frames already drawn, so a video decode, not a second graphics context
 * - Window-sized and shifted by the window's `left` and `top` (plus its 1px border), so it shows what lies behind the window
 * - Shows nothing, the window's own color, while the background has no canvas or the browser cannot stream one
 */
export function AmbientCopy({ left, top }: { left: number; top: number }) {
  const source = useSyncExternalStore(subscribeAmbientCanvas, getAmbientCanvas, NO_CANVAS)
  const video = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const element = video.current
    if (!element || !source || typeof source.captureStream !== "function") return
    const stream = source.captureStream()
    element.srcObject = stream
    void element.play().catch(() => {})
    return () => {
      element.srcObject = null
      stream.getTracks().forEach((track) => track.stop())
    }
  }, [source])
  return <video ref={video} aria-hidden muted playsInline className="pointer-events-none absolute max-w-none object-fill"
    style={{ left: -left - 1, top: -top - 1, width: window.innerWidth, height: window.innerHeight }} />
}
