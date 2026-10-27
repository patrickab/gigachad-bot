"use client"

import { useEffect, useEffectEvent, useLayoutEffect, useRef, type RefObject } from "react"
import type { Point } from "@/lib/domain/edgeRouting"
import { reframeCamera, type ViewportSize } from "@/lib/domain/viewport"

type Camera = { offset: Point; scale: number }

interface ReframeOptions {
  /** Effective zoom of the host embedding this viewport (1 at the top level). */
  hostScale: number
  /** Whether the camera can be read and written; the frame size is recorded from the moment it turns true. */
  ready: boolean
  /** Current camera, null while there is none to carry over. */
  read: () => Camera | null
  /** Apply the reframed camera; `size` is the frame size it was computed for. */
  write: (offset: Point, scale: number, size: ViewportSize) => void
  /** Leave the camera alone when the previous frame size was empty (a hidden or unmeasured frame). */
  skipFromEmpty?: boolean
}

/**
 * Keep the world point at the center of a top-left pinned viewport centered while its frame resizes, and magnify it with the host's zoom.
 *
 * - Both re-project from the frame size the current camera was set for, which this hook records
 * - Host zoom runs in a layout effect, before the `ResizeObserver` sees the frame its host already resized, so the resize is not applied twice
 */
export function useViewportReframe(ref: RefObject<HTMLElement | null>, { hostScale, ready, read, write, skipFromEmpty = false }: ReframeOptions): void {
  const sizeRef = useRef<ViewportSize>({ width: 0, height: 0 })
  const hostScaleRef = useRef(hostScale)

  const reframe = useEffectEvent((factor: number) => {
    const element = ref.current
    if (!element || !ready) return
    const next = { width: element.clientWidth, height: element.clientHeight }
    const previous = sizeRef.current
    sizeRef.current = next
    const camera = read()
    if (!camera || (skipFromEmpty && (previous.width === 0 || previous.height === 0))) return
    const view = reframeCamera(camera.offset, camera.scale, previous, next, factor)
    write(view.offset, view.scale, next)
  })

  useEffect(() => {
    const element = ref.current
    if (!element || !ready || typeof ResizeObserver === "undefined") return
    sizeRef.current = { width: element.clientWidth, height: element.clientHeight }
    const observer = new ResizeObserver(() => {
      const { width, height } = sizeRef.current
      if (element.clientWidth !== width || element.clientHeight !== height) reframe(1)
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref, ready])

  useLayoutEffect(() => {
    const factor = hostScale / hostScaleRef.current
    hostScaleRef.current = hostScale
    if (factor !== 1) reframe(factor)
  }, [hostScale])
}
