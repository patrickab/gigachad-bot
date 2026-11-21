"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react"
import { useTwoFingerGesture } from "@/hooks/useTwoFingerGesture"
import { useViewportReframe } from "@/hooks/useViewportReframe"
import { inOwnAttachment } from "@/features/editor/canvas/attachments/canvasTransfer"
import { cameraToOffset, offsetToCamera, zoomCamera, type ZoomAnchor } from "@/lib/domain/viewport"
import type { Point } from "@/lib/domain/edgeRouting"

const MIN_SCALE = 0.15
const MAX_SCALE = 3
const ZOOM_SETTLE_MS = 250 // attachments re-rasterize at full resolution this long after the zoom stops changing
const WHEEL_ZOOM_STEP = 1.08
const ZOOM_GLIDE_MS = 60 // time constant of the glide to a wheel zoom's target
const COAST_MS = 260 // time constant of a released pan slowing down
const COAST_MIN_SPEED = 0.03 // px per ms, a slower pan stops
const COAST_FLICK_MS = 80 // a pan released later than this after its last move was held still: no coasting
const FRAME_MS = 1000 / 60

const prefersReducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false

interface CanvasCameraOptions {
  /** Legacy saved viewport, restored once on mount; camera movement stays device-local afterwards. */
  viewport?: { scale: number; centerX: number; centerY: number }
  /** Effective zoom of the canvas embedding this one (1 at the top level). */
  hostScale: number
  /** Nesting depth: nested canvases zoom around their center instead of the cursor. */
  depth: number
}

/**
 * Pan/zoom state of one canvas. `scale` is the effective on-screen scale: the canvas's own zoom
 * times its host's. Every camera write goes through one setter that keeps refs and state in step.
 *
 * - A wheel zoom glides to its target and a flicked pan coasts to a stop, both instant with reduced motion
 * - Any new gesture takes over from a glide or coast still running
 */
export function useCanvasCamera(containerRef: RefObject<HTMLDivElement | null>, { viewport, hostScale, depth }: CanvasCameraOptions) {
  const zoomAnchor: ZoomAnchor = depth > 0 ? "viewport-center" : "cursor"
  const [scale, setScale] = useState((viewport?.scale ?? 0.5) * hostScale)
  // offset is derived once the container mounts; fall back to a reasonable default
  const [offset, setOffset] = useState({ x: 40, y: 40 })
  const scaleRef = useRef(scale)
  const offsetRef = useRef(offset)
  const hostScaleRef = useRef(hostScale)
  useLayoutEffect(() => { hostScaleRef.current = hostScale }, [hostScale])

  const setCamera = useCallback((nextOffset: Point, nextScale = scaleRef.current) => {
    scaleRef.current = nextScale
    offsetRef.current = nextOffset
    setScale(nextScale)
    setOffset(nextOffset)
  }, [])

  // The canvas's own zoom is clamped; its host's zoom rides on top of it.
  const clampScale = (next: number) => Math.min(MAX_SCALE * hostScaleRef.current, Math.max(MIN_SCALE * hostScaleRef.current, next))

  // Restore a legacy viewport once on mount.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const camera = { scale: (viewport?.scale ?? 0.5) * hostScaleRef.current, centerX: viewport?.centerX ?? 0, centerY: viewport?.centerY ?? 0 }
    setCamera(cameraToOffset(camera, { width: el.clientWidth, height: el.clientHeight }), camera.scale)
  // Camera state is local after initialization.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Then keep the view centered across resizes (the browser window for a top-level canvas,
  // fullscreen or a resize drag for a nested one) and magnified with the host's zoom.
  // Called after the restore effect, so it records the frame the restore used.
  useViewportReframe(containerRef, {
    hostScale,
    ready: true,
    read: () => ({ offset: offsetRef.current, scale: scaleRef.current }),
    write: (nextOffset, nextScale) => setCamera(nextOffset, nextScale),
  })

  // Camera motion that outlives its gesture: a wheel zoom gliding to `zoom.target` around the point `zoom.at`, a released pan coasting at `coast` px per ms
  const motion = useRef<{ frame: number; last: number; zoom: { target: number; at: Point } | null; coast: Point | null }>({ frame: 0, last: 0, zoom: null, coast: null })

  const stopMotion = useCallback(() => {
    cancelAnimationFrame(motion.current.frame)
    motion.current = { frame: 0, last: 0, zoom: null, coast: null }
  }, [])
  useEffect(() => stopMotion, [stopMotion])

  const stepMotion = useCallback((elapsedMs: number) => {
    const el = containerRef.current
    const state = motion.current
    if (!el) return
    if (state.zoom) {
      const size = { width: el.clientWidth, height: el.clientHeight }
      const current = offsetToCamera(offsetRef.current, scaleRef.current, size)
      const remaining = state.zoom.target / current.scale
      const factor = Math.abs(Math.log(remaining)) < 0.004 ? remaining : remaining ** (1 - Math.exp(-elapsedMs / ZOOM_GLIDE_MS))
      const camera = zoomCamera(current, factor, size, zoomAnchor, state.zoom.at)
      setCamera(cameraToOffset(camera, size), camera.scale)
      if (factor === remaining) state.zoom = null
    }
    if (state.coast) {
      setCamera({ x: offsetRef.current.x + state.coast.x * elapsedMs, y: offsetRef.current.y + state.coast.y * elapsedMs })
      const slowed = Math.exp(-elapsedMs / COAST_MS)
      state.coast = { x: state.coast.x * slowed, y: state.coast.y * slowed }
      if (Math.hypot(state.coast.x, state.coast.y) < COAST_MIN_SPEED) state.coast = null
    }
  }, [containerRef, zoomAnchor, setCamera])

  const runMotion = useCallback(() => {
    const state = motion.current
    if (state.frame) return
    const frame = () => {
      const now = performance.now()
      stepMotion(Math.min(Math.max(now - state.last, 1), 3 * FRAME_MS))
      state.last = now
      state.frame = state.zoom || state.coast ? requestAnimationFrame(frame) : 0
    }
    state.last = performance.now()
    state.frame = requestAnimationFrame(frame)
  }, [stepMotion])

  // Wheel zoom
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (inOwnAttachment(e.target, el)) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const step = e.deltaY < 0 ? WHEEL_ZOOM_STEP : 1 / WHEEL_ZOOM_STEP
      // Steps of one spin add up on the target, the camera glides after it
      motion.current.coast = null
      motion.current.zoom = { target: clampScale((motion.current.zoom?.target ?? scaleRef.current) * step), at: { x: e.clientX - rect.left, y: e.clientY - rect.top } }
      if (prefersReducedMotion()) {
        stepMotion(Infinity)
        return
      }
      // The first step lands with the event, so the zoom answers at once
      if (!motion.current.frame) stepMotion(FRAME_MS)
      runMotion()
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [containerRef, stepMotion, runMotion])

  // Two-finger touch: pinch-zoom + pan. Fingers that land in a nested window belong to
  // that window's pinch/pan, not ours.
  useTwoFingerGesture(containerRef, {
    ignoreTarget: (target) => !!containerRef.current && inOwnAttachment(target, containerRef.current),
    begin: () => {
      stopMotion()
      return { scale: scaleRef.current, ox: offsetRef.current.x, oy: offsetRef.current.y }
    },
    update: (t, { ratio, cx, cy, mx, my }) => {
      const el = containerRef.current
      if (!el) return
      const size = { width: el.clientWidth, height: el.clientHeight }
      const nextScale = clampScale(t.scale * ratio)
      const current = offsetToCamera({ x: t.ox, y: t.oy }, t.scale, size)
      const camera = zoomCamera(current, nextScale / t.scale, size, zoomAnchor, { x: cx, y: cy })
      const zoomed = cameraToOffset(camera, size)
      setCamera({ x: zoomed.x + mx - cx, y: zoomed.y + my - cy }, camera.scale)
    },
  })

  // Attachments scale with canvas zoom, but re-rendering the PDF at every zoom
  // frame re-rasterizes all pages (flicker). Instead the gesture scales them as
  // bitmaps via CSS transform, and once the zoom settles the layout width snaps
  // to the true screen size for one sharp re-render — PdfViewer preserves the
  // reading position across that width change.
  const [settledScale, setSettledScale] = useState(scale)
  useEffect(() => {
    const t = setTimeout(() => setSettledScale(scale), ZOOM_SETTLE_MS)
    return () => clearTimeout(t)
  }, [scale])

  // Client px to canvas units, inverting the current pan/zoom.
  const screenToCanvas = useCallback((clientX: number, clientY: number): [number, number] => {
    const el = containerRef.current
    if (!el) return [0, 0]
    const rect = el.getBoundingClientRect()
    return [(clientX - rect.left - offsetRef.current.x) / scaleRef.current, (clientY - rect.top - offsetRef.current.y) / scaleRef.current]
  }, [containerRef])

  // Top-left canvas position that centers a w-wide box of the given height/width aspect in the view.
  const pointAtCenter = useCallback((w: number, aspect: number) => {
    const el = containerRef.current
    const center = offsetToCamera(offsetRef.current, scaleRef.current, { width: el ? el.clientWidth : 800, height: el ? el.clientHeight : 600 })
    return { cx: center.centerX - w / 2, cy: center.centerY - (w * aspect) / 2 }
  }, [containerRef])

  // --- Pan (middle mouse, right mouse or space+drag) ---
  const [isPanning, setIsPanning] = useState(false)
  const panStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 })
  // The pan's last pointer sample and its smoothed speed in px per ms, for the coast after release
  const panPace = useRef({ x: 0, y: 0, at: 0, speed: { x: 0, y: 0 } })

  const startPan = useCallback((e: ReactPointerEvent) => {
    e.preventDefault()
    stopMotion()
    setIsPanning(true)
    panStart.current = { x: e.clientX, y: e.clientY, ox: offsetRef.current.x, oy: offsetRef.current.y }
    panPace.current = { x: e.clientX, y: e.clientY, at: performance.now(), speed: { x: 0, y: 0 } }
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }, [stopMotion])

  const movePan = useCallback((e: ReactPointerEvent) => {
    if (!isPanning) return
    const pace = panPace.current
    const now = performance.now()
    const elapsed = Math.max(now - pace.at, 1)
    const speed = { x: 0.6 * (e.clientX - pace.x) / elapsed + 0.4 * pace.speed.x, y: 0.6 * (e.clientY - pace.y) / elapsed + 0.4 * pace.speed.y }
    panPace.current = { x: e.clientX, y: e.clientY, at: now, speed }
    setCamera({ x: panStart.current.ox + e.clientX - panStart.current.x, y: panStart.current.oy + e.clientY - panStart.current.y })
  }, [isPanning, setCamera])

  const endPan = useCallback(() => {
    setIsPanning((wasPanning) => {
      const { at, speed } = panPace.current
      if (wasPanning && performance.now() - at < COAST_FLICK_MS && Math.hypot(speed.x, speed.y) > COAST_MIN_SPEED && !prefersReducedMotion()) {
        motion.current.coast = speed
        runMotion()
      }
      return false
    })
  }, [runMotion])

  return { scale, offset, settledScale, scaleRef, offsetRef, screenToCanvas, pointAtCenter, isPanning, startPan, movePan, endPan }
}
