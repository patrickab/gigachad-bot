"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type RefObject } from "react"
import { useTwoFingerGesture } from "@/hooks/useTwoFingerGesture"
import { inOwnAttachment } from "@/features/editor/canvas/attachments/canvasTransfer"
import { cameraToOffset, offsetToCamera, reframeCamera, zoomCamera, type ViewportPoint, type ZoomAnchor } from "@/lib/domain/viewport"

const MIN_SCALE = 0.15
const MAX_SCALE = 3
const ZOOM_SETTLE_MS = 250 // attachments re-rasterize at full resolution this long after the zoom stops changing
const WHEEL_ZOOM_STEP = 1.08

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
 */
export function useCanvasCamera(containerRef: RefObject<HTMLDivElement | null>, { viewport, hostScale, depth }: CanvasCameraOptions) {
  const zoomAnchor: ZoomAnchor = depth > 0 ? "viewport-center" : "cursor"
  const [scale, setScale] = useState((viewport?.scale ?? 0.5) * hostScale)
  // offset is derived once the container mounts; fall back to a reasonable default
  const [offset, setOffset] = useState({ x: 40, y: 40 })
  const scaleRef = useRef(scale)
  const offsetRef = useRef(offset)
  scaleRef.current = scale
  offsetRef.current = offset
  const hostScaleRef = useRef(hostScale)
  // Frame size the current offset was computed for; resizes re-project from it.
  const viewportSizeRef = useRef({ width: 0, height: 0 })
  const restoredRef = useRef(false)

  const setCamera = useCallback((nextOffset: ViewportPoint, nextScale = scaleRef.current) => {
    scaleRef.current = nextScale
    offsetRef.current = nextOffset
    setScale(nextScale)
    setOffset(nextOffset)
  }, [])

  // The canvas's own zoom is clamped; its host's zoom rides on top of it.
  const clampScale = (next: number) => Math.min(MAX_SCALE * hostScaleRef.current, Math.max(MIN_SCALE * hostScaleRef.current, next))

  // Restore a legacy viewport once on mount, then keep the view centered across resizes.
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const camera = { scale: (viewport?.scale ?? 0.5) * hostScaleRef.current, centerX: viewport?.centerX ?? 0, centerY: viewport?.centerY ?? 0 }
    setCamera(cameraToOffset(camera, { width: el.clientWidth, height: el.clientHeight }), camera.scale)
    restoredRef.current = true
    viewportSizeRef.current = { width: el.clientWidth, height: el.clientHeight }
    const ro = new ResizeObserver(() => {
      const next = { width: el.clientWidth, height: el.clientHeight }
      const previous = viewportSizeRef.current
      if (!restoredRef.current || (next.width === previous.width && next.height === previous.height)) return
      // Keep the world point at the viewport's center fixed across resizes: the
      // browser window for a top-level canvas, fullscreen or a resize drag for a
      // nested one. Pinning the top-left instead would slide the middle away.
      const view = reframeCamera(offsetRef.current, scaleRef.current, previous, next)
      viewportSizeRef.current = next
      setCamera(view.offset)
    })
    ro.observe(el)
    return () => ro.disconnect()
  // Camera state is local after initialization.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Host zoom: the frame has already been resized by the same factor, so scale
  // the view with it around the center. Runs before the ResizeObserver sees the
  // new size and records it, so the resize isn't applied a second time.
  useLayoutEffect(() => {
    const factor = hostScale / hostScaleRef.current
    hostScaleRef.current = hostScale
    const el = containerRef.current
    if (factor === 1 || !el || !restoredRef.current) return
    const next = { width: el.clientWidth, height: el.clientHeight }
    const view = reframeCamera(offsetRef.current, scaleRef.current, viewportSizeRef.current, next, factor)
    viewportSizeRef.current = next
    setCamera(view.offset, view.scale)
  }, [hostScale, containerRef, setCamera])

  // Wheel zoom
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (inOwnAttachment(e.target, el)) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const size = { width: el.clientWidth, height: el.clientHeight }
      const current = offsetToCamera(offsetRef.current, scaleRef.current, size)
      const nextScale = clampScale(current.scale * (e.deltaY < 0 ? WHEEL_ZOOM_STEP : 1 / WHEEL_ZOOM_STEP))
      const camera = zoomCamera(current, nextScale / current.scale, size, zoomAnchor, { x: e.clientX - rect.left, y: e.clientY - rect.top })
      setCamera(cameraToOffset(camera, size), camera.scale)
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  // clampScale only reads a ref
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoomAnchor, containerRef, setCamera])

  // Two-finger touch: pinch-zoom + pan. Fingers that land in a nested window belong to
  // that window's pinch/pan, not ours.
  useTwoFingerGesture(containerRef, {
    ignoreTarget: (target) => !!containerRef.current && inOwnAttachment(target, containerRef.current),
    begin: () => ({ scale: scaleRef.current, ox: offsetRef.current.x, oy: offsetRef.current.y }),
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

  const startPan = useCallback((e: ReactPointerEvent) => {
    e.preventDefault()
    setIsPanning(true)
    panStart.current = { x: e.clientX, y: e.clientY, ox: offsetRef.current.x, oy: offsetRef.current.y }
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }, [])

  const movePan = useCallback((e: ReactPointerEvent) => {
    if (!isPanning) return
    setCamera({ x: panStart.current.ox + e.clientX - panStart.current.x, y: panStart.current.oy + e.clientY - panStart.current.y })
  }, [isPanning, setCamera])

  const endPan = useCallback(() => setIsPanning(false), [])

  return { scale, offset, settledScale, scaleRef, offsetRef, screenToCanvas, pointAtCenter, isPanning, startPan, movePan, endPan }
}
