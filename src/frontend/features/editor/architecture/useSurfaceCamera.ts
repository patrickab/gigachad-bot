import { useCallback, useEffect, useRef, type RefObject, type WheelEvent as ReactWheelEvent } from "react"
import type { NodeChange } from "@xyflow/react"
import type { GraphFlowInstance, GraphFlowNode } from "@/features/editor/architecture/architectureFlow"
import { useViewportReframe } from "@/hooks/useViewportReframe"
import { useTwoFingerGesture } from "@/hooks/useTwoFingerGesture"

/**
 * The surface's camera: fitting, wheel and pinch zoom, and keeping the view
 * steady while the frame resizes or the host canvas zooms.
 */
export function useSurfaceCamera({ flow, viewportRef, onNodesChange, autoFit, hostScale, readOnly }: {
  flow: GraphFlowInstance | null
  viewportRef: RefObject<HTMLDivElement | null>
  onNodesChange: (changes: NodeChange<GraphFlowNode>[]) => void
  autoFit: boolean
  hostScale: number
  readOnly: boolean
}) {
  const fitView = useCallback(() => flow?.fitView({ padding: 0.22, duration: 180 }), [flow])
  // Nodes measure and auto-grow after the first fit, which would leave the
  // graph cropped. Until the user takes the camera, keep refitting as sizes settle.
  const followFitRef = useRef(true)
  const followTimerRef = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(followTimerRef.current), [])
  const takeCamera = useCallback(() => { followFitRef.current = false }, [])
  const handleNodesChange = useCallback((changes: NodeChange<GraphFlowNode>[]) => {
    onNodesChange(changes)
    if (!followFitRef.current || !changes.some((change) => change.type === "dimensions")) return
    window.clearTimeout(followTimerRef.current)
    followTimerRef.current = window.setTimeout(() => { if (followFitRef.current) void flow?.fitView({ padding: 0.22 }) }, 120)
  }, [onNodesChange, flow])
  useEffect(() => {
    if (!autoFit) return
    followFitRef.current = true
    // Two frames: the host's new size has to be laid out and picked up by React
    // Flow's own size observer before a fit can use it.
    let frame = requestAnimationFrame(() => { frame = requestAnimationFrame(() => { void fitView() }) })
    return () => cancelAnimationFrame(frame)
    // Refit on the transition to true only, not whenever the flow instance changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoFit])
  const zoomGraphAtCenter = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    if (!flow || event.deltaY === 0) return
    takeCamera()
    event.preventDefault()
    event.stopPropagation()
    void (event.deltaY < 0 ? flow.zoomIn() : flow.zoomOut())
  }, [flow, takeCamera])
  // Two-finger pinch + pan: the same hook every canvas uses.
  useTwoFingerGesture(viewportRef, {
    enabled: !readOnly && !!flow,
    begin: () => { takeCamera(); return flow!.getViewport() },
    update: (start, { ratio, cx, cy, mx, my }) => {
      // Same range as the ReactFlow minZoom/maxZoom props.
      const zoom = Math.min(2 * hostScale, Math.max(0.2 * hostScale, start.zoom * ratio))
      flow!.setViewport({ x: mx - (cx - start.x) / start.zoom * zoom, y: my - (cy - start.y) / start.zoom * zoom, zoom })
    },
  })
  // React Flow pins its viewport to the top-left corner, so when the frame
  // resizes (fullscreen, a resize drag) or the host canvas zooms, keep whatever
  // sat in the center before in the center after.
  useViewportReframe(viewportRef, {
    hostScale,
    ready: !!flow,
    read: () => {
      const { x, y, zoom } = flow!.getViewport()
      return { offset: { x, y }, scale: zoom }
    },
    write: (offset, scale) => void flow!.setViewport({ ...offset, zoom: scale }),
  })
  return { fitView, takeCamera, handleNodesChange, zoomGraphAtCenter }
}
