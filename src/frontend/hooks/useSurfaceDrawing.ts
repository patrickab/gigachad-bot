import { useCallback, useEffect, useRef, useState, type Dispatch, type PointerEvent as ReactPointerEvent, type RefObject, type SetStateAction } from "react"
import { NODE_MIN_HEIGHT, type GraphFlowEdge, type GraphFlowInstance, type GraphFlowNode } from "@/components/architectureFlow"
import { NODE_MIN_WIDTH } from "@/components/architectureLabelWidth"
import { pointInPolygon } from "@/lib/drawing"
import { centre, type Point, type Rect, type RouteCard } from "@/lib/edgeRouting"

// Any closed loop drawn on the canvas becomes a node in its bounding box; an
// open stroke (a line, a tick) is ignored. Every element is drawn as the same
// box, so the loop's shape carries no meaning beyond its extent.
const MIN_DRAW_SIZE = 30
const MAX_OPEN_GAP_RATIO = 0.3

/**
 * The box a freehand loop encloses, in the points' own coordinate space (the
 * surface passes flow coordinates). Null for fewer than three points, an open
 * stroke (ends further apart than MAX_OPEN_GAP_RATIO of its length), or a loop
 * under MIN_DRAW_SIZE on both axes.
 */
export function drawnBox(points: Point[]): Rect | null {
  if (points.length < 3) return null
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const width = maxX - minX
  const height = maxY - minY
  if (Math.max(width, height) < MIN_DRAW_SIZE) return null

  let pathLength = 0
  for (let index = 1; index < points.length; index += 1) pathLength += Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
  const gap = Math.hypot(points[points.length - 1].x - points[0].x, points[points.length - 1].y - points[0].y)
  if (pathLength === 0 || gap / pathLength > MAX_OPEN_GAP_RATIO) return null
  return { x: minX, y: minY, width, height }
}

// Minimum on-screen size a drawn stroke's bounding box must reach to register,
// independent of canvas zoom.
const DRAW_MIN_SCREEN_SIZE = 30
const isTap = (points: readonly Point[]) => {
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  return points.length === 0 || Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) < DRAW_MIN_SCREEN_SIZE
}

/**
 * Pen drawing and the lasso. Drawing needs no tool: a closed pen stroke becomes
 * a node (`addDraft`). The lasso is the one opt-in tool: its loop selects the
 * nodes whose centres it encloses. Points are sampled in screen pixels and
 * converted to flow space when the stroke ends.
 */
export function useSurfaceDrawing({ flow, viewportRef, readOnly, cards, setNodes, setEdges, addDraft }: {
  flow: GraphFlowInstance | null
  viewportRef: RefObject<HTMLDivElement | null>
  readOnly: boolean
  cards: readonly RouteCard[]
  setNodes: Dispatch<SetStateAction<GraphFlowNode[]>>
  setEdges: Dispatch<SetStateAction<GraphFlowEdge[]>>
  addDraft: (box: Rect) => void
}) {
  const [lasso, setLasso] = useState(false)
  useEffect(() => {
    if (!lasso) return
    const onKeyDown = (event: globalThis.KeyboardEvent) => { if (event.key === "Escape") setLasso(false) }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [lasso])
  // Points accumulate in a ref so every pointer sample doesn't re-render the
  // whole surface; the preview only syncs to state once per animation frame.
  const drawPointsRef = useRef<Point[]>([])
  const drawFrameRef = useRef<number | null>(null)
  const [drawPreview, setDrawPreview] = useState<Point[] | null>(null)
  const drawOriginRef = useRef({ left: 0, top: 0 })
  useEffect(() => () => { if (drawFrameRef.current !== null) cancelAnimationFrame(drawFrameRef.current) }, [])
  const startDraw = useCallback((point: { clientX: number, clientY: number }) => {
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect) return
    drawOriginRef.current = { left: rect.left, top: rect.top }
    drawPointsRef.current = [{ x: point.clientX, y: point.clientY }]
    setDrawPreview(drawPointsRef.current)
  }, [viewportRef])
  const onDrawPointerMove = useCallback((event: { clientX: number, clientY: number }) => {
    if (drawPointsRef.current.length === 0) return
    drawPointsRef.current.push({ x: event.clientX, y: event.clientY })
    if (drawFrameRef.current !== null) return
    drawFrameRef.current = requestAnimationFrame(() => {
      drawFrameRef.current = null
      setDrawPreview([...drawPointsRef.current])
    })
  }, [])
  // Whether a gesture was a tap is judged in screen pixels, before the points
  // are converted to flow space, so it does not change with zoom.
  const finishDraw = useCallback((commit: boolean) => {
    if (drawFrameRef.current !== null) { cancelAnimationFrame(drawFrameRef.current); drawFrameRef.current = null }
    const points = drawPointsRef.current
    drawPointsRef.current = []
    setDrawPreview(null)
    if (!commit || !flow || points.length === 0) return
    const tap = isTap(points)
    if (lasso) {
      // A tap clears the selection; a loop selects every node whose centre it
      // encloses, plus the connections between them, so one Delete removes the lot.
      const polygon = tap ? [] : points.map((p): [number, number] => { const at = flow.screenToFlowPosition(p); return [at.x, at.y] })
      const picked = new Set(cards.filter((card) => {
        const middle = centre(card)
        return pointInPolygon(middle.x, middle.y, polygon)
      }).map((card) => card.id))
      setNodes((current) => current.map((node) => ({ ...node, selected: picked.has(node.id) })))
      setEdges((current) => current.map((edge) => ({ ...edge, selected: picked.has(edge.source) && picked.has(edge.target) })))
      // Leave the tool once something is caught: the overlay would otherwise
      // swallow the drag that moves the selection.
      if (picked.size > 0) setLasso(false)
      return
    }
    // A tap is not a stroke: the pane click handler deselects.
    if (tap) return
    const box = drawnBox(points.map((p) => flow.screenToFlowPosition(p)))
    if (box) addDraft({ x: box.x, y: box.y, width: Math.max(NODE_MIN_WIDTH, Math.round(box.width)), height: Math.max(NODE_MIN_HEIGHT, Math.round(box.height)) })
  }, [flow, addDraft, lasso, cards, setNodes, setEdges])
  // Pen draws, on the bare canvas and on any card that is not activated, which
  // is just more paper; touch and mouse pan. Window listeners, not pointer
  // capture: the pane still receives its click.
  const onPanePointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (readOnly || lasso || event.pointerType !== "pen" || event.button !== 0) return
    const target = event.target as Element
    const card = target.closest(".react-flow__node")
    const onCard = !!card && !card.classList.contains("selected") && !target.closest(".nodrag")
    if (!onCard && !target.classList.contains("react-flow__pane")) return
    // Stops the pen's compatibility mousedown, which would also start React Flow's pan.
    event.preventDefault()
    startDraw(event)
    const cleanup = () => {
      window.removeEventListener("pointermove", onDrawPointerMove)
      window.removeEventListener("pointerup", up)
      window.removeEventListener("pointercancel", cancel)
    }
    const up = () => {
      cleanup()
      const stroke = !isTap(drawPointsRef.current)
      finishDraw(true)
      if (!onCard || !stroke) return
      // The stroke's closing click would otherwise activate the card it was drawn on.
      const swallow = (click: MouseEvent) => click.stopPropagation()
      window.addEventListener("click", swallow, { capture: true, once: true })
      setTimeout(() => window.removeEventListener("click", swallow, { capture: true }))
    }
    const cancel = () => { cleanup(); finishDraw(false) }
    window.addEventListener("pointermove", onDrawPointerMove)
    window.addEventListener("pointerup", up)
    window.addEventListener("pointercancel", cancel)
  }, [readOnly, lasso, startDraw, onDrawPointerMove, finishDraw])
  // The lasso draws on an overlay that captures the pointer.
  const lassoOverlay = {
    onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId)
      startDraw(event)
    },
    onPointerMove: onDrawPointerMove,
    onPointerUp: () => finishDraw(true),
    onPointerCancel: () => finishDraw(false),
  }
  const { left, top } = drawOriginRef.current
  const preview = drawPreview && drawPreview.length > 1 ? `M ${drawPreview.map((p) => `${p.x - left} ${p.y - top}`).join(" L ")}` : null
  return { lasso, setLasso, preview, onPanePointerDown, lassoOverlay }
}
