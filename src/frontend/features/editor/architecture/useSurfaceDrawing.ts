import { useCallback, useEffect, useRef, useState, type Dispatch, type PointerEvent as ReactPointerEvent, type RefObject, type SetStateAction } from "react"
import { NODE_MIN_HEIGHT, type GraphFlowEdge, type GraphFlowInstance } from "@/features/editor/architecture/architectureFlow"
import { NODE_MIN_WIDTH } from "@/features/editor/architecture/architectureLabelWidth"
import { trackPointer } from "@/lib/pointer"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { pointInPolygon, pointsBounds } from "@/lib/drawing"
import { centre, type Point, type Rect, type RouteCard } from "@/lib/domain/edgeRouting"

// Any closed loop drawn on the canvas becomes a node in its bounding box; an
// open stroke (a line, a tick) is ignored. Every element is drawn as the same
// box, so the loop's shape carries no meaning beyond its extent.
const MIN_DRAW_SIZE = 30
const MAX_OPEN_GAP_RATIO = 0.3

const boundsOf = (points: readonly Point[]) => pointsBounds(points.map(({ x, y }) => [x, y]))

/**
 * The box a freehand loop encloses, in the points' own coordinate space (the
 * surface passes flow coordinates). Null for fewer than three points, an open
 * stroke (ends further apart than MAX_OPEN_GAP_RATIO of its length), or a loop
 * under MIN_DRAW_SIZE on both axes.
 */
export function drawnBox(points: Point[]): Rect | null {
  const bounds = points.length < 3 ? null : boundsOf(points)
  if (!bounds) return null
  const { minX, minY } = bounds
  const width = bounds.maxX - minX
  const height = bounds.maxY - minY
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
  const bounds = boundsOf(points)
  return !bounds || Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) < DRAW_MIN_SCREEN_SIZE
}

/**
 * Pen drawing and the lasso. Drawing needs no tool: a closed pen stroke becomes
 * a node (`addDraft`). The lasso is the one opt-in tool: its loop selects the
 * nodes whose centres it encloses. Points are sampled in screen pixels and
 * converted to flow space when the stroke ends.
 */
export function useSurfaceDrawing({ flow, viewportRef, readOnly, cards, selectNodes, setEdges, addDraft }: {
  flow: GraphFlowInstance | null
  viewportRef: RefObject<HTMLDivElement | null>
  readOnly: boolean
  cards: readonly RouteCard[]
  selectNodes: (ids: ReadonlySet<string>) => void
  setEdges: Dispatch<SetStateAction<GraphFlowEdge[]>>
  addDraft: (box: Rect) => void
}) {
  const [lasso, setLasso] = useState(false)
  useEscapeLayer(() => setLasso(false), lasso)
  // Points accumulate in a ref so every pointer sample doesn't re-render the
  // whole surface; the preview only syncs to state once per animation frame.
  const drawPointsRef = useRef<Point[]>([])
  const drawFrameRef = useRef<number | null>(null)
  // The pointer samples of the stroke in progress, with the viewport's page offset they were taken against.
  const [drawPreview, setDrawPreview] = useState<{ left: number, top: number, points: Point[] } | null>(null)
  useEffect(() => () => { if (drawFrameRef.current !== null) cancelAnimationFrame(drawFrameRef.current) }, [])
  const startDraw = useCallback((point: { clientX: number, clientY: number }) => {
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect) return
    drawPointsRef.current = [{ x: point.clientX, y: point.clientY }]
    setDrawPreview({ left: rect.left, top: rect.top, points: [...drawPointsRef.current] })
  }, [viewportRef])
  const onDrawPointerMove = useCallback((event: { clientX: number, clientY: number }) => {
    if (drawPointsRef.current.length === 0) return
    drawPointsRef.current.push({ x: event.clientX, y: event.clientY })
    if (drawFrameRef.current !== null) return
    drawFrameRef.current = requestAnimationFrame(() => {
      drawFrameRef.current = null
      setDrawPreview((current) => current && { ...current, points: [...drawPointsRef.current] })
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
      selectNodes(picked)
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
  }, [flow, addDraft, lasso, cards, selectNodes, setEdges])
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
    trackPointer(onDrawPointerMove, (cancelled) => {
      if (cancelled) { finishDraw(false); return }
      const stroke = !isTap(drawPointsRef.current)
      finishDraw(true)
      if (!onCard || !stroke) return
      // The stroke's closing click would otherwise activate the card it was drawn on.
      const swallow = (click: MouseEvent) => click.stopPropagation()
      window.addEventListener("click", swallow, { capture: true, once: true })
      setTimeout(() => window.removeEventListener("click", swallow, { capture: true }))
    })
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
  const preview = drawPreview && drawPreview.points.length > 1 ? `M ${drawPreview.points.map((p) => `${p.x - drawPreview.left} ${p.y - drawPreview.top}`).join(" L ")}` : null
  return { lasso, setLasso, preview, onPanePointerDown, lassoOverlay }
}
