// Connection routing with libavoid (WebAssembly, served from /libavoid.wasm):
// right-angled routes that keep clear of every card a connection does not
// join, with parallel stretches nudged apart so lines never run on top of
// each other. Routing is synchronous once the library has loaded. Also the
// unrouted fallback geometry: attach slots on card sides, elbows, rounded paths.
import { AvoidLib } from "libavoid-js"

/** How connections are drawn: a preference of the window or canvas hosting the graph. */
export type EdgeStyle = "curved" | "elbow"

export interface Point { x: number, y: number }

export type Side = "top" | "right" | "bottom" | "left"

export interface Rect extends Point {
  width: number
  height: number
}

export interface RouteCard extends Rect {
  id: string
  parent: string | null
  /** Frames other cards; connections pass its border freely and never attach to it by pin. */
  compound: boolean
}

interface RouteEdge {
  id: string
  source: string
  target: string
  /** Offset of a via point from the midpoint between the two cards' centres, along the normal of the line joining them; 0 routes freely. */
  bend: number
  /** Where the connection meets a frame endpoint; a plain card picks its own side. */
  sourcePoint?: Point
  targetPoint?: Point
}

// The slice of libavoid's embind API used here; the package ships no usable typings.
interface Handle { delete(): void }
interface PolyLine { size(): number, at(index: number): Point }
interface ShapeRef extends Handle { readonly _shape: never }
interface ConnRef extends Handle { displayRoute(): PolyLine, setRoutingCheckpoints(checkpoints: Handle): void }
interface Router extends Handle {
  processTransaction(): boolean
  setRoutingParameter(parameter: unknown, value: number): void
  setRoutingOption(option: unknown, value: boolean): void
}
interface PinHandle { setExclusive(exclusive: boolean): void }
interface Avoid {
  RouterFlag: { OrthogonalRouting: { value: number } }
  RoutingParameter: Record<"segmentPenalty" | "crossingPenalty" | "shapeBufferDistance" | "idealNudgingDistance", unknown>
  RoutingOption: Record<"nudgeOrthogonalSegmentsConnectedToShapes" | "nudgeSharedPathsWithCommonEndPoint", unknown>
  Router: new (flags: number) => Router
  Point: new (x: number, y: number) => Point & Handle
  Rectangle: new (topLeft: Point, bottomRight: Point) => Handle
  ShapeRef: new (router: Router, polygon: Handle) => ShapeRef
  ShapeConnectionPin: new (shape: ShapeRef, classId: number, xOffset: number, yOffset: number, proportional: boolean, insideOffset: number, directions: number) => PinHandle
  ConnEnd: { new (shape: ShapeRef, classId: number): Handle, new (point: Point): Handle }
  ConnRef: new (router: Router, source: Handle, target: Handle) => ConnRef
  Checkpoint: new (point: Point) => Handle
  CheckpointVector: new () => Handle & { push_back(checkpoint: Handle): void }
}

// Shape connection pin class shared by every pin so ConnEnd can pick any free one.
const PIN_CLASS = 1
// libavoid's ConnDirFlags: the direction a connection leaves a pin in.
const DIR_UP = 1
const DIR_DOWN = 2
const DIR_LEFT = 4
const DIR_RIGHT = 8
// Clearance kept around cards, and between parallel stretches of different connections.
const CLEARANCE = 16
const NUDGE_GAP = 12

let avoid: Avoid | null = null
let loading: Promise<boolean> | null = null

/** Loads libavoid once; resolves false (and logs why) if it cannot load, leaving connections unrouted. */
export function loadRouter(wasmUrl = "/libavoid.wasm"): Promise<boolean> {
  loading ??= AvoidLib.load(wasmUrl).then(() => {
    avoid = AvoidLib.getInstance() as unknown as Avoid
    return true
  }, (error: unknown) => {
    console.error("Connection routing is unavailable; connections are drawn without avoiding cards.", error)
    return false
  })
  return loading
}

/**
 * Routes every connection; null when libavoid has not loaded. A connection is
 * missing from the result when it cannot be routed, e.g. one end inside the other.
 * Every card is an obstacle, frames are not: all connections share one router,
 * which is what lets libavoid see them all and nudge them apart.
 */
export function routeEdges(cards: readonly RouteCard[], edges: readonly RouteEdge[]): Map<string, Point[]> | null {
  const lib = avoid
  if (!lib) return null
  const byId = new Map(cards.map((card) => [card.id, card]))
  const inside = (id: string, frame: string): boolean => {
    for (let card = byId.get(id); card?.parent; card = byId.get(card.parent)) if (card.parent === frame) return true
    return false
  }
  // One end inside the other: there is no outside route between them.
  const routable = edges.filter((edge) => byId.has(edge.source) && byId.has(edge.target) && !inside(edge.source, edge.target) && !inside(edge.target, edge.source))

  const router = new lib.Router(lib.RouterFlag.OrthogonalRouting.value)
  const values: Handle[] = []
  const value = <T extends Handle>(handle: T) => (values.push(handle), handle)
  const point = (at: Point) => value(new lib.Point(at.x, at.y))
  try {
    router.setRoutingParameter(lib.RoutingParameter.shapeBufferDistance, CLEARANCE)
    router.setRoutingParameter(lib.RoutingParameter.idealNudgingDistance, NUDGE_GAP)
    // A bend costs 50 and a crossing 200, so routes cross only to avoid several bends
    router.setRoutingParameter(lib.RoutingParameter.segmentPenalty, 50)
    router.setRoutingParameter(lib.RoutingParameter.crossingPenalty, 200)
    router.setRoutingOption(lib.RoutingOption.nudgeOrthogonalSegmentsConnectedToShapes, true)
    router.setRoutingOption(lib.RoutingOption.nudgeSharedPathsWithCommonEndPoint, true)
    // Each connection end takes its own exclusive pin; libavoid picks the side.
    const degree = new Map<string, number>()
    for (const edge of routable) for (const end of [edge.source, edge.target]) degree.set(end, (degree.get(end) ?? 0) + 1)
    const shapes = new Map<string, ShapeRef>()
    for (const card of cards) {
      if (card.compound) continue
      const shape = new lib.ShapeRef(router, value(new lib.Rectangle(point(card), point({ x: card.x + card.width, y: card.y + card.height }))))
      shapes.set(card.id, shape)
      const pins = degree.get(card.id) ?? 0
      for (let index = 1; index <= pins; index += 1) {
        const along = index / (pins + 1)
        // Pin position is proportional (0..1 of the card): one per side at `along`, each leaving outward
        for (const [x, y, direction] of [[along, 0, DIR_UP], [along, 1, DIR_DOWN], [0, along, DIR_LEFT], [1, along, DIR_RIGHT]] as const) {
          new lib.ShapeConnectionPin(shape, PIN_CLASS, x, y, true, 0, direction).setExclusive(true)
        }
      }
    }
    // A frame has no shape: attach at its given point, else the top-centre
    const end = (id: string, at: Point | undefined) => {
      const shape = shapes.get(id)
      if (shape) return value(new lib.ConnEnd(shape, PIN_CLASS))
      const card = byId.get(id)!
      return value(new lib.ConnEnd(point(at ?? { x: card.x + card.width / 2, y: card.y })))
    }
    const connectors = routable.map((edge) => {
      const connector = new lib.ConnRef(router, end(edge.source, edge.sourcePoint), end(edge.target, edge.targetPoint))
      if (edge.bend) {
        const checkpoints = value(new lib.CheckpointVector())
        checkpoints.push_back(value(new lib.Checkpoint(point(bendPoint(byId.get(edge.source)!, byId.get(edge.target)!, edge.bend)))))
        connector.setRoutingCheckpoints(checkpoints)
      }
      return [edge.id, connector] as const
    })
    router.processTransaction()
    const routes = new Map<string, Point[]>()
    for (const [id, connector] of connectors) {
      const line = connector.displayRoute()
      const points: Point[] = []
      for (let index = 0; index < line.size(); index += 1) {
        const at = line.at(index)
        points.push({ x: at.x, y: at.y })
      }
      if (points.length >= 2) routes.set(id, points)
    }
    return routes
  } finally {
    // The router owns its shapes, pins and connectors; plain values are ours to free.
    for (const handle of values) handle.delete()
    router.delete()
  }
}

export const centre = (rect: Rect): Point => ({ x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 })

/** The point `offset` off the midpoint of `a` and `b`, along the normal of the line joining them. */
export function normalPoint(a: Point, b: Point, offset: number): Point {
  const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
  return { x: (a.x + b.x) / 2 - (b.y - a.y) / length * offset, y: (a.y + b.y) / 2 + (b.x - a.x) / length * offset }
}

/** The offset that puts `normalPoint(a, b, offset)` level with `point`: its inverse. */
export function normalOffset(a: Point, b: Point, point: Point): number {
  const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
  return (point.x - (a.x + b.x) / 2) * -(b.y - a.y) / length + (point.y - (a.y + b.y) / 2) * (b.x - a.x) / length
}

/** Where a connection with this bend passes: off the midpoint between the card centres, along the normal of the line joining them. */
export const bendPoint = (source: Rect, target: Rect, bend: number): Point => normalPoint(centre(source), centre(target), bend)

/** The bend that makes a connection pass `point`'s projection onto the normal: the inverse of `bendPoint`. */
export const bendThrough = (source: Rect, target: Rect, point: Point): number => normalOffset(centre(source), centre(target), point)

// Soft avoidance: connections sharing a card side get their own slot on it
// instead of all leaving from the midpoint, so their paths run in separate
// lanes. Lines may still cross. This is the fallback before libavoid loads.
const SIDE_SPREAD = 0.68

interface AttachPoint extends Point {
  position: Side
}

export interface EdgeAttachment {
  source: AttachPoint
  target: AttachPoint
}

/** The point `ratio` (0 to 1) along `side` of `rect`. */
function slotPoint({ x, y, width, height }: Rect, side: Side, ratio: number): AttachPoint {
  if (side === "top") return { x: x + width * ratio, y, position: side }
  if (side === "bottom") return { x: x + width * ratio, y: y + height, position: side }
  if (side === "left") return { x, y: y + height * ratio, position: side }
  return { x: x + width, y: y + height * ratio, position: side }
}

function sideToward(rect: Rect, toward: Rect): Side {
  const from = centre(rect)
  const to = centre(toward)
  const dx = to.x - from.x
  const dy = to.y - from.y
  // Compare against the card's own aspect so wide cards still prefer left/right.
  if (Math.abs(dx) * rect.height > Math.abs(dy) * rect.width) return dx > 0 ? "right" : "left"
  return dy > 0 ? "bottom" : "top"
}

/**
 * Slot points for every edge, keyed by edge id. An edge with an endpoint
 * missing from `cards` is omitted.
 */
export function edgeAttachments(cards: readonly RouteCard[], edges: ReadonlyArray<Pick<RouteEdge, "id" | "source" | "target">>): Map<string, EdgeAttachment> {
  const byId = new Map(cards.map((card) => [card.id, card]))
  const ends = edges.flatMap((edge) => {
    const source = byId.get(edge.source)
    const target = byId.get(edge.target)
    if (!source || !target) return []
    return [
      { edgeId: edge.id, role: "source" as const, nodeId: edge.source, rect: source, side: sideToward(source, target), toward: target },
      { edgeId: edge.id, role: "target" as const, nodeId: edge.target, rect: target, side: sideToward(target, source), toward: source },
    ]
  })
  const groups = new Map<string, typeof ends>()
  for (const end of ends) {
    const key = `${end.nodeId}:${end.side}`
    groups.set(key, [...(groups.get(key) ?? []), end])
  }
  const points = new Map<string, AttachPoint>()
  for (const group of groups.values()) {
    // Order along the side by where the other card sits, so neighbouring lines
    // keep their relative order and do not cross just to reach their slot.
    const axis = group[0].side === "top" || group[0].side === "bottom" ? "x" : "y"
    const sorted = [...group].sort((a, b) => centre(a.toward)[axis] - centre(b.toward)[axis])
    sorted.forEach((end, index) => {
      const share = (index + 1) / (sorted.length + 1)
      const ratio = 0.5 + (share - 0.5) * SIDE_SPREAD
      points.set(`${end.edgeId}:${end.role}`, slotPoint(end.rect, end.side, ratio))
    })
  }
  return new Map(edges.flatMap((edge) => {
    const source = points.get(`${edge.id}:source`)
    const target = points.get(`${edge.id}:target`)
    return source && target ? [[edge.id, { source, target }] as const] : []
  }))
}

/** A polyline with each corner rounded off by up to `radius`, never more than half of either adjoining segment. */
export function roundedPath(points: readonly Point[], radius: number): string {
  let d = `M ${points[0].x} ${points[0].y}`
  for (let index = 1; index < points.length - 1; index += 1) {
    const prev = points[index - 1]
    const corner = points[index]
    const next = points[index + 1]
    const before = Math.hypot(corner.x - prev.x, corner.y - prev.y)
    const after = Math.hypot(next.x - corner.x, next.y - corner.y)
    if (before < 0.01 || after < 0.01) continue
    const cut = Math.min(radius, before / 2, after / 2)
    d += ` L ${corner.x - (corner.x - prev.x) / before * cut} ${corner.y - (corner.y - prev.y) / before * cut}`
      + ` Q ${corner.x} ${corner.y} ${corner.x + (next.x - corner.x) / after * cut} ${corner.y + (next.y - corner.y) / after * cut}`
  }
  const last = points[points.length - 1]
  return `${d} L ${last.x} ${last.y}`
}

/** The point halfway along a polyline. */
export function polylineMidpoint(points: readonly Point[]): Point {
  const lengths = points.slice(1).map((point, index) => Math.hypot(point.x - points[index].x, point.y - points[index].y))
  let left = lengths.reduce((sum, length) => sum + length, 0) / 2
  for (const [index, length] of lengths.entries()) {
    if (left <= length && length > 0) {
      const from = points[index]
      const to = points[index + 1]
      return { x: from.x + (to.x - from.x) * left / length, y: from.y + (to.y - from.y) * left / length }
    }
    left -= length
  }
  return points[0]
}

/** Arrowhead at `tip`, pointing away from `from`: 10px wings spread 0.4 rad either side. */
export function arrowHeadPath(tip: Point, from: Point): string {
  const angle = Math.atan2(tip.y - from.y, tip.x - from.x)
  const size = 10
  const wing = 0.4
  const left = { x: tip.x - Math.cos(angle - wing) * size, y: tip.y - Math.sin(angle - wing) * size }
  const right = { x: tip.x - Math.cos(angle + wing) * size, y: tip.y - Math.sin(angle + wing) * size }
  return `M ${left.x} ${left.y} L ${tip.x} ${tip.y} L ${right.x} ${right.y} Z`
}

// Orthogonal connector, like a flowchart: leave each card perpendicular to its
// side. Facing sides give a Z whose middle lane `bend` shifts along the axis;
// perpendicular sides give a single L. Corners are rounded.
export const CORNER_RADIUS = 8

/** The unrouted elbow between two attach points; `axis` is the way a Z's middle lane slides, null for an L. */
export function orthogonalRoute(from: AttachPoint, to: AttachPoint, bend: number) {
  const horizontal = (side: Side) => side === "left" || side === "right"
  let axis: "x" | "y" | null = null
  let raw: Point[]
  if (horizontal(from.position) && horizontal(to.position)) {
    const lane = (from.x + to.x) / 2 + bend
    raw = [from, { x: lane, y: from.y }, { x: lane, y: to.y }, to]
    axis = "x"
  } else if (!horizontal(from.position) && !horizontal(to.position)) {
    const lane = (from.y + to.y) / 2 + bend
    raw = [from, { x: from.x, y: lane }, { x: to.x, y: lane }, to]
    axis = "y"
  } else {
    raw = [from, horizontal(from.position) ? { x: to.x, y: from.y } : { x: from.x, y: to.y }, to]
  }
  const points = raw.filter((point, index) => index === 0 || Math.hypot(point.x - raw[index - 1].x, point.y - raw[index - 1].y) > 0.01)
  // The lane is only draggable on a Z; the handle rides its middle segment.
  const draggable = axis !== null && points.length === 4
  const handle = draggable ? { x: (points[1].x + points[2].x) / 2, y: (points[1].y + points[2].y) / 2 } : null
  return { points, d: roundedPath(points, CORNER_RADIUS), axis: draggable ? axis : null, handle }
}
