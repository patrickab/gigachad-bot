// Connection routing with libavoid (WebAssembly, served from /libavoid.wasm):
// right-angled routes that keep clear of every card a connection does not
// join, with parallel stretches nudged apart so lines never run on top of
// each other. Routing is synchronous once the library has loaded.
import { AvoidLib } from "libavoid-js"

export interface Point { x: number, y: number }

export interface RouteCard {
  id: string
  x: number
  y: number
  width: number
  height: number
  parent: string | null
  /** Frames other cards; connections pass its border freely and never attach to it by pin. */
  compound: boolean
}

export interface RouteEdge {
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
        for (const [x, y, direction] of [[along, 0, DIR_UP], [along, 1, DIR_DOWN], [0, along, DIR_LEFT], [1, along, DIR_RIGHT]] as const) {
          new lib.ShapeConnectionPin(shape, PIN_CLASS, x, y, true, 0, direction).setExclusive(true)
        }
      }
    }
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

const centre = (rect: { x: number, y: number, width: number, height: number }) => ({ x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 })

/** Where a connection with this bend passes: off the midpoint between the card centres, along the normal of the line joining them. */
export function bendPoint(source: { x: number, y: number, width: number, height: number }, target: { x: number, y: number, width: number, height: number }, bend: number): Point {
  const a = centre(source)
  const b = centre(target)
  const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
  return { x: (a.x + b.x) / 2 - (b.y - a.y) / length * bend, y: (a.y + b.y) / 2 + (b.x - a.x) / length * bend }
}

/** The bend that makes a connection pass `point`'s projection onto the normal: the inverse of `bendPoint`. */
export function bendThrough(source: { x: number, y: number, width: number, height: number }, target: { x: number, y: number, width: number, height: number }, point: Point): number {
  const a = centre(source)
  const b = centre(target)
  const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
  return (point.x - (a.x + b.x) / 2) * -(b.y - a.y) / length + (point.y - (a.y + b.y) / 2) * (b.x - a.x) / length
}
