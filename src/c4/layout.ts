// Saved positions, stored the LikeC4 way: one `.likec4/<viewId>.likec4.snap`
// per view, holding the whole laid-out view as JSON5 (what LikeC4's own editor
// writes when a view is arranged by hand). Internally a snapshot is reduced to
// pins: a rect per node and a bend per connection, merged back over LikeC4's
// auto-layout so the saved nodes stay put and nodes added since go beside them.
import type { DiagramEdge, DiagramNode, LayoutedElementView } from '@likec4/core/types'
import JSON5 from 'json5'

export interface NodeRect {
  x: number
  y: number
  width: number
  height: number
}

/** A saved node position; a size left out keeps LikeC4's. */
export interface NodePin {
  x: number
  y: number
  width?: number
  height?: number
}

export interface ViewPins {
  nodes: Record<string, NodePin>
  /** `source->target` -> bend: the curve's offset from the line between the node centres. */
  edges: Record<string, number>
}

/** Per view id; a view without an entry is auto-laid-out and has no snapshot. */
export type Pins = Record<string, ViewPins>

/** Snapshot files by workspace-relative path, e.g. `.likec4/index.likec4.snap`. */
export type Snapshots = Record<string, string>

const SNAPSHOT_DIR = '.likec4/'
const SNAPSHOT_SUFFIX = '.likec4.snap'

export const snapshotPath = (viewId: string) => `${SNAPSHOT_DIR}${viewId}${SNAPSHOT_SUFFIX}`

export const edgeKey = (source: string, target: string) => `${source}->${target}`

const centre = (rect: NodeRect) => ({ x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 })

/** Midpoint of the centre line and its unit normal; a bend moves the curve along that normal. */
function chord(source: NodeRect, target: NodeRect) {
  const a = centre(source)
  const b = centre(target)
  const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
  return { a, b, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, normal: { x: -(b.y - a.y) / length, y: (b.x - a.x) / length } }
}

/** Where the ray from the rect's centre towards `toward` leaves the rect. */
function border(rect: NodeRect, toward: { x: number, y: number }) {
  const c = centre(rect)
  const dx = toward.x - c.x
  const dy = toward.y - c.y
  if (dx === 0 && dy === 0) return c
  const scale = Math.min(dx ? rect.width / 2 / Math.abs(dx) : Infinity, dy ? rect.height / 2 / Math.abs(dy) : Infinity)
  return { x: c.x + dx * scale, y: c.y + dy * scale }
}

const rectOf = (node: NodeRect): NodeRect => ({ x: node.x, y: node.y, width: node.width, height: node.height })

const isObject = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null
const hasCoordinates = (value: unknown, keys: string[]): value is Record<string, unknown> =>
  isObject(value) && keys.every((key) => Number.isFinite(value[key]))
const isSavedNode = (node: unknown): node is NodeRect & { id: string } =>
  hasCoordinates(node, ['x', 'y', 'width', 'height']) && typeof node.id === 'string'
const isSavedEdge = (edge: unknown): edge is { source: string, target: string, controlPoints?: { x: number, y: number }[] | null } =>
  isObject(edge) && typeof edge.source === 'string' && typeof edge.target === 'string'
  && (edge.controlPoints == null || (Array.isArray(edge.controlPoints) && edge.controlPoints.every((point) => hasCoordinates(point, ['x', 'y']))))

/**
 * The pins the snapshot files hold. A file that does not parse, or has any node or
 * edge without string ids and finite coordinates, is ignored as a whole, so its view
 * is auto-laid-out.
 */
export function readPins(snapshots: Snapshots): Pins {
  const pins: Pins = {}
  for (const [path, text] of Object.entries(snapshots)) {
    if (!path.startsWith(SNAPSHOT_DIR) || !path.endsWith(SNAPSHOT_SUFFIX)) continue
    let view: unknown
    try {
      view = JSON5.parse(text)
    } catch {
      continue
    }
    if (!isObject(view)) continue
    const { nodes: savedNodes, edges: savedEdges } = view
    if (!Array.isArray(savedNodes) || !savedNodes.every(isSavedNode) || !Array.isArray(savedEdges) || !savedEdges.every(isSavedEdge)) continue
    const nodes = Object.fromEntries(savedNodes.map((node) => [node.id, rectOf(node)]))
    const edges: Record<string, number> = {}
    for (const edge of savedEdges) {
      const control = edge.controlPoints?.[0]
      const source = nodes[edge.source]
      const target = nodes[edge.target]
      if (!control || !source || !target) continue
      const { mid, normal } = chord(source, target)
      edges[edgeKey(edge.source, edge.target)] = Math.round((control.x - mid.x) * normal.x + (control.y - mid.y) * normal.y)
    }
    pins[path.slice(SNAPSHOT_DIR.length, -SNAPSHOT_SUFFIX.length)] = { nodes, edges }
  }
  return pins
}

/** One cubic Bezier from border to border through the bend, as LikeC4 stores edge paths. */
function edgePath(edge: DiagramEdge, source: NodeRect, target: NodeRect, bend: number | undefined): DiagramEdge {
  const { a, b, mid, normal } = chord(source, target)
  // The quadratic control sits twice as far out as the point the curve passes through.
  const control = bend ? { x: mid.x + normal.x * bend * 2, y: mid.y + normal.y * bend * 2 } : mid
  const start = border(source, bend ? control : b)
  const end = border(target, bend ? control : a)
  const c1 = { x: start.x + (2 / 3) * (control.x - start.x), y: start.y + (2 / 3) * (control.y - start.y) }
  const c2 = { x: end.x + (2 / 3) * (control.x - end.x), y: end.y + (2 / 3) * (control.y - end.y) }
  const at = { x: (start.x + 3 * c1.x + 3 * c2.x + end.x) / 8, y: (start.y + 3 * c1.y + 3 * c2.y + end.y) / 8 }
  const { drifts: _drifts, ...rest } = edge
  return {
    ...rest,
    points: [[start.x, start.y], [c1.x, c1.y], [c2.x, c2.y], [end.x, end.y]],
    controlPoints: bend ? [{ x: mid.x + normal.x * bend, y: mid.y + normal.y * bend }] : null,
    ...(edge.labelBBox ? { labelBBox: { ...edge.labelBBox, x: at.x - edge.labelBBox.width / 2, y: at.y - edge.labelBBox.height / 2 } } : {}),
  }
}

const COMPOUND_PADDING = 32
/** Room above the first child for the compound's own title. */
const COMPOUND_TITLE = 64
/** Between the arranged part of a view and elements added to it since. */
const BLOCK_GAP = 120

/**
 * LikeC4's auto-layout with `pins` applied, step by step:
 * - a pinned node takes its pin, and a pinned bend reshapes its connection;
 * - an unpinned descendant moves by as much as its parent did, keeping its place inside it;
 * - unpinned top-level groups, i.e. elements added since the view was arranged, move as
 *   one block beside the arranged graph;
 * - a compound grows to contain its children and never shrinks.
 * A pin wins over the inherited and block moves; only the growth can still widen a
 * pinned compound. A connection without a bend between two unmoved nodes keeps LikeC4's route.
 */
export function pinView(view: LayoutedElementView, pins: ViewPins | undefined): LayoutedElementView {
  if (!pins) return view
  const place = (node: DiagramNode, rect: NodeRect): DiagramNode =>
    ({ ...node, ...rect, labelBBox: { ...node.labelBBox, x: node.labelBBox.x + rect.x - node.x, y: node.labelBBox.y + rect.y - node.y } })
  // LikeC4's placement, before any pin.
  const laid = new Map<string, DiagramNode>(view.nodes.map((node) => [node.id, node]))
  const byId = new Map(view.nodes.map((node): [string, DiagramNode] => {
    const { drifts: _drifts, ...rest } = node
    const pin = pins.nodes[node.id]
    return [node.id, pin ? place(rest, { x: pin.x, y: pin.y, width: pin.width ?? node.width, height: pin.height ?? node.height }) : rest]
  }))
  // An unpinned child keeps its place inside its parent: it moves by as much as
  // the parent did. Parents go first, so a move passes down the whole subtree.
  for (const node of [...view.nodes].sort((a, b) => a.level - b.level)) {
    const parent = node.parent ? byId.get(node.parent) : undefined
    if (pins.nodes[node.id] || !parent) continue
    const from = laid.get(parent.id)!
    if (parent.x === from.x && parent.y === from.y) continue
    const current = byId.get(node.id)!
    byId.set(node.id, place(current, { x: current.x + parent.x - from.x, y: current.y + parent.y - from.y, width: current.width, height: current.height }))
  }
  // LikeC4 lays out the whole view, so its spots for elements added since the
  // view was arranged collide with the pinned ones. Those top-level newcomers
  // keep their arrangement among themselves and move, as one block, to the right.
  const root = (node: DiagramNode): DiagramNode => (node.parent && byId.has(node.parent) ? root(byId.get(node.parent)!) : node)
  const block = view.nodes.filter((node) => !pins.nodes[node.id] && !pins.nodes[root(node).id])
  const settled = view.nodes.filter((node) => pins.nodes[node.id]).map((node) => byId.get(node.id)!)
  if (block.length > 0 && settled.length > 0) {
    const dx = Math.max(...settled.map((node) => node.x + node.width)) + BLOCK_GAP - Math.min(...block.map((node) => node.x))
    const dy = Math.min(...settled.map((node) => node.y)) - Math.min(...block.map((node) => node.y))
    for (const node of block) byId.set(node.id, place(byId.get(node.id)!, { x: node.x + dx, y: node.y + dy, width: node.width, height: node.height }))
  }
  // A compound always encloses its children, as in LikeC4: a pin saved before a
  // child arrived (or a child pinned elsewhere) only ever grows it, innermost first.
  for (const node of [...view.nodes].sort((a, b) => b.level - a.level)) {
    const children = node.children.map((id) => byId.get(id)).filter((child) => child !== undefined)
    if (children.length === 0) continue
    const current = byId.get(node.id)!
    const x = Math.min(current.x, ...children.map((child) => child.x - COMPOUND_PADDING))
    const y = Math.min(current.y, ...children.map((child) => child.y - COMPOUND_TITLE))
    const right = Math.max(current.x + current.width, ...children.map((child) => child.x + child.width + COMPOUND_PADDING))
    const bottom = Math.max(current.y + current.height, ...children.map((child) => child.y + child.height + COMPOUND_PADDING))
    byId.set(node.id, place(current, { x, y, width: right - x, height: bottom - y }))
  }
  const unmoved = (id: string) => {
    const before = laid.get(id)!
    const after = byId.get(id)!
    return after.x === before.x && after.y === before.y && after.width === before.width && after.height === before.height
  }
  const edges = view.edges.map((edge) => {
    const bend = pins.edges[edgeKey(edge.source, edge.target)]
    if (!bend && unmoved(edge.source) && unmoved(edge.target)) {
      const { drifts: _drifts, ...rest } = edge
      return rest
    }
    return edgePath(edge, byId.get(edge.source)!, byId.get(edge.target)!, bend)
  })
  const nodes = view.nodes.map((node) => byId.get(node.id)!)
  const xs = nodes.flatMap((node) => [node.x, node.x + node.width])
  const ys = nodes.flatMap((node) => [node.y, node.y + node.height])
  const bounds = nodes.length === 0 ? view.bounds : {
    x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys),
  }
  const { drifts: _drifts, ...rest } = view
  return { ...rest, _layout: 'manual', nodes, edges, bounds }
}

/** The snapshot file text, formatted as LikeC4 writes it. */
export const snapshotText = (view: LayoutedElementView) => `${JSON5.stringify(view, { space: 2, quote: "'" })}\n`
