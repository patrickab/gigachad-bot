// Pure edits the architecture surface applies to its graph: each takes the
// graph's nodes and returns the new list, with absolute positions throughout.
import type { Point } from "@/lib/domain/edgeRouting"
import { isDraft, nextDraftId, type DiagramNode } from "./diagramGraph"
import { GRID_SIZE, snapToGrid, type NodePatch } from "./architectureFlow"

/** A node as React Flow reports it at the end of a drag: positioned relative to its parent. */
export interface DraggedNode {
  id: string
  position: Point
  measured?: { width?: number, height?: number }
}

/** Applies a card's patch, turning its parent-relative position into an absolute one. */
export function patchNode(nodes: readonly DiagramNode[], id: string, patch: NodePatch): DiagramNode[] {
  const node = nodes.find((candidate) => candidate.id === id)
  const parent = node?.parent ? nodes.find((candidate) => candidate.id === node.parent) : undefined
  const absolute = patch.position && parent
    ? { ...patch, position: { x: patch.position.x + parent.position.x, y: patch.position.y + parent.position.y } }
    : patch
  return nodes.map((candidate) => candidate.id === id ? { ...candidate, ...absolute } : candidate)
}

/**
 * Clones `candidates` as titled drafts, so each copy becomes a new element,
 * offset diagonally so the copies read as new objects rather than sitting
 * exactly on their sources. Ids avoid `occupied` and each other.
 * Compounds are skipped: copying one would not copy what it contains.
 */
export function duplicateNodes(candidates: readonly DiagramNode[], occupied: Iterable<string>): DiagramNode[] {
  const taken = new Set(occupied)
  const clones: DiagramNode[] = []
  for (const node of candidates) {
    if (node.compound) continue
    const id = nextDraftId(taken)
    taken.add(id)
    clones.push({ ...node, id, position: { x: snapToGrid(node.position.x + GRID_SIZE * 3), y: snapToGrid(node.position.y + GRID_SIZE * 3) } })
  }
  return clones
}

/**
 * The deepest node whose box holds the point: where a drawn or dropped node
 * belongs. Drawing may nest into any element (a leaf then becomes a parent),
 * dropping only into one that already frames others. Ids are element FQNs, so
 * nesting depth is the FQN's length, and `excluded` drops a node's own subtree.
 */
export function innermostNode(nodes: readonly DiagramNode[], x: number, y: number, excluded: ReadonlySet<string>, compoundsOnly: boolean): string | null {
  let best: string | null = null
  for (const node of nodes) {
    if ((compoundsOnly && !node.compound) || isDraft(node.id) || [...excluded].some((id) => node.id === id || node.id.startsWith(`${id}.`))) continue
    if (x < node.position.x || y < node.position.y || x > node.position.x + node.width || y > node.position.y + node.height) continue
    if (best === null || node.id.split(".").length > best.split(".").length) best = node.id
  }
  return best
}

/**
 * Snaps every dragged node, carries nested nodes along (React Flow moves them
 * relative to their parent; the graph stores absolute positions), and a
 * single node dropped into a different compound moves into that element.
 */
export function settleDrag(nodes: readonly DiagramNode[], dragged: readonly DraggedNode[]): DiagramNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const shifts = new Map<string, Point>()
  const placed = new Map<string, Point>()
  for (const moved of dragged) {
    const node = byId.get(moved.id)
    if (!node) continue
    const origin = node.parent ? byId.get(node.parent)?.position ?? { x: 0, y: 0 } : { x: 0, y: 0 }
    const position = { x: snapToGrid(origin.x + moved.position.x), y: snapToGrid(origin.y + moved.position.y) }
    placed.set(node.id, position)
    shifts.set(node.id, { x: position.x - node.position.x, y: position.y - node.position.y })
  }
  const shiftOf = (node: DiagramNode): Point | undefined => {
    for (let parent = node.parent; parent; parent = byId.get(parent)?.parent ?? null) {
      const shift = shifts.get(parent)
      if (shift) return shift
    }
    return undefined
  }
  let reparent: { id: string, parent: string | null } | null = null
  const [single] = dragged
  const lone = dragged.length === 1 && single ? byId.get(single.id) : undefined
  if (lone && !lone.compound) {
    const at = placed.get(lone.id)!
    const width = single.measured?.width ?? lone.width
    const height = single.measured?.height ?? lone.height
    const parent = innermostNode(nodes, at.x + width / 2, at.y + height / 2, new Set([lone.id]), true)
    if (parent !== lone.parent) reparent = { id: lone.id, parent }
  }
  return nodes.map((node) => {
    const position = placed.get(node.id)
    const shift = position ? undefined : shiftOf(node)
    const next = position ? { ...node, position } : shift ? { ...node, position: { x: node.position.x + shift.x, y: node.position.y + shift.y } } : node
    return reparent?.id === node.id ? { ...next, parent: reparent.parent } : next
  })
}
