// The React Flow side of the architecture surface: node and edge shapes, and
// the conversion between the graph's absolute positions and React Flow's nesting.
import type { Edge, Node, ReactFlowInstance } from "@xyflow/react"
import type { DiagramEdge, DiagramNode } from "@/features/editor/architecture/diagramGraph"
import type { Point, RouteCard } from "@/lib/domain/edgeRouting"

/** The size of a card React Flow has not measured yet, and of a card's first render. */
export const FALLBACK_SIZE = { width: 224, height: 96 }
export const NODE_MIN_HEIGHT = 48

// Drag/resize commits round to this grid, so autosaved positions stay stable
// pixel values instead of accumulating sub-pixel drift across edit sessions.
export const GRID_SIZE = 8
export const snapToGrid = (value: number) => Math.round(value / GRID_SIZE) * GRID_SIZE

export interface NodeRemoval {
  label: string
  run: (ids: string[], edges?: DiagramEdge[]) => void
}

/** What a card may change about its node; a position is relative to the node's parent. */
export type NodePatch = Partial<Pick<DiagramNode, "title" | "bullets" | "kind" | "width" | "height" | "position">>

export interface ArchitectureNodeData extends DiagramNode, Record<string, unknown> {
  onChange: (id: string, patch: NodePatch) => void
  /** Kinds a selected node may switch to; empty when it cannot. */
  kinds: readonly string[]
  /** True exactly once, for the node the user just created. */
  claimAutoEdit?: (id: string) => boolean
  /** The corner X: what it says and does; absent when read-only. */
  remove?: NodeRemoval
}

export type GraphFlowNode = Node<ArchitectureNodeData, "architecture-node">
export type GraphFlowEdge = Edge<DiagramEdge & Record<string, unknown>, "architecture-edge">
export type GraphFlowInstance = ReactFlowInstance<GraphFlowNode, GraphFlowEdge>

// React Flow wants a parent before its children, and each child positioned
// relative to its parent; the graph keeps absolute positions throughout.
export function toFlowNodes(nodes: DiagramNode[], onNodeChange: ArchitectureNodeData["onChange"], kinds: readonly string[], claimAutoEdit?: ArchitectureNodeData["claimAutoEdit"], remove?: NodeRemoval): GraphFlowNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const depth = (node: DiagramNode): number => {
    const parent = node.parent ? byId.get(node.parent) : undefined
    return parent ? 1 + depth(parent) : 0
  }
  return [...nodes].sort((a, b) => depth(a) - depth(b)).map((node) => {
    const parent = node.parent ? byId.get(node.parent) : undefined
    return {
      id: node.id, type: "architecture-node",
      position: parent ? { x: node.position.x - parent.position.x, y: node.position.y - parent.position.y } : node.position,
      ...(parent ? { parentId: parent.id } : {}),
      style: { width: node.width, height: node.height },
      data: { ...node, onChange: onNodeChange, kinds, claimAutoEdit, remove },
    }
  })
}

export function toFlowEdges(edges: DiagramEdge[]): GraphFlowEdge[] {
  return edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target, type: "architecture-edge", data: { ...edge } }))
}

// Live flow nodes as cards in absolute coordinates, so routing works across nesting levels mid-drag.
// A node React Flow has not measured yet counts at FALLBACK_SIZE.
export function routeCards(nodes: readonly GraphFlowNode[]): RouteCard[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const absolute = (node: GraphFlowNode): Point => {
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    if (!parent) return node.position
    const origin = absolute(parent)
    return { x: origin.x + node.position.x, y: origin.y + node.position.y }
  }
  return nodes.map((node) => ({
    id: node.id, ...absolute(node),
    width: node.measured?.width ?? FALLBACK_SIZE.width, height: node.measured?.height ?? FALLBACK_SIZE.height,
    parent: node.parentId ?? null, compound: node.data.compound,
  }))
}

// Fields React Flow owns on a node/edge; the graph never describes them.
const RETAINED_BY_FLOW = ["selected", "measured", "dragging"] as const

// Rebuild flow items from the graph, carrying over only what React Flow owns. A
// blind replace clears selection and drops measurements on every edit, making
// edges jump and handles fade mid-interaction. Merging the other way round is
// just as wrong: it would keep keys the rebuilt item deliberately omits, e.g. a
// stale `parentId` after a node is dropped out of its frame.
export function reconcile<T extends { id: string }>(current: T[], next: T[]): T[] {
  const previous = new Map(current.map((item) => [item.id, item as Record<string, unknown>]))
  return next.map((item) => {
    const existing = previous.get(item.id)
    if (!existing) return item
    // An update landing mid-drag must not yank the node back to its stored position.
    if (existing.dragging) return existing as T
    const merged: Record<string, unknown> = { ...item }
    for (const key of RETAINED_BY_FLOW) if (key in existing) merged[key] = existing[key]
    return merged as T
  })
}
