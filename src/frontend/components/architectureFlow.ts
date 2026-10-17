import type { Edge, Node, ReactFlowInstance } from "@xyflow/react"
import type { DiagramEdge, DiagramNode } from "@/lib/diagramGraph"

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

export interface ArchitectureNodeData extends DiagramNode, Record<string, unknown> {
  onChange: (id: string, patch: Partial<Pick<DiagramNode, "title" | "bullets" | "kind" | "width" | "height" | "position">>) => void
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
