import { parse, stringify } from "yaml"

export const ARCHITECTURE_GRAPH_VERSION = 1

export interface ArchitectureGraphPosition {
  x: number
  y: number
}

export interface ArchitectureGraphNode {
  id: string
  title: string
  bullets: string[]
  position: ArchitectureGraphPosition
}

export type ArchitectureGraphEdgeDirection = "one-way" | "bidirectional"

export interface ArchitectureGraphEdge {
  id: string
  source: string
  target: string
  direction: ArchitectureGraphEdgeDirection
  label?: string
}

export interface ArchitectureGraph {
  version: typeof ARCHITECTURE_GRAPH_VERSION
  title: string
  nodes: ArchitectureGraphNode[]
  edges: ArchitectureGraphEdge[]
}

export interface ArchitectureGraphDocument {
  name: string
  path: string
  content: string
  hasDraft: boolean
}

export function isArchitectureGraphPath(path: string): boolean {
  return path.endsWith(".architecture.yaml")
}

export const emptyArchitectureGraph = (title = "Untitled architecture"): ArchitectureGraph => ({
  version: ARCHITECTURE_GRAPH_VERSION,
  title,
  nodes: [],
  edges: [],
})

function text(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} must be a non-empty string`)
  return value.trim()
}

function record(value: unknown, field: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${field} must be an object`)
  return value as Record<string, unknown>
}

function number(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${field} must be a finite number`)
  return value
}

/** Validates the compact, portable v1 graph document used by both Source and Diagram views. */
export function validateArchitectureGraph(value: unknown): ArchitectureGraph {
  const graph = record(value, "Architecture Graph")
  if (graph.version !== ARCHITECTURE_GRAPH_VERSION) throw new Error(`version must be ${ARCHITECTURE_GRAPH_VERSION}`)
  const nodesValue = graph.nodes
  const edgesValue = graph.edges
  if (!Array.isArray(nodesValue)) throw new Error("nodes must be an array")
  if (!Array.isArray(edgesValue)) throw new Error("edges must be an array")

  const nodeIds = new Set<string>()
  const nodes = nodesValue.map((value, index) => {
    const node = record(value, `nodes[${index}]`)
    const id = text(node.id, `nodes[${index}].id`)
    if (nodeIds.has(id)) throw new Error(`Duplicate node id: ${id}`)
    nodeIds.add(id)
    const position = record(node.position, `nodes[${index}].position`)
    if (!Array.isArray(node.bullets) || node.bullets.some((bullet) => typeof bullet !== "string")) {
      throw new Error(`nodes[${index}].bullets must be an array of strings`)
    }
    return {
      id,
      title: text(node.title, `nodes[${index}].title`),
      bullets: node.bullets.map((bullet) => bullet.trim()).filter(Boolean),
      position: { x: number(position.x, `nodes[${index}].position.x`), y: number(position.y, `nodes[${index}].position.y`) },
    }
  })

  const edgeIds = new Set<string>()
  const edges = edgesValue.map((value, index) => {
    const edge = record(value, `edges[${index}]`)
    const id = text(edge.id, `edges[${index}].id`)
    if (edgeIds.has(id)) throw new Error(`Duplicate edge id: ${id}`)
    edgeIds.add(id)
    const source = text(edge.source, `edges[${index}].source`)
    const target = text(edge.target, `edges[${index}].target`)
    if (!nodeIds.has(source) || !nodeIds.has(target)) throw new Error(`edges[${index}] references a missing node`)
    if (edge.direction !== "one-way" && edge.direction !== "bidirectional") throw new Error(`edges[${index}].direction must be one-way or bidirectional`)
    if (edge.label !== undefined && typeof edge.label !== "string") throw new Error(`edges[${index}].label must be a string`)
    return { id, source, target, direction: edge.direction as ArchitectureGraphEdgeDirection, ...(edge.label?.trim() ? { label: edge.label.trim() } : {}) }
  })

  return { version: ARCHITECTURE_GRAPH_VERSION, title: text(graph.title, "title"), nodes, edges }
}

export function parseArchitectureGraph(source: string): ArchitectureGraph {
  try {
    return validateArchitectureGraph(parse(source))
  } catch (error) {
    throw new Error(error instanceof Error ? error.message : "Invalid Architecture Graph YAML")
  }
}

export function serializeArchitectureGraph(graph: ArchitectureGraph): string {
  return stringify(validateArchitectureGraph(graph), { lineWidth: 0 })
}

export function nextArchitectureGraphId(prefix: "node" | "edge", existing: Iterable<string>): string {
  const ids = new Set(existing)
  let suffix = 1
  while (ids.has(`${prefix}-${suffix}`)) suffix += 1
  return `${prefix}-${suffix}`
}
