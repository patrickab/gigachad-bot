// LikeC4 architecture workspaces: wire types for /api/architecture and the
// adapter between a parsed view and the drawing surface's graph. The `.c4`
// sources are the truth; the surface only proposes changes, which this module
// turns into operations for the C4 service: model and view edits, plus
// `layout` operations for positions (saved as LikeC4 manual layouts).

export interface ArchitectureElement {
  id: string
  name: string
  kind: string
  title: string
  description: string
  parent: string | null
  /** The source file declaring it. */
  file: string | null
}

export interface ArchitectureViewNode {
  id: string
  title: string
  kind: string
  description: string
  parent: string | null
  compound: boolean
  x: number
  y: number
  width: number
  height: number
}

export interface ArchitectureViewEdge {
  /** `source->target`, the layout key; a view edge may merge several relations. */
  id: string
  source: string
  target: string
  label: string | null
  relations: string[]
  /** Saved curve offset, null when drawn straight. */
  bend: number | null
}

export interface ArchitectureView {
  id: string
  title: string
  /** The source file declaring the view; null for one LikeC4 generates. */
  file: string | null
  /** `view x of shop` draws shop's surroundings; a landscape view (null) draws what it names. */
  scope: string | null
  /** Built only from element includes and excludes, so gestures map onto it. */
  editable: boolean
  /** Positions are saved; otherwise LikeC4 lays it out afresh on every change. */
  manual: boolean
  nodes: ArchitectureViewNode[]
  edges: ArchitectureViewEdge[]
}

export interface ArchitectureModel {
  /** Parse diagnostics; `line` is zero-based, as LikeC4 reports it. */
  errors: Array<{ message: string, file: string, line: number }>
  kinds: string[]
  elements: ArchitectureElement[]
  views: ArchitectureView[]
  tree: ArchitectureTreeEntry[]
}

/**
 * How a source file is listed, from what it declares: a system declares one
 * top-level element (`backend/backend.c4`), a module one element extending
 * another file's (`backend/api.c4`), a view file only views.
 */
export interface ArchitectureTreeEntry {
  path: string
  role: "system" | "module" | "view" | "file"
  element: string | null
  /** The view a window opens for the file; a module opens its system's. */
  view: string | null
}

export interface NodePin {
  x: number
  y: number
  width?: number
  height?: number
}

export type EdgeStyle = "curved" | "elbow"

export interface ArchitectureWorkspace {
  model: ArchitectureModel
  created: Array<string | null>
}

export type ArchitectureOperation =
  /** Creates `<id>/<id>.c4`, one system and its view; `created` holds that path. */
  | { op: "createSystem", title: string }
  /** Creates a file beside the system's, extending it with one container; `created` holds that path. */
  | { op: "createModule", system: string, title: string }
  /** Creates a saved view in its own file, `views/<id>.c4`; `created` holds that path. */
  | { op: "createView", title: string }
  | { op: "addFileView" }
  | { op: "addElement", parent: string | null, kind: string, title: string, description?: string, layout?: { view: string } & NodePin }
  | { op: "addRelation", source: string, target: string, label?: string }
  | { op: "delete", elements: string[], relations: string[] }
  | { op: "setTitle", element: string, title: string }
  | { op: "setDescription", element: string, description: string }
  | { op: "setKind", element: string, kind: string }
  | { op: "setLabel", relation: string, label: string }
  | { op: "rename", element: string, id: string }
  | { op: "reparent", element: string, parent: string | null }
  /** With `descendants`, writes `include x, x.**`: the element framing everything inside it. */
  | { op: "includeInView", view: string, elements: string[], descendants?: boolean }
  /** Edits the view only; the elements stay in the model. */
  | { op: "removeFromView", view: string, elements: string[] }
  /** A null entry returns a node to LikeC4's placement or straightens a connection. */
  | { op: "layout", view: string, nodes: Record<string, NodePin | null>, edges: Record<string, { bend: number } | null> }

// --- Surface graph -----------------------------------------------------------

export interface DiagramNode {
  /** Element FQN, or a DRAFT_PREFIX id for a drawn node that has no title yet. */
  id: string
  title: string
  /** The element description, one line per bullet. */
  bullets: string[]
  kind: string
  /** Enclosing node in this view (a compound), not necessarily the element's model parent. */
  parent: string | null
  compound: boolean
  /** Absolute flow coordinates, also for nested nodes. */
  position: { x: number, y: number }
  width?: number
  height?: number
}

export interface DiagramEdge {
  id: string
  source: string
  target: string
  label?: string
  /** Empty for a connection drawn but not yet in the model. */
  relations: string[]
  path?: { bend: number }
}

export interface DiagramGraph {
  nodes: DiagramNode[]
  edges: DiagramEdge[]
}

/** LikeC4 ids cannot contain `~`, so a draft id never collides with an element. */
export const DRAFT_PREFIX = "~draft-"
export const isDraft = (id: string) => id.startsWith(DRAFT_PREFIX)

export function nextDraftId(taken: Iterable<string>): string {
  const used = new Set(taken)
  let index = 1
  while (used.has(`${DRAFT_PREFIX}${index}`)) index += 1
  return `${DRAFT_PREFIX}${index}`
}

/** Any `.c4` source in a project's workspace, nested folders included: `graph/<slug>/backend/api.c4`. */
export const isArchitecturePath = (path: string) => /^graph\/[^/]+\/(?:[^/]+\/)*[^/]+\.c4$/.test(path)
/** `graph/<slug>/backend/api.c4` -> `<slug>`. */
export const architectureSlug = (path: string) => path.split("/")[1] ?? ""
/** `graph/<slug>/backend/api.c4` -> `backend/api.c4`, the path inside the workspace. */
export const architectureSource = (path: string) => path.split("/").slice(2).join("/")

/** The view as the surface draws it; LikeC4 already applied the saved positions. */
export function graphFromView(view: ArchitectureView): DiagramGraph {
  return {
    nodes: view.nodes.map((node) => ({
      id: node.id,
      title: node.title,
      bullets: node.description.split("\n").map((line) => line.trim()).filter(Boolean),
      kind: node.kind,
      parent: node.parent,
      compound: node.compound,
      position: { x: node.x, y: node.y },
      width: node.width,
      height: node.height,
    })),
    edges: view.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      ...(edge.label ? { label: edge.label } : {}),
      relations: edge.relations,
      ...(edge.bend !== null ? { path: { bend: edge.bend } } : {}),
    })),
  }
}

/** The narrowest card: resize floor and minimum drawn size; a card's labels raise it (`nodeMinWidth`). */
export const NODE_MIN_WIDTH = 80
// Horizontal room the title bar needs beyond its heading and kind label: its
// padding, the gap between the two, and a little slack for the caret while editing.
const TITLE_WIDTH_SLACK = 34

// One hidden title bar measures every card's heading and kind label in the real
// (hand-drawn) font: the surface's font, set here too since the spans live on <body>.
// Cached per text; the cache is dropped once that font has loaded, since earlier
// measurements used the fallback.
const labelWidths = new Map<string, number>()
let labelMeasure: { title: HTMLSpanElement, kind: HTMLSpanElement } | null = null
/**
 * Settles once the page's fonts have loaded, after dropping every width cached
 * before then, so the next `nodeMinWidth` call measures in the real font;
 * callers re-measure when it settles. Already resolved where there is no
 * document or Font Loading API.
 */
export const labelFontsReady: Promise<void> = typeof document === "undefined" || !document.fonts
  ? Promise.resolve()
  : document.fonts.ready.then(() => { labelWidths.clear() })
/** The narrowest a card can be and still show its whole heading and kind label. */
export function nodeMinWidth(title: string, kind: string): number {
  if (typeof document === "undefined") return NODE_MIN_WIDTH
  const key = `${title}\u0000${kind}`
  let width = labelWidths.get(key)
  if (width === undefined) {
    if (!labelMeasure) {
      const span = (className: string) => {
        const element = document.createElement("span")
        element.className = className
        element.setAttribute("aria-hidden", "true")
        element.style.cssText = "position:absolute;left:-9999px;top:0;visibility:hidden;white-space:nowrap;font-family:var(--font-handwriting),cursive"
        document.body.appendChild(element)
        return element
      }
      labelMeasure = { title: span("architecture-diagram-title"), kind: span("architecture-diagram-node-kind") }
    }
    labelMeasure.title.textContent = title
    labelMeasure.kind.textContent = kind
    width = Math.max(NODE_MIN_WIDTH, Math.ceil(labelMeasure.title.offsetWidth + labelMeasure.kind.offsetWidth) + TITLE_WIDTH_SLACK)
    labelWidths.set(key, width)
  }
  return width
}

/** Room a parent keeps around its children, and above them for its own title (as the C4 service's layout does). */
const FRAME_PADDING = 32
const FRAME_TITLE = 64
/** Space between siblings that had to be pushed apart. */
const SIBLING_GAP = 24
/** Used for a node without a stored size; the surface's default card. */
export const FALLBACK_SIZE = { width: 224, height: 96 }

/**
 * Makes every node fit what it shows, once, before anything is drawn: each is
 * at least `minWidth` wide, overlapping siblings are pushed right (with
 * everything inside them), and each parent grows around its children. Done
 * innermost first, so a child that grows pushes its siblings and its parent
 * grows to hold them, all the way up. Nothing ever shrinks, and a layout that
 * already fits comes back unchanged.
 */
export function fitGraph(nodes: DiagramNode[], minWidth: (node: DiagramNode) => number): DiagramNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const children = new Map<string | null, string[]>()
  for (const node of nodes) {
    const parent = node.parent && byId.has(node.parent) ? node.parent : null
    children.set(parent, [...children.get(parent) ?? [], node.id])
  }
  const rect = (id: string) => {
    const node = byId.get(id)!
    return { x: node.position.x, y: node.position.y, width: node.width ?? FALLBACK_SIZE.width, height: node.height ?? FALLBACK_SIZE.height }
  }
  const shift = (id: string, dx: number) => {
    const node = byId.get(id)!
    byId.set(id, { ...node, position: { x: node.position.x + dx, y: node.position.y } })
    for (const child of children.get(id) ?? []) shift(child, dx)
  }
  const fit = (parent: string | null) => {
    const ids = children.get(parent) ?? []
    for (const id of ids) fit(id)
    // Left to right; a push can only move a node further right, so repeat until nothing overlaps.
    for (let moved = true; moved;) {
      moved = false
      const ordered = [...ids].sort((a, b) => rect(a).x - rect(b).x || rect(a).y - rect(b).y)
      for (let j = 1; j < ordered.length; j++) {
        for (let i = 0; i < j; i++) {
          const a = rect(ordered[i]!)
          const b = rect(ordered[j]!)
          if (b.x >= a.x + a.width || b.y >= a.y + a.height || a.y >= b.y + b.height) continue
          shift(ordered[j]!, a.x + a.width + SIBLING_GAP - b.x)
          moved = true
        }
      }
    }
    if (parent === null) return
    const node = byId.get(parent)!
    const own = rect(parent)
    const inner = ids.map(rect)
    const x = Math.min(own.x, ...inner.map((child) => child.x - FRAME_PADDING))
    const y = Math.min(own.y, ...inner.map((child) => child.y - FRAME_TITLE))
    const right = Math.max(own.x + Math.max(own.width, minWidth(node)), ...inner.map((child) => child.x + child.width + FRAME_PADDING))
    const bottom = Math.max(own.y + own.height, ...inner.map((child) => child.y + child.height + FRAME_PADDING))
    if (x !== own.x || y !== own.y || right - x !== own.width || bottom - y !== own.height) byId.set(parent, { ...node, position: { x, y }, width: right - x, height: bottom - y })
  }
  for (const node of nodes) {
    const width = minWidth(node)
    if ((children.get(node.id) ?? []).length === 0 && width > rect(node.id).width) byId.set(node.id, { ...node, width })
  }
  fit(null)
  // Untouched nodes keep their identity, so nothing downstream sees a change that did not happen.
  return nodes.map((node) => {
    const next = byId.get(node.id)!
    const same = next.position.x === node.position.x && next.position.y === node.position.y && next.width === node.width && next.height === node.height
    return same ? node : next
  })
}

export interface GraphChange {
  /** Model, view, and layout operations, in the order to apply them. */
  ops: ArchitectureOperation[]
  /** Untitled drawn nodes: kept locally until a title makes them an element. */
  drafts: DiagramNode[]
  /** A change the model cannot express; shown instead of applied. */
  rejected: string | null
}

const rectOf = (node: DiagramNode): NodePin => ({
  x: node.position.x, y: node.position.y,
  ...(node.width !== undefined ? { width: node.width } : {}),
  ...(node.height !== undefined ? { height: node.height } : {}),
})

/**
 * Turns one surface edit into operations. A delete names only the nodes and
 * connections the surface removed, and the C4 service takes every other use of
 * them along. A node drawn outside every element goes into the view's scope;
 * the server knows which.
 */
export function diffGraph(before: DiagramGraph, after: DiagramGraph, view: string, kinds: readonly string[]): GraphChange {
  const ops: ArchitectureOperation[] = []
  const layout: Extract<ArchitectureOperation, { op: "layout" }> = { op: "layout", view, nodes: {}, edges: {} }
  const drafts: DiagramNode[] = []
  const previous = new Map(before.nodes.map((node) => [node.id, node]))
  const next = new Set(after.nodes.map((node) => node.id))
  const refuse = (reason: string): GraphChange => ({ ops: [], drafts: before.nodes.filter((node) => isDraft(node.id)), rejected: reason })

  for (const node of after.nodes) {
    const title = node.title.trim()
    if (isDraft(node.id)) {
      if (!title) { drafts.push(node); continue }
      ops.push({
        // Drawn nodes are components unless chosen otherwise; systems and containers come from the tree.
        op: "addElement", parent: node.parent, kind: node.kind || (kinds.includes("component") ? "component" : kinds[0] ?? "component"), title,
        ...(node.bullets.length > 0 ? { description: node.bullets.join("\n") } : {}),
        layout: { view, ...rectOf(node) },
      })
      continue
    }
    const old = previous.get(node.id)
    if (!old) continue
    if (title && title !== old.title) ops.push({ op: "setTitle", element: node.id, title })
    if (node.bullets.join("\n") !== old.bullets.join("\n")) ops.push({ op: "setDescription", element: node.id, description: node.bullets.join("\n") })
    if (node.kind !== old.kind) ops.push({ op: "setKind", element: node.id, kind: node.kind })
    const moved = node.position.x !== old.position.x || node.position.y !== old.position.y || node.width !== old.width || node.height !== old.height
    if (node.parent !== old.parent) {
      ops.push({ op: "reparent", element: node.id, parent: node.parent })
      // Reparenting keeps the name, so the new FQN is known before the server answers.
      const name = node.id.slice(node.id.lastIndexOf(".") + 1)
      layout.nodes[node.parent ? `${node.parent}.${name}` : name] = rectOf(node)
    } else if (moved) layout.nodes[node.id] = rectOf(node)
  }

  const removedElements = before.nodes.filter((node) => !next.has(node.id) && !isDraft(node.id)).map((node) => node.id)
  const previousEdges = new Map(before.edges.map((edge) => [edge.id, edge]))
  const nextEdges = new Set(after.edges.map((edge) => edge.id))
  const removedRelations = before.edges.filter((edge) => !nextEdges.has(edge.id)).flatMap((edge) => edge.relations)

  for (const edge of after.edges) {
    const old = previousEdges.get(edge.id)
    if (!old) {
      if (isDraft(edge.source) || isDraft(edge.target)) return refuse("Name a new node before connecting it")
      ops.push({ op: "addRelation", source: edge.source, target: edge.target, ...(edge.label ? { label: edge.label } : {}) })
      continue
    }
    if ((edge.label ?? "") !== (old.label ?? "")) {
      if (edge.relations.length !== 1) return refuse(`This connection merges ${edge.relations.length} relationships, edit their labels in Source`)
      ops.push({ op: "setLabel", relation: edge.relations[0], label: edge.label ?? "" })
    }
    if (edge.path?.bend !== old.path?.bend) layout.edges[edge.id] = edge.path ?? null
  }
  if (removedElements.length > 0 || removedRelations.length > 0) {
    ops.push({ op: "delete", elements: removedElements, relations: removedRelations })
  }
  if (Object.keys(layout.nodes).length > 0 || Object.keys(layout.edges).length > 0) ops.push(layout)
  return { ops, drafts, rejected: null }
}
