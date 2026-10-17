// Wire types for /api/architecture: the LikeC4 workspace the C4 service returns
// and the operations it accepts. The `.c4` sources are the truth; the surface
// only proposes changes, which `diffGraph` turns into these operations (model
// and view edits, plus `layout` operations for positions, saved as LikeC4 manual layouts).

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

interface ArchitectureViewEdge {
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

interface ArchitectureModel {
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

