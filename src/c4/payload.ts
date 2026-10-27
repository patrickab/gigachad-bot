// Projects a parsed workspace into the plain JSON the backend and UI consume.
import type { LayoutedElementView } from '@likec4/core/types'
import { parentOf, nameOf, topLevelOf } from './fqn.ts'
import { edgeKey, pinView, snapshotPath, snapshotText, type Pins, type Snapshots, type ViewPins } from './layout.ts'
import type { Workspace, WorkspaceError } from './workspace.ts'

interface ElementPayload {
  id: string
  name: string
  kind: string
  title: string
  description: string
  parent: string | null
  /** The source file declaring it (an `extend` elsewhere does not count). */
  file: string | null
}

interface NodePayload {
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

interface EdgePayload {
  /** `source->target`: stable across parses, unlike LikeC4's hashed edge ids. */
  id: string
  source: string
  target: string
  label: string | null
  relations: string[]
  /** Saved curve offset, null when the connection is drawn straight. */
  bend: number | null
}

interface ViewPayload {
  id: string
  title: string
  /** The source file declaring the view; null for views LikeC4 generates, like a default `index`. */
  file: string | null
  /** Element the view is scoped to (`view x of shop`), null for landscape views. */
  scope: string | null
  /** Only views made of element includes and excludes map drawing gestures onto the source unambiguously. */
  editable: boolean
  /** Whether positions are saved (a `.likec4/<id>.likec4.snap` exists); otherwise LikeC4 lays it out afresh. */
  manual: boolean
  nodes: NodePayload[]
  edges: EdgePayload[]
}

/**
 * How the UI lists a source file, from what it declares (never from its name):
 * a system declares one top-level element, a module one element nested in
 * another file's, a view file only views; anything else is a plain file.
 */
interface TreeEntry {
  path: string
  role: 'system' | 'module' | 'view' | 'file'
  /** The system's or module's element. */
  element: string | null
  /** The view a window opens for the file: a module opens its system's. */
  view: string | null
}

export interface ModelPayload {
  errors: WorkspaceError[]
  kinds: string[]
  elements: ElementPayload[]
  views: ViewPayload[]
  tree: TreeEntry[]
}

interface Rendered {
  model: ModelPayload
  /** A snapshot file for every pinned view that still exists. */
  snapshots: Snapshots
}

/** LikeC4 rich text is `{ txt }` or `{ md }`. A bare string passes through. */
const markdownText = (value: unknown): string => {
  if (!value || typeof value !== 'object') return typeof value === 'string' ? value : ''
  const record = value as { txt?: string, md?: string }
  return record.txt ?? record.md ?? ''
}

function viewPayload(view: LayoutedElementView, title: string, file: string | null, editable: boolean, pins: ViewPins | undefined): ViewPayload {
  return {
    id: view.id,
    title,
    file,
    scope: view.viewOf ?? null,
    editable,
    manual: pins !== undefined,
    nodes: view.nodes.map((node) => ({
      id: node.id,
      title: node.title,
      kind: node.kind,
      description: markdownText(node.description),
      parent: node.parent ?? null,
      compound: node.children.length > 0,
      x: node.x,
      y: node.y,
      width: node.width,
      height: node.height,
    })),
    edges: view.edges.map((edge) => {
      const id = edgeKey(edge.source, edge.target)
      return { id, source: edge.source, target: edge.target, label: edge.label ?? null, relations: [...edge.relations], bend: pins?.edges[id] ?? null }
    }),
  }
}

function tree(ws: Workspace, views: ViewPayload[]): TreeEntry[] {
  // Views in source order, so "a file's view" is the first it declares.
  const ordered = [...ws.viewIds()].map((id) => ({ id, file: ws.documentPath(ws.viewNode(id)!), scope: views.find((view) => view.id === id)?.scope ?? null }))
  return Object.keys(ws.sources).sort().map((path): TreeEntry => {
    const roots = ws.rootsOf(path)
    const own = ordered.find((view) => view.file === path)?.id ?? null
    const root = roots.length === 1 ? roots[0]! : null
    if (root === null) return { path, role: roots.length === 0 && own ? 'view' : 'file', element: null, view: own }
    const scoped = ordered.find((view) => view.scope === topLevelOf(root))?.id ?? null
    return parentOf(root) === null
      ? { path, role: 'system', element: root, view: own ?? scoped }
      : { path, role: 'module', element: root, view: scoped ?? own }
  })
}

/** The model as the UI draws it, with `pins` applied, plus the snapshot file of every pinned view. */
export async function renderModel(ws: Workspace, pins: Pins): Promise<Rendered> {
  const layouted = await ws.layoutedModel()
  const kinds = Object.keys(layouted.specification.elements).sort()
  const elements: ElementPayload[] = [...layouted.elements()].map((element) => {
    const node = ws.elementNode(element.id)
    return {
      id: element.id,
      name: nameOf(element.id),
      kind: element.kind,
      title: element.title,
      description: markdownText(element.$element.description),
      parent: parentOf(element.id),
      file: node ? ws.documentPath(node) : null,
    }
  })
  const views: ViewPayload[] = []
  const snapshots: Snapshots = {}
  for (const view of layouted.views()) {
    const diagram = view.$view
    if (diagram._type !== 'element') continue
    const pinned = pinView(diagram, pins[view.id])
    const node = ws.viewNode(view.id)
    const title = diagram.title ?? (diagram.viewOf ? layouted.element(diagram.viewOf).title : diagram.id)
    views.push(viewPayload(pinned, title, node ? ws.documentPath(node) : null, ws.isEditable(view.id), pins[view.id]))
    if (pins[view.id]) snapshots[snapshotPath(view.id)] = snapshotText(pinned)
  }
  return { model: { errors: ws.errors, kinds, elements, views, tree: tree(ws, views) }, snapshots }
}
