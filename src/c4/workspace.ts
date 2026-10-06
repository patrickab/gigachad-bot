// Parses a LikeC4 workspace held in memory and projects it into the plain JSON
// the backend and UI consume. Also exposes the syntax-tree lookups that
// operations.ts needs to edit source text surgically.
import type { LikeC4Model } from '@likec4/core/model'
import type { LayoutedElementView } from '@likec4/core/types'
import { fromSources, type LikeC4 } from '@likec4/language-services/node'
import { AstUtils, CstUtils, type AstNode, type LangiumDocument } from 'langium'
import { edgeKey, pinView, snapshotPath, snapshotText, type Pins, type Snapshots, type ViewPins } from './layout.ts'

export type Sources = Record<string, string>

export interface WorkspaceError {
  message: string
  file: string
  /** Zero-based, as LikeC4 reports it; shown to people as `line + 1`. */
  line: number
}

export interface ElementPayload {
  id: string
  name: string
  kind: string
  title: string
  description: string
  parent: string | null
  /** The source file declaring it (an `extend` elsewhere does not count). */
  file: string | null
}

export interface NodePayload {
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

export interface EdgePayload {
  /** `source->target`: stable across parses, unlike LikeC4's hashed edge ids. */
  id: string
  source: string
  target: string
  label: string | null
  relations: string[]
  /** Saved curve offset, null when the connection is drawn straight. */
  bend: number | null
}

export interface ViewPayload {
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
export interface TreeEntry {
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

export interface Rendered {
  model: ModelPayload
  /** A snapshot file for every pinned view that still exists. */
  snapshots: Snapshots
}

const WORKSPACE_URI_PREFIX = '/workspace/'

/** Predicates an editable view may use: `*` and element references, optionally with `.*`, `.**`, `._`. */
const EDITABLE_EXPRESSIONS = new Set(['WildcardExpression', 'FqnRefExpr'])

type ViewNode = AstNode & { name?: string, viewOf?: unknown, extends?: unknown, body?: { rules?: AstNode[] } }
type PredicateNode = AstNode & { isInclude?: boolean, exprs?: ExpressionsNode }
export type ExpressionsNode = AstNode & { value?: AstNode, prev?: ExpressionsNode }

/** Entries of a predicate's comma list, in source order. */
export function expressionEntries(exprs: ExpressionsNode | undefined): AstNode[] {
  const entries: AstNode[] = []
  for (let cursor = exprs; cursor; cursor = cursor.prev) if (cursor.value) entries.unshift(cursor.value)
  return entries
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

const markdownText = (value: unknown): string => {
  if (!value || typeof value !== 'object') return typeof value === 'string' ? value : ''
  const record = value as { txt?: string, md?: string }
  return record.txt ?? record.md ?? ''
}

export const parentOf = (fqn: string): string | null => {
  const index = fqn.lastIndexOf('.')
  return index === -1 ? null : fqn.slice(0, index)
}

export const nameOf = (fqn: string) => fqn.slice(fqn.lastIndexOf('.') + 1)

export class Workspace {
  readonly sources: Sources
  readonly errors: WorkspaceError[]
  private readonly lc: LikeC4
  private readonly elementNodes = new Map<string, AstNode>()
  private readonly fqnByNode = new Map<AstNode, string>()
  private readonly viewNodes = new Map<string, ViewNode>()
  private computed: LikeC4Model.Computed | null = null

  private constructor(lc: LikeC4, sources: Sources) {
    this.lc = lc
    this.sources = sources
    this.errors = lc.getErrors().map((error) => ({
      message: error.message,
      file: this.relativePath(error.sourceFsPath),
      line: error.line,
    }))
  }

  static async open(sources: Sources): Promise<Workspace> {
    const lc = await fromSources(sources, { printErrors: false })
    const workspace = new Workspace(lc, sources)
    await workspace.index()
    return workspace
  }

  async dispose(): Promise<void> {
    await this.lc.dispose()
  }

  private relativePath(fsPathOrUri: string): string {
    const index = fsPathOrUri.indexOf(WORKSPACE_URI_PREFIX)
    return index === -1 ? fsPathOrUri : fsPathOrUri.slice(index + WORKSPACE_URI_PREFIX.length)
  }

  private get langium() {
    // `langium` is TypeScript-protected on LikeC4, not runtime-private; the
    // public API stops at the model, but surgical edits need the syntax tree.
    return (this.lc as unknown as { langium: { shared: { workspace: { LangiumDocuments: { all: { toArray(): LangiumDocument[] } } } } } }).langium
  }

  documents(): Array<{ path: string, document: LangiumDocument }> {
    return this.langium.shared.workspace.LangiumDocuments.all.toArray()
      .filter((document) => document.uri.path.startsWith(WORKSPACE_URI_PREFIX))
      .map((document) => ({ path: this.relativePath(document.uri.path), document }))
      .sort((a, b) => a.path.localeCompare(b.path))
  }

  documentPath(node: AstNode): string {
    return this.relativePath(AstUtils.getDocument(node).uri.path)
  }

  async model() {
    this.computed ??= await this.lc.computedModel()
    return this.computed
  }

  private nodeAt(location: { uri: string, range: { start: { line: number, character: number } } }): AstNode | undefined {
    const path = this.relativePath(location.uri)
    const entry = this.documents().find((candidate) => candidate.path === path)
    if (!entry) return undefined
    const offset = entry.document.textDocument.offsetAt(location.range.start)
    const root = entry.document.parseResult.value.$cstNode
    return root ? CstUtils.findLeafNodeAtOffset(root, offset)?.astNode : undefined
  }

  private async index(): Promise<void> {
    for (const { document } of this.documents()) {
      for (const node of AstUtils.streamAst(document.parseResult.value)) {
        const view = node as ViewNode
        if (node.$type === 'ElementView' && view.name && !this.viewNodes.has(view.name)) this.viewNodes.set(view.name, view)
      }
    }
    const model = await this.model()
    for (const element of model.elements()) {
      const location = this.lc.languageServices.locate({ element: element.id })
      let node = location ? this.nodeAt(location) : undefined
      while (node && node.$type !== 'Element') node = node.$container
      if (!node) continue
      this.elementNodes.set(element.id, node)
      this.fqnByNode.set(node, element.id)
    }
  }

  elementNode(fqn: string): AstNode | undefined {
    return this.elementNodes.get(fqn)
  }

  fqnOf(node: AstNode | undefined): string | undefined {
    return node ? this.fqnByNode.get(node) : undefined
  }

  relationNode(id: string): AstNode | undefined {
    const location = this.lc.languageServices.locate({ relation: id as never })
    let node = location ? this.nodeAt(location) : undefined
    while (node && node.$type !== 'Relation') node = node.$container
    return node
  }

  async elementIds(): Promise<string[]> {
    return [...(await this.model()).elements()].map((element) => element.id)
  }

  /** Elements `file` declares under no ancestor it also declares, in model order. */
  rootsOf(file: string): string[] {
    const declared = [...this.elementNodes].filter(([, node]) => this.documentPath(node) === file).map(([fqn]) => fqn)
    return declared.filter((fqn) => !declared.some((other) => fqn.startsWith(`${other}.`)))
  }

  /** The `view <id> { ... }` declaration, for element views written in source. */
  viewNode(id: string): ViewNode | undefined {
    return this.viewNodes.get(id)
  }

  viewIds(): Set<string> {
    return new Set(this.viewNodes.keys())
  }

  /** Element ids a view currently draws, from the computed (not laid-out) model. */
  async viewElements(id: string): Promise<Set<string>> {
    const view = (await this.model()).findView(id)
    return new Set(view ? [...view.nodes()].map((node) => node.id as string) : [])
  }

  /** True for element views built only from element includes and excludes (plus autoLayout). */
  isEditable(id: string): boolean {
    const view = this.viewNodes.get(id)
    if (!view || view.extends) return false
    return (view.body?.rules ?? []).every((rule) => {
      if (rule.$type === 'ViewRuleAutoLayout') return true
      if (rule.$type !== 'ViewRulePredicate') return false
      return expressionEntries((rule as PredicateNode).exprs).every((entry) => EDITABLE_EXPRESSIONS.has(entry.$type))
    })
  }

  /** The model as the UI draws it, with `pins` applied, plus the snapshot file of every pinned view. */
  async render(pins: Pins): Promise<Rendered> {
    const layouted = await this.lc.layoutedModel()
    const kinds = Object.keys(layouted.specification.elements).sort()
    const elements: ElementPayload[] = [...layouted.elements()].map((element) => {
      const node = this.elementNodes.get(element.id)
      return {
        id: element.id,
        name: nameOf(element.id),
        kind: element.kind,
        title: element.title,
        description: markdownText(element.$element.description),
        parent: parentOf(element.id),
        file: node ? this.documentPath(node) : null,
      }
    })
    const views: ViewPayload[] = []
    const snapshots: Snapshots = {}
    for (const view of layouted.views()) {
      const diagram = view.$view
      if (diagram._type !== 'element') continue
      const pinned = pinView(diagram, pins[view.id])
      const node = this.viewNodes.get(view.id)
      const title = diagram.title ?? (diagram.viewOf ? layouted.element(diagram.viewOf).title : diagram.id)
      views.push(viewPayload(pinned, title, node ? this.documentPath(node) : null, this.isEditable(view.id), pins[view.id]))
      if (pins[view.id]) snapshots[snapshotPath(view.id)] = snapshotText(pinned)
    }
    return { model: { errors: this.errors, kinds, elements, views, tree: this.tree(views) }, snapshots }
  }

  private tree(views: ViewPayload[]): TreeEntry[] {
    // Views in source order, so "a file's view" is the first it declares.
    const ordered = [...this.viewNodes.entries()].map(([id, node]) => ({ id, file: this.documentPath(node), scope: views.find((view) => view.id === id)?.scope ?? null }))
    return Object.keys(this.sources).sort().map((path): TreeEntry => {
      const roots = this.rootsOf(path)
      const own = ordered.find((view) => view.file === path)?.id ?? null
      const root = roots.length === 1 ? roots[0]! : null
      if (root === null) return { path, role: roots.length === 0 && own ? 'view' : 'file', element: null, view: own }
      const scoped = ordered.find((view) => view.scope === root.split('.')[0])?.id ?? null
      return parentOf(root) === null
        ? { path, role: 'system', element: root, view: own ?? scoped }
        : { path, role: 'module', element: root, view: scoped ?? own }
    })
  }
}
