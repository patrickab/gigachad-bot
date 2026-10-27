// Parses a LikeC4 workspace held in memory and indexes the syntax-tree nodes
// that ops/ edits surgically; payload.ts projects it into JSON for the UI.
import type { LikeC4Model } from '@likec4/core/model'
import type { RelationId } from '@likec4/core/types'
import { fromSources, type LikeC4 } from '@likec4/language-services/node'
import { AstUtils, CstUtils, type AstNode, type LangiumDocument } from 'langium'
import { expressionEntries, isAutoLayoutRule, isEditableExpression, isElement, isPredicate, isRelation, isView, langiumDocuments, type ElementNode, type RelationNode, type ViewNode } from './ast.ts'
import { isWithin } from './fqn.ts'

export type Sources = Record<string, string>

export interface WorkspaceError {
  message: string
  file: string
  /** Zero-based, as LikeC4 reports it; shown to people as `line + 1`. */
  line: number
}

/** Sources load under this virtual folder. Stripping it gives workspace-relative paths. */
const WORKSPACE_URI_PREFIX = '/workspace/'

type SourceLocation = { uri: string, range: { start: { line: number, character: number } } }

/** One parsed workspace: LikeC4's model plus the syntax-tree nodes `ops/` edits. Dispose when done. */
export class Workspace {
  readonly sources: Sources
  readonly errors: WorkspaceError[]
  private readonly lc: LikeC4
  /** FQN -> `Element` node and back, filled by `index()`. */
  private readonly elementNodes = new Map<string, ElementNode>()
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

  documents(): Array<{ path: string, document: LangiumDocument }> {
    return langiumDocuments(this.lc)
      .filter((document) => document.uri.path.startsWith(WORKSPACE_URI_PREFIX))
      .map((document) => ({ path: this.relativePath(document.uri.path), document }))
      .sort((a, b) => a.path.localeCompare(b.path))
  }

  documentPath(node: AstNode): string {
    return this.relativePath(AstUtils.getDocument(node).uri.path)
  }

  /** Computed (not laid-out) model, memoized. */
  async model() {
    this.computed ??= await this.lc.computedModel()
    return this.computed
  }

  /** Laid-out model, for drawing. */
  layoutedModel() {
    return this.lc.layoutedModel()
  }

  /** The nearest node matching `is` at or above a `locate` result, which lands on a leaf inside it. */
  private enclosing<T extends AstNode>(location: SourceLocation | null, is: (node: AstNode) => node is T): T | undefined {
    for (let node = location ? this.nodeAt(location) : undefined; node; node = node.$container) if (is(node)) return node
    return undefined
  }

  private nodeAt(location: SourceLocation): AstNode | undefined {
    const path = this.relativePath(location.uri)
    const entry = this.documents().find((candidate) => candidate.path === path)
    if (!entry) return undefined
    const offset = entry.document.textDocument.offsetAt(location.range.start)
    const root = entry.document.parseResult.value.$cstNode
    return root ? CstUtils.findLeafNodeAtOffset(root, offset)?.astNode : undefined
  }

  /** Indexes views (first declaration of a name wins) and element FQN <-> syntax node. */
  private async index(): Promise<void> {
    for (const { document } of this.documents()) {
      for (const node of AstUtils.streamAst(document.parseResult.value)) {
        if (isView(node) && node.name && !this.viewNodes.has(node.name)) this.viewNodes.set(node.name, node)
      }
    }
    const model = await this.model()
    for (const element of model.elements()) {
      const location = this.lc.languageServices.locate({ element: element.id })
      const node = this.enclosing(location, isElement)
      if (!node) continue
      this.elementNodes.set(element.id, node)
      this.fqnByNode.set(node, element.id)
    }
  }

  elementNode(fqn: string): ElementNode | undefined {
    return this.elementNodes.get(fqn)
  }

  fqnOf(node: AstNode | undefined): string | undefined {
    return node ? this.fqnByNode.get(node) : undefined
  }

  relationNode(id: string): RelationNode | undefined {
    return this.enclosing(this.lc.languageServices.locate({ relation: id as RelationId }), isRelation)
  }

  async elementIds(): Promise<string[]> {
    return [...(await this.model()).elements()].map((element) => element.id)
  }

  /** Elements `file` declares under no ancestor it also declares, in model order. */
  rootsOf(file: string): string[] {
    const declared = [...this.elementNodes].filter(([, node]) => this.documentPath(node) === file).map(([fqn]) => fqn)
    return declared.filter((fqn) => !declared.some((other) => other !== fqn && isWithin(fqn, other)))
  }

  /** The `view <id> { ... }` declaration, for element views written in source. */
  viewNode(id: string): ViewNode | undefined {
    return this.viewNodes.get(id)
  }

  /** Ids of the element views written in source, in declaration order. */
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
      if (isAutoLayoutRule(rule)) return true
      if (!isPredicate(rule)) return false
      return expressionEntries(rule.exprs).every(isEditableExpression)
    })
  }
}
