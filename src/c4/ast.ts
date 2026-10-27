// Narrow types for the LikeC4 syntax-tree nodes the editor reads. Every grammar
// `$type` string and property shape lives here, so a LikeC4 grammar change
// breaks in one place instead of silently in many.
import type { LikeC4 } from '@likec4/language-services/node'
import type { AstNode, LangiumDocument } from 'langium'

/** A `key value` property in an element body, like `title 'Shop'`. */
export interface PropertyNode extends AstNode {
  key?: string
  value?: AstNode
}

/** An element's `{ }` body: properties first, then tags, then nested elements. */
export interface ElementBody extends AstNode {
  props?: PropertyNode[]
  tags?: AstNode
}

export interface ElementNode extends AstNode {
  readonly $type: 'Element'
  body?: ElementBody
}

export interface RelationNode extends AstNode {
  readonly $type: 'Relation'
}

/** `extend <element> { }`, adding children to an element declared elsewhere. */
export interface ExtendNode extends AstNode {
  readonly $type: 'ExtendElement'
  element?: AstNode
}

/** `view <name> of <viewOf> { <rules> }`. */
export interface ViewNode extends AstNode {
  readonly $type: 'ElementView'
  name?: string
  extends?: unknown
  viewOf?: AstNode
  body?: AstNode & { rules?: AstNode[] }
}

/** An `include` or `exclude` rule. */
export interface PredicateNode extends AstNode {
  readonly $type: 'ViewRulePredicate'
  isInclude?: boolean
  exprs?: ExpressionsNode
}

/** A predicate's comma list, linked from its last entry backwards. */
export interface ExpressionsNode extends AstNode {
  value?: AstNode
  prev?: ExpressionsNode
}

/** An element reference in a predicate; `selector` is the `.*`, `.**` or `._` suffix. */
export interface FqnRefNode extends AstNode {
  readonly $type: 'FqnRefExpr'
  ref?: { value?: { ref?: AstNode } }
  selector?: string
}

interface DocumentRoot extends AstNode {
  models?: AstNode[]
}

const isType = <T extends AstNode>(type: T['$type']) => (node: AstNode | undefined): node is T => node?.$type === type

export const isElement = isType<ElementNode>('Element')
export const isRelation = isType<RelationNode>('Relation')
export const isExtend = isType<ExtendNode>('ExtendElement')
export const isView = isType<ViewNode>('ElementView')
export const isPredicate = isType<PredicateNode>('ViewRulePredicate')
export const isFqnRef = isType<FqnRefNode>('FqnRefExpr')
export const isAutoLayoutRule = (node: AstNode) => node.$type === 'ViewRuleAutoLayout'

/** Predicates an editable view may use: `*` and element references, optionally with `.*`, `.**`, `._`. */
export const isEditableExpression = (node: AstNode) => node.$type === 'WildcardExpression' || isFqnRef(node)

/** Entries of a predicate's comma list, in source order. */
export function expressionEntries(exprs: ExpressionsNode | undefined): AstNode[] {
  const entries: AstNode[] = []
  for (let cursor = exprs; cursor; cursor = cursor.prev) if (cursor.value) entries.unshift(cursor.value)
  return entries
}

/** The body property named `key`, like an element's `title` or `description`. */
export const bodyProperty = (element: ElementNode, key: string): PropertyNode | undefined =>
  element.body?.props?.find((prop) => prop.key === key)

/** The first `model { }` block of a document. */
export const modelBlockOf = (document: LangiumDocument): AstNode | undefined =>
  (document.parseResult.value as DocumentRoot).models?.[0]

/** Every Langium document LikeC4 loaded. */
export function langiumDocuments(lc: LikeC4): LangiumDocument[] {
  // `langium` is TypeScript-protected on LikeC4, not runtime-private; the
  // public API stops at the model, but surgical edits need the syntax tree.
  const { langium } = lc as unknown as { langium: { shared: { workspace: { LangiumDocuments: { all: { toArray(): LangiumDocument[] } } } } } }
  return langium.shared.workspace.LangiumDocuments.all.toArray()
}
