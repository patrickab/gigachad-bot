// Structured edits from the UI, applied as minimal text edits at syntax-tree
// positions. Everything outside the touched spans stays byte-identical, so
// hand-written or AI-written formatting and comments survive.
import { AstUtils, GrammarUtils, type AstNode, type CstNode } from 'langium'
import { edgeKey, type NodePin, type Pins, type Snapshots, type ViewPins } from './layout.ts'
import { expressionEntries, nameOf, parentOf, Workspace, type ExpressionsNode, type ModelPayload, type Sources } from './workspace.ts'

export type Operation =
  | { op: 'createPackage', title: string }
  | { op: 'createModule', package: string, title: string }
  | { op: 'createView', title: string }
  | { op: 'addFileView' }
  | { op: 'addElement', parent: string | null, kind: string, title: string, description?: string, layout?: { view: string } & NodePin }
  | { op: 'addRelation', source: string, target: string, label?: string }
  | { op: 'delete', elements: string[], relations: string[] }
  | { op: 'setTitle', element: string, title: string }
  | { op: 'setDescription', element: string, description: string }
  | { op: 'setKind', element: string, kind: string }
  | { op: 'setLabel', relation: string, label: string }
  | { op: 'rename', element: string, id: string }
  | { op: 'reparent', element: string, parent: string | null }
  | { op: 'includeInView', view: string, elements: string[], descendants?: boolean }
  | { op: 'removeFromView', view: string, elements: string[] }
  | { op: 'layout', view: string, nodes: Record<string, NodePin | null>, edges: Record<string, { bend: number } | null> }

export interface ApplyResult {
  sources: Sources
  /** Every snapshot file the workspace should now hold; any other `.likec4/*.snap` is stale. */
  snapshots: Snapshots
  model: ModelPayload
  /**
   * Per operation: the FQN an `addElement` created, the path a `createPackage`,
   * `createModule` or `createView` created, the view id an `addFileView` declared, otherwise null.
   */
  created: Array<string | null>
}

/** A request the current model cannot honour; the message is shown to the user. */
export class OperationError extends Error {}

interface TextEdit {
  file: string
  start: number
  end: number
  text: string
}

interface Plan {
  edits: TextEdit[]
  renames?: Record<string, string>
  created?: string
}

const INDENT_UNIT = '  '
/** Saved views live one per file under this folder; the file name is the view id. */
const VIEWS_DIR = 'views/'

/** Element kinds a workspace starts with, written into its first package. */
const DEFAULT_SPECIFICATION = `specification {
  element actor {
    style {
      shape person
    }
  }
  element system
  element container
  element component
  element database {
    style {
      shape cylinder
    }
  }
}

`
const ID_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/

const quote = (value: string) => `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`

/** lowerCamelCase ASCII id from a title: "Payment Service" -> "paymentService". */
export function idFromTitle(title: string): string {
  const words = title.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').split(/[^A-Za-z0-9]+/).filter(Boolean)
  const id = words.map((word, index) => {
    const lower = word.toLowerCase()
    return index === 0 ? lower : lower.charAt(0).toUpperCase() + lower.slice(1)
  }).join('')
  if (!id) return 'element'
  return /^[0-9]/.test(id) ? `element${id.charAt(0).toUpperCase()}${id.slice(1)}` : id
}

/** "payment-service" or "paymentService" -> "Payment Service". */
export function titleFromName(name: string): string {
  const words = name.replace(/([a-z0-9])([A-Z])/g, '$1 $2').split(/[^A-Za-z0-9]+/).filter(Boolean)
  return words.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') || 'Architecture'
}

function uniqueId(base: string, taken: ReadonlySet<string>): string {
  if (!taken.has(base)) return base
  let suffix = 2
  while (taken.has(`${base}${suffix}`)) suffix += 1
  return `${base}${suffix}`
}

const lineStart = (text: string, offset: number) => text.lastIndexOf('\n', offset - 1) + 1
const indentAt = (text: string, offset: number) => /^[ \t]*/.exec(text.slice(lineStart(text, offset)))![0]
/** `lines` (relative indentation, "" for blank) joined, each prefixed with `indent`. */
const indented = (lines: string[], indent: string) => lines.map((line) => (line ? indent + line : line)).join('\n')

/** Widens a span to its full lines when nothing else shares them, so deletions leave no blank line. */
function wholeLines(text: string, start: number, end: number): [number, number] {
  const from = lineStart(text, start)
  let to = text.indexOf('\n', end)
  if (to === -1) to = text.length
  if (text.slice(from, start).trim() || text.slice(end, to).trim()) return [start, end]
  return [from, to < text.length ? to + 1 : to]
}

const cst = (node: AstNode): CstNode => {
  if (!node.$cstNode) throw new OperationError('Source position unavailable')
  return node.$cstNode
}

// findNodeForProperty's index does not select among repeated values, so pick from the full list.
const propertyNode = (node: AstNode, property: string, index = 0): CstNode | undefined =>
  GrammarUtils.findNodesForProperty(cst(node), property)[index]

/** LikeC4 derives view titles from element titles and rejects newlines in them. */
const singleLine = (value: string) => value.replace(/\s+/g, ' ').trim()

/** One line stays a quoted string; several become a dedented ''' block, one bullet per line. */
function descriptionValue(description: string, indent: string): string {
  const lines = description.split('\n').map((line) => line.trim()).filter(Boolean)
  if (lines.length <= 1) return quote(lines[0] ?? '')
  const body = lines.map((line) => `${indent}${INDENT_UNIT}${line.replaceAll("'''", "''")}`).join('\n')
  return `'''\n${body}\n${indent}'''`
}

const isWithin = (fqn: string, ancestor: string) => fqn === ancestor || fqn.startsWith(`${ancestor}.`)

function applyEdits(sources: Sources, edits: TextEdit[]): Sources {
  const next: Sources = { ...sources }
  const byFile = new Map<string, TextEdit[]>()
  for (const edit of edits) byFile.set(edit.file, [...(byFile.get(edit.file) ?? []), edit])
  for (const [file, fileEdits] of byFile) {
    // Later spans first, so earlier offsets stay valid; ties keep request order.
    const ordered = fileEdits.map((edit, index) => ({ edit, index }))
      .sort((a, b) => b.edit.start - a.edit.start || b.index - a.index)
    let text = next[file] ?? ''
    let floor = Infinity
    for (const { edit } of ordered) {
      if (edit.end > floor) throw new OperationError('Internal error: overlapping edits')
      text = text.slice(0, edit.start) + edit.text + text.slice(edit.end)
      floor = edit.start
    }
    next[file] = text
  }
  return next
}

class Planner {
  private readonly ws: Workspace
  /** The file a window edits: new top-level elements go there; null when the request names none. */
  private readonly home: string | null

  constructor(ws: Workspace, home: string | null) {
    this.ws = ws
    this.home = home
  }

  private homeFile(): string {
    if (this.home === null || !(this.home in this.ws.sources)) throw new OperationError('Open a file to draw into')
    return this.home
  }

  private text(file: string) {
    return this.ws.sources[file] ?? ''
  }

  private element(fqn: string): AstNode {
    const node = this.ws.elementNode(fqn)
    if (!node) throw new OperationError(`Element not found: ${fqn}`)
    return node
  }

  private relation(id: string): AstNode {
    const node = this.ws.relationNode(id)
    if (!node) throw new OperationError(`Connection not found: ${id}`)
    return node
  }

  private async kinds(): Promise<Set<string>> {
    return new Set(Object.keys((await this.ws.model()).specification.elements))
  }

  private async childNames(parent: string | null): Promise<Set<string>> {
    const ids = await this.ws.elementIds()
    return new Set(ids.filter((id) => parentOf(id) === parent).map(nameOf))
  }

  /** The first `model { }` block of `file`. */
  private modelBlock(file: string): AstNode | null {
    const entry = this.ws.documents().find(({ path }) => path === file)
    return (entry?.document.parseResult.value as AstNode & { models?: AstNode[] } | undefined)?.models?.[0] ?? null
  }

  /** Appends `lines` to `file`'s first `model { }`, creating one at the end of the file when it has none. */
  private appendToModel(file: string, lines: string[]): TextEdit {
    const block = this.modelBlock(file)
    if (block) return this.appendToBlock(block, lines)
    const text = this.text(file)
    const prefix = text && !text.endsWith('\n') ? '\n' : ''
    return { file, start: text.length, end: text.length, text: `${prefix}model {\n${indented(lines, INDENT_UNIT)}\n}\n` }
  }

  /**
   * Inserts `lines` (relative indentation, "" for blank) as the last children
   * of `parent`, creating the element's `{ }` body when it has none yet. Top-level
   * elements go into the home file's model block.
   */
  private appendChildren(parent: string | null, lines: string[]): TextEdit {
    if (parent === null) return this.appendToModel(this.homeFile(), lines)
    const element = this.element(parent)
    const body = (element as AstNode & { body?: AstNode }).body
    return body ? this.appendToBlock(body, lines) : this.openBody(element, lines)
  }

  /** Gives an element without a body a `{ }` one holding `lines`. */
  private openBody(element: AstNode, lines: string[]): TextEdit {
    const file = this.ws.documentPath(element)
    const indent = indentAt(this.text(file), cst(element).offset)
    const end = cst(element).end
    return { file, start: end, end, text: ` {\n${indented(lines, indent + INDENT_UNIT)}\n${indent}}` }
  }

  private appendToBlock(block: AstNode, lines: string[]): TextEdit {
    const file = this.ws.documentPath(block)
    const text = this.text(file)
    const node = cst(block)
    const close = node.end - 1
    if (text[close] !== '}') throw new OperationError('Unexpected block shape in source')
    const owner = indentAt(text, node.offset)
    const content = indented(lines, owner + INDENT_UNIT)
    const closeLine = lineStart(text, close)
    if (!text.slice(closeLine, close).trim()) return { file, start: closeLine, end: closeLine, text: `${content}\n` }
    return { file, start: close, end: close, text: `\n${content}\n${owner}` }
  }

  private replace(node: CstNode, file: string, text: string): TextEdit {
    return { file, start: node.offset, end: node.end, text }
  }

  private remove(file: string, start: number, end: number): TextEdit {
    const [from, to] = wholeLines(this.text(file), start, end)
    return { file, start: from, end: to, text: '' }
  }

  async addElement(op: Extract<Operation, { op: 'addElement' }>): Promise<Plan> {
    const title = singleLine(op.title)
    if (!title) throw new OperationError('An element needs a title')
    if (!(await this.kinds()).has(op.kind)) throw new OperationError(`Unknown element kind: ${op.kind}`)
    // Drawn on the open canvas of `view x of shop`, an element belongs to shop.
    const parent = op.parent ?? (op.layout ? await this.scopeOf(op.layout.view) : null)
    if (parent !== null) this.element(parent)
    const id = uniqueId(idFromTitle(title), await this.childNames(parent))
    const created = parent ? `${parent}.${id}` : id
    const description = op.description?.trim()
    const declaration = `${id} = ${op.kind} ${quote(title)}`
    const lines = description
      ? [`${declaration} {`, ...`${INDENT_UNIT}description ${descriptionValue(description, INDENT_UNIT)}`.split('\n'), '}']
      : [declaration]
    return { edits: [this.appendChildren(parent, lines)], created }
  }

  private async scopeOf(view: string): Promise<string | null> {
    const found = (await this.ws.model()).findView(view)
    return found?.isElementView() ? found.viewOf?.id ?? null : null
  }

  // --- Files and views ---------------------------------------------------------

  /** The file a new package, module or view gets: `id` made unique against `taken` names and existing files. */
  private newFile(folder: string, title: string, taken: ReadonlySet<string>): { id: string, path: string } {
    const id = uniqueId(idFromTitle(title), new Set([...taken, ...Object.keys(this.ws.sources)
      .filter((path) => path.startsWith(folder) && !path.slice(folder.length).includes('/'))
      .map((path) => path.slice(folder.length, -'.c4'.length))]))
    return { id, path: `${folder}${id}.c4` }
  }

  private async kindFor(preferred: string): Promise<string> {
    const kinds = await this.kinds()
    return kinds.size === 0 || kinds.has(preferred) ? preferred : [...kinds][0]!
  }

  /** A package is a folder, `<id>/<id>.c4`, declaring one system and the view of it. */
  async createPackage(op: Extract<Operation, { op: 'createPackage' }>): Promise<Plan> {
    const title = singleLine(op.title)
    if (!title) throw new OperationError('A package needs a name')
    // Neither an element nor a folder may already use the id, so `views/` is never a package.
    const folders = Object.keys(this.ws.sources).filter((path) => path.includes('/')).map((path) => path.slice(0, path.indexOf('/')))
    const id = uniqueId(idFromTitle(title), new Set([...await this.childNames(null), ...folders, VIEWS_DIR.slice(0, -1)]))
    const path = `${id}/${id}.c4`
    const specification = (await this.kinds()).size === 0 ? DEFAULT_SPECIFICATION : ''
    const text = `${specification}model {
  ${id} = ${await this.kindFor('system')} ${quote(title)}
}

views {
  view ${uniqueId(id, this.ws.viewIds())} of ${id} {
    include *
  }
}
`
    return { edits: [{ file: path, start: 0, end: 0, text }], created: path }
  }

  /** A module is a file beside its package's, adding one container to it with `extend`. */
  async createModule(op: Extract<Operation, { op: 'createModule' }>): Promise<Plan> {
    const title = singleLine(op.title)
    if (!title) throw new OperationError('A module needs a name')
    const pkg = this.element(op.package)
    if (parentOf(op.package) !== null) throw new OperationError('Modules belong to a top-level package')
    const home = this.ws.documentPath(pkg)
    const folder = home.includes('/') ? home.slice(0, home.lastIndexOf('/') + 1) : `${op.package}/`
    const { id, path } = this.newFile(folder, title, await this.childNames(op.package))
    const text = `model {
  extend ${op.package} {
    ${id} = ${await this.kindFor('container')} ${quote(title)}
  }
}
`
    return { edits: [{ file: path, start: 0, end: 0, text }], created: path }
  }

  /** A saved view is its own file, `views/<id>.c4`, and starts empty. */
  createView(op: Extract<Operation, { op: 'createView' }>): Plan {
    const title = singleLine(op.title)
    if (!title) throw new OperationError('A view needs a name')
    const { id, path } = this.newFile(VIEWS_DIR, title, this.ws.viewIds())
    const text = `views {\n  view ${id} {\n    title ${quote(title)}\n  }\n}\n`
    return { edits: [{ file: path, start: 0, end: 0, text }], created: path }
  }

  /** Gives a file without a view one: a view of its single system, else a list of what it declares. */
  addFileView(): Plan {
    const file = this.homeFile()
    const roots = this.ws.rootsOf(file)
    const base = file.slice(file.lastIndexOf('/') + 1, -'.c4'.length)
    const single = roots.length === 1 ? roots[0]! : null
    const id = uniqueId(single ? nameOf(single) : idFromTitle(titleFromName(base)), this.ws.viewIds())
    const body = single ? [`view ${id} of ${single} {`, `${INDENT_UNIT}include *`, '}'] : [`view ${id} {`, ...(roots.length ? [`${INDENT_UNIT}include ${roots.join(', ')}`] : []), '}']
    const text = this.text(file)
    const prefix = text && !text.endsWith('\n') ? '\n\n' : text ? '\n' : ''
    const views = `${prefix}views {\n${indented(body, INDENT_UNIT)}\n}\n`
    return { edits: [{ file, start: text.length, end: text.length, text: views }], created: id }
  }

  private editableView(id: string) {
    const view = this.ws.viewNode(id)
    if (!view) throw new OperationError(`View not found: ${id}`)
    if (!this.ws.isEditable(id)) throw new OperationError('This view uses filters or styles, edit it in its source file')
    return view
  }

  /** Every `include`/`exclude` rule of a view, with the element each plain entry names. */
  private predicates(view: AstNode) {
    const rules = (view as AstNode & { body?: { rules?: AstNode[] } }).body?.rules ?? []
    return rules.filter((rule) => rule.$type === 'ViewRulePredicate').map((rule) => {
      const predicate = rule as AstNode & { isInclude?: boolean, exprs?: ExpressionsNode }
      const entries = expressionEntries(predicate.exprs).map((entry) => {
        const ref = entry as AstNode & { ref?: { value?: { ref?: AstNode } }, selector?: string }
        // Only a bare reference names exactly one element; `x.*` and friends name a family.
        const element = entry.$type === 'FqnRefExpr' && !ref.selector ? this.ws.fqnOf(ref.ref?.value?.ref) : undefined
        return { node: entry, element }
      })
      return { rule: predicate, include: !!predicate.isInclude, entries }
    })
  }

  /** Drops the given entries from their rules; a rule left empty goes entirely. */
  private dropEntries(view: AstNode, doomed: Set<AstNode>): TextEdit[] {
    const file = this.ws.documentPath(view)
    const edits: TextEdit[] = []
    for (const { rule, entries } of this.predicates(view)) {
      if (!entries.some((entry) => doomed.has(entry.node))) continue
      const kept = entries.filter((entry) => !doomed.has(entry.node))
      if (kept.length === 0) {
        edits.push(this.remove(file, cst(rule).offset, cst(rule).end))
        continue
      }
      const start = cst(entries[0]!.node).offset
      const end = cst(entries.at(-1)!.node).end
      edits.push({ file, start, end, text: kept.map((entry) => cst(entry.node).text).join(', ') })
    }
    return edits
  }

  private appendRule(view: AstNode, line: string): TextEdit {
    const body = (view as AstNode & { body?: AstNode }).body
    if (!body) throw new OperationError('Unexpected view shape in source')
    return this.appendToBlock(body, [line])
  }

  includeInView(op: Extract<Operation, { op: 'includeInView' }>): Plan {
    const view = this.editableView(op.view)
    for (const fqn of op.elements) this.element(fqn)
    if (op.elements.length === 0) return { edits: [] }
    const wanted = new Set(op.elements)
    const predicates = this.predicates(view)
    // An element the view excludes by name comes back by dropping that exclusion.
    // Whether a wider rule then draws it is checked afterwards (see followUp).
    const unexcluded = predicates.filter((rule) => !rule.include).flatMap((rule) => rule.entries)
      .filter((entry) => entry.element && wanted.has(entry.element))
    const doomed = new Set(unexcluded.map((entry) => entry.node))
    const named = new Set([
      ...predicates.filter((rule) => rule.include).flatMap((rule) => rule.entries).map((entry) => entry.element),
      ...unexcluded.map((entry) => entry.element),
    ])
    const added = op.elements.filter((fqn) => op.descendants || !named.has(fqn))
      .flatMap((fqn) => (op.descendants ? [fqn, `${fqn}.**`] : [fqn]))
    const edits = this.dropEntries(view, doomed)
    if (added.length > 0) edits.push(this.appendRule(view, `include ${added.join(', ')}`))
    return { edits }
  }

  /** Drops the element's own include; when something broader still draws it, excludes it by name. */
  removeFromView(op: Extract<Operation, { op: 'removeFromView' }>): Plan {
    const view = this.editableView(op.view)
    for (const fqn of op.elements) this.element(fqn)
    const unwanted = new Set(op.elements)
    const own = this.predicates(view).filter((rule) => rule.include).flatMap((rule) => rule.entries)
      .filter((entry) => entry.element && unwanted.has(entry.element))
    const edits = this.dropEntries(view, new Set(own.map((entry) => entry.node)))
    const named = new Set(own.map((entry) => entry.element))
    const excluded = op.elements.filter((fqn) => !named.has(fqn))
    if (excluded.length > 0) edits.push(this.appendRule(view, `exclude ${excluded.join(', ')}`))
    return { edits }
  }

  async addRelation(op: Extract<Operation, { op: 'addRelation' }>): Promise<Plan> {
    this.element(op.source)
    this.element(op.target)
    if (op.source === op.target) throw new OperationError('A connection needs two different elements')
    const label = op.label?.trim()
    const line = `${op.source} -> ${op.target}${label ? ` ${quote(label)}` : ''}`
    // A connection is stored once, beside its source element; every view showing both ends draws it.
    const file = this.ws.documentPath(this.element(op.source))
    return { edits: [this.appendToModel(file, [line])] }
  }

  delete(op: Extract<Operation, { op: 'delete' }>): Plan {
    const spans = [
      ...op.elements.map((fqn) => this.element(fqn)),
      ...op.relations.map((id) => this.relation(id)),
    ].map((node) => ({ file: this.ws.documentPath(node), start: cst(node).offset, end: cst(node).end }))
    if (spans.length === 0) throw new OperationError('Nothing selected to delete')
    // A child or a nested connection goes with its deleted ancestor's block.
    const outermost = spans.filter((span) => !spans.some((other) => other !== span && other.file === span.file
      && other.start <= span.start && other.end >= span.end && (other.start !== span.start || other.end !== span.end)))
    const unique = outermost.filter((span, index) => outermost.findIndex((other) => other.file === span.file && other.start === span.start) === index)
    return { edits: unique.map((span) => this.remove(span.file, span.start, span.end)) }
  }

  private bodyProperty(element: AstNode, key: string): (AstNode & { value?: AstNode }) | undefined {
    const props = (element as AstNode & { body?: { props?: Array<AstNode & { key?: string }> } }).body?.props ?? []
    return props.find((prop) => prop.key === key)
  }

  setTitle(op: Extract<Operation, { op: 'setTitle' }>): Plan {
    const title = singleLine(op.title)
    if (!title) throw new OperationError('An element needs a title')
    const element = this.element(op.element)
    const file = this.ws.documentPath(element)
    const bodyTitle = this.bodyProperty(element, 'title')?.value
    if (bodyTitle) return { edits: [this.replace(cst(bodyTitle), file, quote(title))] }
    const inline = propertyNode(element, 'props', 0)
    if (inline) return { edits: [this.replace(inline, file, quote(title))] }
    const kind = propertyNode(element, 'kind')
    if (!kind) throw new OperationError('Unexpected element shape in source')
    return { edits: [{ file, start: kind.end, end: kind.end, text: ` ${quote(title)}` }] }
  }

  setDescription(op: Extract<Operation, { op: 'setDescription' }>): Plan {
    const description = op.description.trim()
    const element = this.element(op.element)
    const file = this.ws.documentPath(element)
    const text = this.text(file)
    // The second inline string (`kind 'Title' 'Summary'`) is the summary, never the description.
    const existing = this.bodyProperty(element, 'description')
    if (existing) {
      const edit = description
        ? this.replace(cst(existing.value!), file, descriptionValue(description, indentAt(text, cst(existing).offset)))
        : this.remove(file, cst(existing).offset, cst(existing).end)
      return { edits: [edit] }
    }
    if (!description) return { edits: [] }
    const indent = indentAt(text, cst(element).offset) + INDENT_UNIT
    return { edits: [this.insertBodyProperty(element, `description ${descriptionValue(description, indent)}`)] }
  }

  /** Properties precede nested elements in a body: insert after the last property or tag line, else first. */
  private insertBodyProperty(element: AstNode, line: string): TextEdit {
    const body = (element as AstNode & { body?: AstNode & { props?: AstNode[], tags?: AstNode } }).body
    if (!body) return this.openBody(element, [line])
    const file = this.ws.documentPath(element)
    const text = this.text(file)
    const anchor = body.props?.at(-1) ?? body.tags
    const inner = indentAt(text, cst(element).offset) + INDENT_UNIT
    if (anchor) {
      const anchorEnd = cst(anchor).end
      const lineEnd = text.indexOf('\n', anchorEnd)
      const at = lineEnd === -1 ? text.length : lineEnd
      return { file, start: at, end: at, text: `\n${inner}${line}` }
    }
    const open = cst(body).offset + 1
    return { file, start: open, end: open, text: `\n${inner}${line}` }
  }

  async setKind(op: Extract<Operation, { op: 'setKind' }>): Promise<Plan> {
    if (!(await this.kinds()).has(op.kind)) throw new OperationError(`Unknown element kind: ${op.kind}`)
    const element = this.element(op.element)
    const kind = propertyNode(element, 'kind')
    if (!kind) throw new OperationError('Unexpected element shape in source')
    return { edits: [this.replace(kind, this.ws.documentPath(element), op.kind)] }
  }

  setLabel(op: Extract<Operation, { op: 'setLabel' }>): Plan {
    const relation = this.relation(op.relation)
    const file = this.ws.documentPath(relation)
    const label = singleLine(op.label)
    const title = propertyNode(relation, 'title')
    if (title) {
      const edit = label ? this.replace(title, file, quote(label)) : { file, start: propertyNode(relation, 'target')!.end, end: title.end, text: '' }
      return { edits: [edit] }
    }
    if (!label) return { edits: [] }
    const target = propertyNode(relation, 'target')
    if (!target) throw new OperationError('Unexpected connection shape in source')
    return { edits: [{ file, start: target.end, end: target.end, text: ` ${quote(label)}` }] }
  }

  /** Every reference in the workspace, with the element FQN it resolves to. */
  private *references(): Generator<{ holder: AstNode, cst: CstNode, target: string, file: string }> {
    for (const { path, document } of this.ws.documents()) {
      for (const node of AstUtils.streamAst(document.parseResult.value)) {
        for (const reference of AstUtils.streamReferences(node)) {
          const target = this.ws.fqnOf(reference.reference.ref)
          const refNode = reference.reference.$refNode
          if (target && refNode) yield { holder: node, cst: refNode, target, file: path }
        }
      }
    }
  }

  async rename(op: Extract<Operation, { op: 'rename' }>): Promise<Plan> {
    if (!ID_PATTERN.test(op.id)) throw new OperationError('An id may contain only letters, digits, and underscores, and must not start with a digit')
    const element = this.element(op.element)
    const parent = parentOf(op.element)
    if (op.id === nameOf(op.element)) return { edits: [] }
    if ((await this.childNames(parent)).has(op.id)) throw new OperationError(`An element named ${op.id} already exists there`)
    const name = propertyNode(element, 'name')
    if (!name) throw new OperationError('Unexpected element shape in source')
    const edits = [this.replace(name, this.ws.documentPath(element), op.id)]
    for (const reference of this.references()) {
      if (reference.target === op.element) edits.push(this.replace(reference.cst, reference.file, op.id))
    }
    return { edits, renames: { [op.element]: parent ? `${parent}.${op.id}` : op.id } }
  }

  async reparent(op: Extract<Operation, { op: 'reparent' }>): Promise<Plan> {
    const fqn = op.element
    const element = this.element(fqn)
    if (parentOf(fqn) === op.parent) return { edits: [] }
    if (op.parent !== null) {
      this.element(op.parent)
      if (isWithin(op.parent, fqn)) throw new OperationError('An element cannot move into itself')
    }
    const name = nameOf(fqn)
    if ((await this.childNames(op.parent)).has(name)) throw new OperationError(`Cannot move ${name} ${op.parent ? `into ${op.parent}` : 'to the top level'}: another element there is already named ${name}`)
    const moved = op.parent ? `${op.parent}.${name}` : name
    const mapFqn = (target: string) => moved + target.slice(fqn.length)

    const file = this.ws.documentPath(element)
    const text = this.text(file)
    const [cutStart, cutEnd] = wholeLines(text, cst(element).offset, cst(element).end)

    // Every qualified chain (`a.b.c`) reaching into the moved subtree is
    // rewritten whole as an absolute FQN, so it resolves from anywhere.
    const chains = new Map<CstNode, { file: string, text: string }>()
    for (const reference of this.references()) {
      if (!isWithin(reference.target, fqn)) continue
      let head = reference.holder
      while (head.$container && head.$container.$type === head.$type && head.$containerProperty === 'parent') head = head.$container
      const headReference = AstUtils.streamReferences(head).head()
      const headTarget = this.ws.fqnOf(headReference?.reference.ref)
      if (!headTarget || !isWithin(headTarget, fqn)) continue
      chains.set(cst(head), { file: reference.file, text: mapFqn(headTarget) })
    }
    const outside: TextEdit[] = []
    const inside: TextEdit[] = []
    for (const [node, { file: chainFile, text: replacement }] of chains) {
      const edit = { file: chainFile, start: node.offset, end: node.end, text: replacement }
      if (chainFile === file && node.offset >= cutStart && node.end <= cutEnd) inside.push({ ...edit, start: edit.start - cutStart, end: edit.end - cutStart })
      else outside.push(edit)
    }
    const block = applyEdits({ block: text.slice(cutStart, cutEnd) }, inside.map((edit) => ({ ...edit, file: 'block' }))).block
    const oldIndent = indentAt(text, cst(element).offset)
    const relative = block.replace(/\n$/, '').split('\n').map((line) => (line.startsWith(oldIndent) ? line.slice(oldIndent.length) : line))
    return { edits: [...outside, { file, start: cutStart, end: cutEnd, text: '' }, this.appendChildren(op.parent, relative)], renames: { [fqn]: moved } }
  }
}

async function plan(ws: Workspace, op: Operation, home: string | null): Promise<Plan> {
  const planner = new Planner(ws, home)
  switch (op.op) {
    case 'createPackage': return planner.createPackage(op)
    case 'createModule': return planner.createModule(op)
    case 'createView': return planner.createView(op)
    case 'addFileView': return planner.addFileView()
    case 'addElement': return planner.addElement(op)
    case 'addRelation': return planner.addRelation(op)
    case 'delete': return planner.delete(op)
    case 'setTitle': return planner.setTitle(op)
    case 'setDescription': return planner.setDescription(op)
    case 'setKind': return planner.setKind(op)
    case 'setLabel': return planner.setLabel(op)
    case 'rename': return planner.rename(op)
    case 'reparent': return planner.reparent(op)
    case 'includeInView': return planner.includeInView(op)
    case 'removeFromView': return planner.removeFromView(op)
    case 'layout': return { edits: [] }
    default: throw new OperationError(`Unknown operation: ${(op as { op?: unknown }).op}`)
  }
}

const mapPrefix = (fqn: string, renames: Record<string, string>) => {
  for (const [from, to] of Object.entries(renames)) if (isWithin(fqn, from)) return to + fqn.slice(from.length)
  return fqn
}

/** Saved positions follow renamed and moved elements. */
function renamePins(pins: Pins, renames: Record<string, string>): Pins {
  if (Object.keys(renames).length === 0) return pins
  return Object.fromEntries(Object.entries(pins).map(([viewId, view]) => [viewId, {
    nodes: Object.fromEntries(Object.entries(view.nodes).map(([fqn, pin]) => [mapPrefix(fqn, renames), pin])),
    edges: Object.fromEntries(Object.entries(view.edges).map(([key, bend]) => {
      const [source = '', target = ''] = key.split('->')
      return [edgeKey(mapPrefix(source, renames), mapPrefix(target, renames)), bend]
    })),
  }]))
}

/** A null entry unpins: the node returns to LikeC4's placement, the connection straightens. */
function patchPins(pins: Pins, op: Extract<Operation, { op: 'layout' }>): Pins {
  const view: ViewPins = { nodes: { ...pins[op.view]?.nodes }, edges: { ...pins[op.view]?.edges } }
  for (const [fqn, pin] of Object.entries(op.nodes)) {
    if (pin) view.nodes[fqn] = pin
    else delete view.nodes[fqn]
  }
  for (const [key, path] of Object.entries(op.edges)) {
    if (path && path.bend) view.edges[key] = path.bend
    else delete view.edges[key]
  }
  return { ...pins, [op.view]: view }
}

const counts = (values: string[]) => {
  const result = new Map<string, number>()
  for (const value of values) result.set(value, (result.get(value) ?? 0) + 1)
  return result
}

/** Errors the edit introduced; pre-existing breakage elsewhere doesn't block unrelated edits. */
function newErrors(before: Workspace, after: Workspace): string[] {
  const known = counts(before.errors.map((error) => error.message))
  const introduced: string[] = []
  for (const error of after.errors) {
    const left = known.get(error.message) ?? 0
    if (left > 0) known.set(error.message, left - 1)
    else introduced.push(`${error.file}:${error.line + 1}: ${error.message}`)
  }
  return introduced
}

/** A rename or move must reproduce the same model under the FQN mapping. */
async function assertEquivalent(before: Workspace, after: Workspace, renames: Record<string, string>): Promise<void> {
  const shape = async (ws: Workspace, map: (fqn: string) => string) => {
    const model = await ws.model()
    return [
      ...[...model.elements()].map((element) => `element ${map(element.id)} ${element.kind}`),
      ...[...model.relationships()].map((relation) => `relation ${map(relation.source.id)} ${map(relation.target.id)} ${relation.title ?? ''}`),
    ].sort().join('\n')
  }
  if (await shape(before, (fqn) => mapPrefix(fqn, renames)) !== await shape(after, (fqn) => fqn)) {
    throw new OperationError('Could not rewrite every reference safely; nothing was changed')
  }
}

/**
 * A follow-up that makes a gesture's visible effect match the view: a new
 * element drawn in a view the view's rules do not reach, or one whose exclusion
 * was dropped but no rule reaches, gets included by name; an element still drawn
 * after its own include went is excluded by name.
 */
async function followUp(ws: Workspace, operation: Operation, created: string | null): Promise<Operation | null> {
  if (operation.op === 'addElement' && created && operation.layout && ws.isEditable(operation.layout.view)) {
    const drawn = await ws.viewElements(operation.layout.view)
    return drawn.has(created) ? null : { op: 'includeInView', view: operation.layout.view, elements: [created] }
  }
  if (operation.op === 'includeInView') {
    const drawn = await ws.viewElements(operation.view)
    const missing = operation.elements.filter((fqn) => !drawn.has(fqn))
    return missing.length > 0 ? { op: 'includeInView', view: operation.view, elements: missing } : null
  }
  if (operation.op === 'removeFromView') {
    const drawn = await ws.viewElements(operation.view)
    const left = operation.elements.filter((fqn) => drawn.has(fqn))
    return left.length > 0 ? { op: 'removeFromView', view: operation.view, elements: left } : null
  }
  return null
}

/**
 * Applies `operations` in order, all or nothing. `home` is the file new
 * top-level elements are written into (the window's file). Saved positions
 * (`pins`) follow renames, take `layout` operations, and come back as the
 * snapshot files of every pinned view.
 */
export async function applyOperations(sources: Sources, pins: Pins, operations: Operation[], home: string | null): Promise<ApplyResult> {
  let workspace = await Workspace.open(sources)
  let nextPins = pins
  const created: Array<string | null> = []
  // Follow-ups run right after the operation they complete and report nothing themselves.
  const queue = operations.map((operation) => ({ operation, internal: false }))
  try {
    while (queue.length > 0) {
      const { operation, internal } = queue.shift()!
      const { edits, renames = {}, created: made = null } = await plan(workspace, operation, home)
      if (edits.length > 0) {
        const next = await Workspace.open(applyEdits(workspace.sources, edits)).catch((error: unknown) => {
          throw new OperationError(`LikeC4 cannot load the edited model: ${error instanceof Error ? error.message : String(error)}`)
        })
        try {
          const introduced = newErrors(workspace, next)
          if (introduced.length > 0) {
            const hint = operation.op === 'delete' ? 'Deleting would leave references to removed elements. Select their connections too, and remove them from views naming them. ' : ''
            throw new OperationError(`${hint}${introduced[0]}`)
          }
          if (Object.keys(renames).length > 0) await assertEquivalent(workspace, next, renames)
        } catch (error) {
          await next.dispose()
          throw error
        }
        await workspace.dispose()
        workspace = next
      }
      nextPins = renamePins(nextPins, renames)
      if (operation.op === 'layout') nextPins = patchPins(nextPins, operation)
      if (made && operation.op === 'addElement' && operation.layout) {
        const { view, ...pin } = operation.layout
        nextPins = patchPins(nextPins, { op: 'layout', view, nodes: { [made]: pin }, edges: {} })
      }
      if (internal) continue
      created.push(made)
      const extra = await followUp(workspace, operation, made)
      if (extra) queue.unshift({ operation: extra, internal: true })
    }
    const { model, snapshots } = await workspace.render(nextPins)
    return { sources: workspace.sources, snapshots, model, created }
  } finally {
    await workspace.dispose()
  }
}
