// Planners that add elements and connections or edit their properties in place.
import { bodyProperty, type ElementNode } from '../ast.ts'
import { cst, propertyNode, type PlanContext } from './context.ts'
import { idFromTitle, uniqueId } from './ids.ts'
import { OperationError, type Op, type Plan } from './operation.ts'
import { patchPins } from './pins.ts'
import { descriptionValue, indentAt, INDENT_UNIT, insertAt, quote, requireTitle, singleLine, type TextEdit } from './text.ts'

/** The element a view is scoped to (`view x of shop`), null for landscape views. */
async function scopeOf(c: PlanContext, view: string): Promise<string | null> {
  const found = (await c.ws.model()).findView(view)
  return found?.isElementView() ? found.viewOf?.id ?? null : null
}

export async function addElement(c: PlanContext, op: Op<'addElement'>): Promise<Plan> {
  const title = requireTitle(op.title, 'An element needs a title')
  if (!(await c.kinds()).has(op.kind)) throw new OperationError(`Unknown element kind: ${op.kind}`)
  // Drawn on the open canvas of `view x of shop`, an element belongs to shop.
  const parent = op.parent ?? (op.layout ? await scopeOf(c, op.layout.view) : null)
  if (parent !== null) c.element(parent)
  const id = uniqueId(idFromTitle(title), await c.childNames(parent))
  const created = parent ? `${parent}.${id}` : id
  const description = op.description?.trim()
  const declaration = `${id} = ${op.kind} ${quote(title)}`
  const lines = description
    ? [`${declaration} {`, ...`${INDENT_UNIT}description ${descriptionValue(description, INDENT_UNIT)}`.split('\n'), '}']
    : [declaration]
  const edits = [c.appendChildren(parent, lines)]
  if (!op.layout) return { edits, created }
  // Drawn at a position: the new element is saved there.
  const { view, ...pin } = op.layout
  return { edits, created, repin: (pins) => patchPins(pins, { op: 'layout', view, nodes: { [created]: pin }, edges: {} }) }
}

export async function addRelation(c: PlanContext, op: Op<'addRelation'>): Promise<Plan> {
  c.element(op.source)
  c.element(op.target)
  if (op.source === op.target) throw new OperationError('A connection needs two different elements')
  const label = op.label?.trim()
  const line = `${op.source} -> ${op.target}${label ? ` ${quote(label)}` : ''}`
  // A connection is stored once, beside its source element; every view showing both ends draws it.
  const file = c.ws.documentPath(c.element(op.source))
  return { edits: [c.appendToModel(file, [line])] }
}

export function setTitle(c: PlanContext, op: Op<'setTitle'>): Plan {
  const title = requireTitle(op.title, 'An element needs a title')
  const element = c.element(op.element)
  const file = c.ws.documentPath(element)
  const bodyTitle = bodyProperty(element, 'title')?.value
  if (bodyTitle) return { edits: [c.replace(cst(bodyTitle), file, quote(title))] }
  const inline = propertyNode(element, 'props', 0)
  if (inline) return { edits: [c.replace(inline, file, quote(title))] }
  const kind = propertyNode(element, 'kind')
  if (!kind) throw new OperationError('Unexpected element shape in source')
  return { edits: [insertAt(file, kind.end, ` ${quote(title)}`)] }
}

export function setDescription(c: PlanContext, op: Op<'setDescription'>): Plan {
  const description = op.description.trim()
  const element = c.element(op.element)
  const file = c.ws.documentPath(element)
  const text = c.text(file)
  // The second inline string (`kind 'Title' 'Summary'`) is the summary, never the description.
  const existing = bodyProperty(element, 'description')
  if (existing) {
    const edit = description
      ? c.replace(cst(existing.value!), file, descriptionValue(description, indentAt(text, cst(existing).offset)))
      : c.remove(file, cst(existing).offset, cst(existing).end)
    return { edits: [edit] }
  }
  if (!description) return { edits: [] }
  const indent = indentAt(text, cst(element).offset) + INDENT_UNIT
  return { edits: [insertBodyProperty(c, element, `description ${descriptionValue(description, indent)}`)] }
}

/** Properties precede nested elements in a body: insert after the last property or tag line, else first. */
function insertBodyProperty(c: PlanContext, element: ElementNode, line: string): TextEdit {
  const body = element.body
  if (!body) return c.openBody(element, [line])
  const file = c.ws.documentPath(element)
  const text = c.text(file)
  const anchor = body.props?.at(-1) ?? body.tags
  const inner = indentAt(text, cst(element).offset) + INDENT_UNIT
  if (anchor) {
    const anchorEnd = cst(anchor).end
    const lineEnd = text.indexOf('\n', anchorEnd)
    const at = lineEnd === -1 ? text.length : lineEnd
    return insertAt(file, at, `\n${inner}${line}`)
  }
  const open = cst(body).offset + 1
  return insertAt(file, open, `\n${inner}${line}`)
}

export async function setKind(c: PlanContext, op: Op<'setKind'>): Promise<Plan> {
  if (!(await c.kinds()).has(op.kind)) throw new OperationError(`Unknown element kind: ${op.kind}`)
  const element = c.element(op.element)
  const kind = propertyNode(element, 'kind')
  if (!kind) throw new OperationError('Unexpected element shape in source')
  return { edits: [c.replace(kind, c.ws.documentPath(element), op.kind)] }
}

export function setLabel(c: PlanContext, op: Op<'setLabel'>): Plan {
  const relation = c.relation(op.relation)
  const file = c.ws.documentPath(relation)
  const label = singleLine(op.label)
  const title = propertyNode(relation, 'title')
  if (title) {
    const edit = label ? c.replace(title, file, quote(label)) : { file, start: propertyNode(relation, 'target')!.end, end: title.end, text: '' }
    return { edits: [edit] }
  }
  if (!label) return { edits: [] }
  const target = propertyNode(relation, 'target')
  if (!target) throw new OperationError('Unexpected connection shape in source')
  return { edits: [insertAt(file, target.end, ` ${quote(label)}`)] }
}
