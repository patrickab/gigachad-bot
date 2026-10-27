// Planners that edit a view's `include`/`exclude` rules, and the follow-up that
// makes a gesture's visible effect match the view.
import type { AstNode } from 'langium'
import { expressionEntries, isFqnRef, isPredicate, type ViewNode } from '../ast.ts'
import type { Workspace } from '../workspace.ts'
import { cst, type PlanContext } from './context.ts'
import { OperationError, type Op, type Operation, type Plan } from './operation.ts'
import type { TextEdit } from './text.ts'

function editableView(c: PlanContext, id: string): ViewNode {
  const view = c.ws.viewNode(id)
  if (!view) throw new OperationError(`View not found: ${id}`)
  if (!c.ws.isEditable(id)) throw new OperationError('This view uses filters or styles, edit it in its source file')
  return view
}

/** Every `include`/`exclude` rule of a view, with the element each plain entry names. */
export function predicates(c: PlanContext, view: ViewNode) {
  const rules = view.body?.rules ?? []
  return rules.filter(isPredicate).map((rule) => {
    const entries = expressionEntries(rule.exprs).map((entry) => {
      // Only a bare reference names exactly one element; `x.*` and friends name a family.
      const element = isFqnRef(entry) && !entry.selector ? c.ws.fqnOf(entry.ref?.value?.ref) : undefined
      return { node: entry, element }
    })
    return { rule, include: !!rule.isInclude, entries }
  })
}

/** The plain entries of the view's include (or exclude) rules that name one of `elements`. */
function entriesNaming(c: PlanContext, view: ViewNode, elements: string[], include: boolean) {
  const wanted = new Set(elements)
  return predicates(c, view).filter((rule) => rule.include === include).flatMap((rule) => rule.entries)
    .filter((entry) => entry.element && wanted.has(entry.element))
}

/** Drops the given entries from their rules; a rule left empty goes entirely. */
export function dropEntries(c: PlanContext, view: ViewNode, doomed: Set<AstNode>): TextEdit[] {
  const file = c.ws.documentPath(view)
  const edits: TextEdit[] = []
  for (const { rule, entries } of predicates(c, view)) {
    if (!entries.some((entry) => doomed.has(entry.node))) continue
    const kept = entries.filter((entry) => !doomed.has(entry.node))
    if (kept.length === 0) {
      edits.push(c.remove(file, cst(rule).offset, cst(rule).end))
      continue
    }
    const start = cst(entries[0]!.node).offset
    const end = cst(entries.at(-1)!.node).end
    edits.push({ file, start, end, text: kept.map((entry) => cst(entry.node).text).join(', ') })
  }
  return edits
}

function appendRule(c: PlanContext, view: ViewNode, line: string): TextEdit {
  if (!view.body) throw new OperationError('Unexpected view shape in source')
  return c.appendToBlock(view.body, [line])
}

export function includeInView(c: PlanContext, op: Op<'includeInView'>): Plan {
  const view = editableView(c, op.view)
  for (const fqn of op.elements) c.element(fqn)
  if (op.elements.length === 0) return { edits: [] }
  // An element the view excludes by name comes back by dropping that exclusion.
  // Whether a wider rule then draws it is checked afterwards (see followUp).
  const unexcluded = entriesNaming(c, view, op.elements, false)
  const doomed = new Set(unexcluded.map((entry) => entry.node))
  const named = new Set([
    ...predicates(c, view).filter((rule) => rule.include).flatMap((rule) => rule.entries).map((entry) => entry.element),
    ...unexcluded.map((entry) => entry.element),
  ])
  const added = op.elements.filter((fqn) => op.descendants || !named.has(fqn))
    .flatMap((fqn) => (op.descendants ? [fqn, `${fqn}.**`] : [fqn]))
  const edits = dropEntries(c, view, doomed)
  if (added.length > 0) edits.push(appendRule(c, view, `include ${added.join(', ')}`))
  return { edits }
}

/** Drops the element's own include; when something broader still draws it, excludes it by name. */
export function removeFromView(c: PlanContext, op: Op<'removeFromView'>): Plan {
  const view = editableView(c, op.view)
  for (const fqn of op.elements) c.element(fqn)
  const own = entriesNaming(c, view, op.elements, true)
  const edits = dropEntries(c, view, new Set(own.map((entry) => entry.node)))
  const named = new Set(own.map((entry) => entry.element))
  const excluded = op.elements.filter((fqn) => !named.has(fqn))
  if (excluded.length > 0) edits.push(appendRule(c, view, `exclude ${excluded.join(', ')}`))
  return { edits }
}

/**
 * A follow-up that makes a gesture's visible effect match the view: a new
 * element drawn in a view the view's rules do not reach, or one whose exclusion
 * was dropped but no rule reaches, gets included by name; an element still drawn
 * after its own include went is excluded by name.
 */
export async function followUp(ws: Workspace, operation: Operation, created: string | null): Promise<Operation | null> {
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
