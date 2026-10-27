// Planners that delete elements and connections, taking every use along.
import { AstUtils, type AstNode } from 'langium'
import { isExtend, isRelation, isView, type ViewNode } from '../ast.ts'
import { isWithin } from '../fqn.ts'
import { cst, type PlanContext } from './context.ts'
import { OperationError, type Op, type Plan } from './operation.ts'
import { dropEntries, predicates } from './plan-views.ts'
import type { Span, TextEdit } from './text.ts'

/**
 * Removes whole-line `spans` (one inside another goes with it) and applies the
 * entry `edits` that do not fall inside a removed span.
 */
function cut(c: PlanContext, spans: Span[], edits: TextEdit[]): Plan {
  const outermost = spans.filter((span) => !spans.some((other) => other !== span && other.file === span.file
    && other.start <= span.start && other.end >= span.end && (other.start !== span.start || other.end !== span.end)))
  const removals = outermost.filter((span, index) => outermost.findIndex((other) => other.file === span.file && other.start === span.start) === index)
    .map((span) => c.remove(span.file, span.start, span.end))
  const rest = edits.filter((edit) => !removals.some((gone) => gone.file === edit.file && gone.start <= edit.start && gone.end >= edit.end))
  return { edits: [...removals, ...rest] }
}

const inside = (node: AstNode, ancestor: AstNode | undefined) => {
  for (let cursor: AstNode | undefined = node; cursor; cursor = cursor.$container) if (cursor === ancestor) return true
  return false
}

/**
 * Everything naming an element that is gone: one `gone` is about to remove, or (without
 * `gone`) any reference that no longer resolves after a hand edit. That is connections,
 * a `view x of gone`, an `extend gone` block (cut whole)
 * and `include`/`exclude` entries (dropped from their rules).
 */
function uses(c: PlanContext, gone?: string[]): { spans: Span[], edits: TextEdit[] } {
  const dangling: AstNode[] = []
  const views: ViewNode[] = []
  for (const { document } of c.ws.documents()) {
    for (const node of AstUtils.streamAllContents(document.parseResult.value)) {
      if (isView(node)) views.push(node)
      for (const info of AstUtils.streamReferences(node)) {
        const target = c.ws.fqnOf(info.reference.ref)
        if (gone ? target && gone.some((fqn) => isWithin(target, fqn)) : !info.reference.ref) dangling.push(info.container)
      }
    }
  }
  const spans = new Map<AstNode, Span>()
  for (const node of dangling) {
    for (let cursor: AstNode | undefined = node; cursor; cursor = cursor.$container) {
      if (isRelation(cursor) || (isView(cursor) && inside(node, cursor.viewOf)) || (isExtend(cursor) && inside(node, cursor.element))) {
        spans.set(cursor, { file: c.ws.documentPath(cursor), start: cst(cursor).offset, end: cst(cursor).end })
        break
      }
    }
  }
  const edits: TextEdit[] = []
  for (const view of views) {
    const doomed = new Set(predicates(c, view).flatMap((rule) => rule.entries)
      .filter((entry) => dangling.some((node) => inside(node, entry.node))).map((entry) => entry.node))
    if (doomed.size > 0) edits.push(...dropEntries(c, view, doomed))
  }
  return { spans: [...spans.values()], edits }
}

/**
 * Deleting takes every use of the deleted elements along: their connections and the
 * view entries, views and `extend` blocks naming them.
 */
export function deleteElements(c: PlanContext, op: Op<'delete'>): Plan {
  const selected = [...op.elements.map((fqn) => c.element(fqn)), ...op.relations.map((id) => c.relation(id))]
    .map((node) => ({ file: c.ws.documentPath(node), start: cst(node).offset, end: cst(node).end }))
  if (selected.length === 0) throw new OperationError('Nothing selected to delete')
  const { spans, edits } = uses(c, op.elements)
  return cut(c, [...selected, ...spans], edits)
}

/**
 * Hand edits that removed an element leave connections and view entries naming it.
 * Drops every use with an unresolved reference, so deleting an element in text takes
 * its uses along as the diagram does.
 */
export function pruneDangling(c: PlanContext): Plan {
  const { spans, edits } = uses(c)
  return cut(c, spans, edits)
}
