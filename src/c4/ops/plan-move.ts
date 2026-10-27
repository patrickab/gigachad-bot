// Planners that rename or move an element, rewriting every reference to it.
import { AstUtils, type AstNode, type CstNode } from 'langium'
import { isWithin, nameOf, parentOf } from '../fqn.ts'
import { cst, propertyNode, type PlanContext } from './context.ts'
import { ID_PATTERN } from './ids.ts'
import { OperationError, type Op, type Plan } from './operation.ts'
import { applyEdits, indentAt, wholeLines, type TextEdit } from './text.ts'

/** Every reference in the workspace, with the element FQN it resolves to. */
function* references(c: PlanContext): Generator<{ holder: AstNode, cst: CstNode, target: string, file: string }> {
  for (const { path, document } of c.ws.documents()) {
    for (const node of AstUtils.streamAst(document.parseResult.value)) {
      for (const reference of AstUtils.streamReferences(node)) {
        const target = c.ws.fqnOf(reference.reference.ref)
        const refNode = reference.reference.$refNode
        if (target && refNode) yield { holder: node, cst: refNode, target, file: path }
      }
    }
  }
}

export async function rename(c: PlanContext, op: Op<'rename'>): Promise<Plan> {
  if (!ID_PATTERN.test(op.id)) throw new OperationError('An id may contain only letters, digits, and underscores, and must not start with a digit')
  const element = c.element(op.element)
  const parent = parentOf(op.element)
  if (op.id === nameOf(op.element)) return { edits: [] }
  if ((await c.childNames(parent)).has(op.id)) throw new OperationError(`An element named ${op.id} already exists there`)
  const name = propertyNode(element, 'name')
  if (!name) throw new OperationError('Unexpected element shape in source')
  const edits = [c.replace(name, c.ws.documentPath(element), op.id)]
  for (const reference of references(c)) {
    if (reference.target === op.element) edits.push(c.replace(reference.cst, reference.file, op.id))
  }
  return { edits, renames: { [op.element]: parent ? `${parent}.${op.id}` : op.id } }
}

/** Moves an element's text under `parent`, rewriting qualified references into it as absolute FQNs. */
export async function reparent(c: PlanContext, op: Op<'reparent'>): Promise<Plan> {
  const fqn = op.element
  const element = c.element(fqn)
  if (parentOf(fqn) === op.parent) return { edits: [] }
  if (op.parent !== null) {
    c.element(op.parent)
    if (isWithin(op.parent, fqn)) throw new OperationError('An element cannot move into itself')
  }
  const name = nameOf(fqn)
  if ((await c.childNames(op.parent)).has(name)) throw new OperationError(`Cannot move ${name} ${op.parent ? `into ${op.parent}` : 'to the top level'}: another element there is already named ${name}`)
  const moved = op.parent ? `${op.parent}.${name}` : name
  const mapFqn = (target: string) => moved + target.slice(fqn.length)

  const file = c.ws.documentPath(element)
  const text = c.text(file)
  const [cutStart, cutEnd] = wholeLines(text, cst(element).offset, cst(element).end)

  // Every qualified chain (`a.b.c`) reaching into the moved subtree is
  // rewritten whole as an absolute FQN, so it resolves from anywhere.
  const chains = new Map<CstNode, { file: string, text: string }>()
  for (const reference of references(c)) {
    if (!isWithin(reference.target, fqn)) continue
    let head = reference.holder
    while (head.$container && head.$container.$type === head.$type && head.$containerProperty === 'parent') head = head.$container
    const headReference = AstUtils.streamReferences(head).head()
    const headTarget = c.ws.fqnOf(headReference?.reference.ref)
    if (!headTarget || !isWithin(headTarget, fqn)) continue
    chains.set(cst(head), { file: reference.file, text: mapFqn(headTarget) })
  }
  const outside: TextEdit[] = []
  const inside: TextEdit[] = []
  for (const [node, { file: chainFile, text: replacement }] of chains) {
    const edit = { file: chainFile, start: node.offset, end: node.end, text: replacement }
    // Chains inside the cut block are applied to its text, with offsets relative to the block.
    if (chainFile === file && node.offset >= cutStart && node.end <= cutEnd) inside.push({ ...edit, start: edit.start - cutStart, end: edit.end - cutStart })
    else outside.push(edit)
  }
  const block = applyEdits({ block: text.slice(cutStart, cutEnd) }, inside.map((edit) => ({ ...edit, file: 'block' }))).block
  const oldIndent = indentAt(text, cst(element).offset)
  const relative = block.replace(/\n$/, '').split('\n').map((line) => (line.startsWith(oldIndent) ? line.slice(oldIndent.length) : line))
  return { edits: [...outside, { file, start: cutStart, end: cutEnd, text: '' }, c.appendChildren(op.parent, relative)], renames: { [fqn]: moved } }
}
