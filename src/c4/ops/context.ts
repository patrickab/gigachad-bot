// What every planner works against: the parsed workspace, the window's file,
// and the syntax-tree-aware edit builders they share.
import { GrammarUtils, type AstNode, type CstNode } from 'langium'
import { modelBlockOf, type ElementNode, type RelationNode } from '../ast.ts'
import { nameOf, parentOf } from '../fqn.ts'
import type { Workspace } from '../workspace.ts'
import { OperationError } from './operation.ts'
import { appendBlockAtEof, indentAt, indented, INDENT_UNIT, insertAt, lineStart, wholeLines, type TextEdit } from './text.ts'

export const cst = (node: AstNode): CstNode => {
  if (!node.$cstNode) throw new OperationError('Source position unavailable')
  return node.$cstNode
}

// findNodeForProperty's index does not select among repeated values, so pick from the full list.
export const propertyNode = (node: AstNode, property: string, index = 0): CstNode | undefined =>
  GrammarUtils.findNodesForProperty(cst(node), property)[index]

export class PlanContext {
  readonly ws: Workspace
  /** The file a window edits: new top-level elements go there; null when the request names none. */
  private readonly home: string | null

  constructor(ws: Workspace, home: string | null) {
    this.ws = ws
    this.home = home
  }

  homeFile(): string {
    if (this.home === null || !(this.home in this.ws.sources)) throw new OperationError('Open a file to draw into')
    return this.home
  }

  text(file: string) {
    return this.ws.sources[file] ?? ''
  }

  element(fqn: string): ElementNode {
    const node = this.ws.elementNode(fqn)
    if (!node) throw new OperationError(`Element not found: ${fqn}`)
    return node
  }

  relation(id: string): RelationNode {
    const node = this.ws.relationNode(id)
    if (!node) throw new OperationError(`Connection not found: ${id}`)
    return node
  }

  async kinds(): Promise<Set<string>> {
    return new Set(Object.keys((await this.ws.model()).specification.elements))
  }

  async childNames(parent: string | null): Promise<Set<string>> {
    const ids = await this.ws.elementIds()
    return new Set(ids.filter((id) => parentOf(id) === parent).map(nameOf))
  }

  /** Appends `lines` to `file`'s first `model { }`, creating one at the end of the file when it has none. */
  appendToModel(file: string, lines: string[]): TextEdit {
    const entry = this.ws.documents().find(({ path }) => path === file)
    const model = entry && modelBlockOf(entry.document)
    if (model) return this.appendToBlock(model, lines)
    return appendBlockAtEof(file, this.text(file), 'model', lines, 0)
  }

  /**
   * Inserts `lines` (relative indentation, "" for blank) as the last children
   * of `parent`, creating the element's `{ }` body when it has none yet. Top-level
   * elements go into the home file's model block.
   */
  appendChildren(parent: string | null, lines: string[]): TextEdit {
    if (parent === null) return this.appendToModel(this.homeFile(), lines)
    const element = this.element(parent)
    return element.body ? this.appendToBlock(element.body, lines) : this.openBody(element, lines)
  }

  /** Gives an element without a body a `{ }` one holding `lines`. */
  openBody(element: ElementNode, lines: string[]): TextEdit {
    const file = this.ws.documentPath(element)
    const indent = indentAt(this.text(file), cst(element).offset)
    const end = cst(element).end
    return insertAt(file, end, ` {\n${indented(lines, indent + INDENT_UNIT)}\n${indent}}`)
  }

  /** Inserts `lines` (relative indentation) as the last lines inside a `{ }` block. */
  appendToBlock(block: AstNode, lines: string[]): TextEdit {
    const file = this.ws.documentPath(block)
    const text = this.text(file)
    const node = cst(block)
    const close = node.end - 1
    if (text[close] !== '}') throw new OperationError('Unexpected block shape in source')
    const owner = indentAt(text, node.offset)
    const content = indented(lines, owner + INDENT_UNIT)
    const closeLine = lineStart(text, close)
    if (!text.slice(closeLine, close).trim()) return insertAt(file, closeLine, `${content}\n`)
    return insertAt(file, close, `\n${content}\n${owner}`)
  }

  replace(node: CstNode, file: string, text: string): TextEdit {
    return { file, start: node.offset, end: node.end, text }
  }

  /** Deletes a span, widened to whole lines when nothing else shares them. */
  remove(file: string, start: number, end: number): TextEdit {
    const [from, to] = wholeLines(this.text(file), start, end)
    return { file, start: from, end: to, text: '' }
  }
}
