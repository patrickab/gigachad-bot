// Pure source-text helpers: edits, offsets, indentation and LikeC4 literals.
import type { Sources } from '../workspace.ts'
import { OperationError } from './operation.ts'

export interface Span {
  file: string
  start: number
  end: number
}

export interface TextEdit extends Span {
  text: string
}

export const INDENT_UNIT = '  '

/** A zero-width insertion of `text` at offset `at`. */
export const insertAt = (file: string, at: number, text: string): TextEdit => ({ file, start: at, end: at, text })

export const quote = (value: string) => `'${value.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`

export const lineStart = (text: string, offset: number) => text.lastIndexOf('\n', offset - 1) + 1
export const indentAt = (text: string, offset: number) => /^[ \t]*/.exec(text.slice(lineStart(text, offset)))![0]
/** `lines` (relative indentation, "" for blank) joined, each prefixed with `indent`. */
export const indented = (lines: string[], indent: string) => lines.map((line) => (line ? indent + line : line)).join('\n')

/** A top-level `keyword { }` block holding `lines` (relative indentation), ending in a newline. */
export const block = (keyword: string, lines: string[]) => `${keyword} {\n${indented(lines, INDENT_UNIT)}\n}\n`

/** Appends `block(keyword, lines)` to `file` (whose source is `text`), after `gap` blank lines when the file has text. */
export function appendBlockAtEof(file: string, text: string, keyword: string, lines: string[], gap: number): TextEdit {
  const separator = text ? (text.endsWith('\n') ? '' : '\n') + '\n'.repeat(gap) : ''
  return insertAt(file, text.length, separator + block(keyword, lines))
}

/** Widens a span to its full lines when nothing else shares them, so deletions leave no blank line. */
export function wholeLines(text: string, start: number, end: number): [number, number] {
  const from = lineStart(text, start)
  let to = text.indexOf('\n', end)
  if (to === -1) to = text.length
  if (text.slice(from, start).trim() || text.slice(end, to).trim()) return [start, end]
  return [from, to < text.length ? to + 1 : to]
}

/** LikeC4 derives view titles from element titles and rejects newlines in them. */
export const singleLine = (value: string) => value.replace(/\s+/g, ' ').trim()

/** The single-line title of a create or rename operation; an empty one fails with `message`. */
export function requireTitle(value: string, message: string): string {
  const title = singleLine(value)
  if (!title) throw new OperationError(message)
  return title
}

/** One line stays a quoted string; several become a dedented ''' block, one bullet per line. */
export function descriptionValue(description: string, indent: string): string {
  const lines = description.split('\n').map((line) => line.trim()).filter(Boolean)
  if (lines.length <= 1) return quote(lines[0] ?? '')
  const body = lines.map((line) => `${indent}${INDENT_UNIT}${line.replaceAll("'''", "''")}`).join('\n')
  return `'''\n${body}\n${indent}'''`
}

export function applyEdits(sources: Sources, edits: TextEdit[]): Sources {
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
