/**
 * Pure notebook file logic: nbformat text to and from the editor's content, and the merge of a change written on disk into a live shared model.
 *
 * This file never imports `@jupyter/ydoc` or `@jupyterlab/*`. `SharedNotebookLike` lists the `YNotebook` and `YCodeCell` members it relies on, the real model
 * (`@jupyter/ydoc` 4.x, `notebook.sharedModel`) must satisfy it.
 */

import { isRecord } from "@/lib/utils"

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue }
export type JsonObject = { [key: string]: JsonValue }

/** One nbformat cell. `source` and the text of outputs are a string (rejoined) or the file's list of lines. */
export interface NotebookCellContent {
  cell_type: string
  id?: string
  source: string | string[]
  metadata: JsonObject
  outputs?: JsonObject[]
  execution_count?: number | null
  attachments?: JsonObject
  [key: string]: JsonValue | undefined
}

export interface NotebookContent {
  cells: NotebookCellContent[]
  metadata: JsonObject
  nbformat: number
  nbformat_minor: number
}

/** The `@jupyter/ydoc` `YCodeCell`, `YMarkdownCell` and `YRawCell` members this file uses. */
export interface SharedCellLike {
  /** Undefined for a cell loaded without one: jupyter-react keeps no id of a notebook below nbformat 4.5 and makes none for a cell that lacks it. */
  readonly id: string | undefined
  readonly cell_type: string
  getSource(): string
  updateSource(start: number, end: number, value?: string): void
  setMetadata(key: string, value: JsonValue): void
  deleteMetadata(key: string): void
  /** Code cells only. */
  setOutputs?(outputs: unknown[]): void
  execution_count?: number | null
}

/**
 * The `@jupyter/ydoc` `YNotebook` members this file uses.
 *
 * `ydoc` is `YDocument.ydoc`, the Yjs document. Its `update` event is the only signal that sees every change: `YNotebook.changed` reports the cell list,
 * the notebook metadata and the dirty flag but not a cell's source, outputs or execution count (those go to each `YBaseCell.changed`).
 */
export interface SharedNotebookLike {
  readonly cells: readonly SharedCellLike[]
  readonly isDisposed?: boolean
  readonly ydoc: {
    on(event: "update", listener: () => void): void
    off(event: "update", listener: () => void): void
  }
  insertCell(index: number, cell: NotebookCellContent): SharedCellLike
  /** Remove the cells `from` (inclusive) to `to` (exclusive). */
  deleteCellRange(from: number, to: number): void
  /** Move the cell at `from` so it ends at index `to`. */
  moveCell(from: number, to: number): void
  setMetadata(key: string, value: JsonValue): void
  deleteMetadata(key: string): void
  /** Bundle the changes of `f` in one Yjs transaction. Not `undoable` changes stay out of the user's undo stack. */
  transact(f: () => void, undoable?: boolean): void
  /** The nbformat content, `YNotebook.toJSON()` (typed loosely here because its metadata types are wider than JSON). */
  toJSON(): object
}

/**
 * One change to a live notebook. Ops apply in order and every `index` is valid at that moment:
 * first the structure (deletes, then inserts and moves toward the target order), then the fields of the cells at their final index.
 */
export type NotebookOp =
  | { type: "deleteCell"; index: number }
  | { type: "insertCell"; index: number; cell: NotebookCellContent }
  | { type: "moveCell"; from: number; to: number }
  | { type: "source"; index: number; value: string }
  | { type: "outputs"; index: number; outputs: JsonObject[] }
  | { type: "executionCount"; index: number; value: number | null }
  | { type: "cellMetadata"; index: number; set: JsonObject; remove: string[] }
  | { type: "notebookMetadata"; set: JsonObject; remove: string[] }

const isObject = isRecord<JsonValue>
const isString = (value: unknown): value is string => typeof value === "string"
const isLines = (value: unknown): value is string[] => Array.isArray(value) && value.every(isString)

function cloneJson<T extends JsonValue | undefined>(value: T): T {
  if (Array.isArray(value)) return value.map((item) => cloneJson(item)) as T
  if (isObject(value)) {
    const copy: JsonObject = {}
    for (const key of Object.keys(value)) copy[key] = cloneJson(value[key])
    return copy as T
  }
  return value
}

function jsonEqual(a: JsonValue | undefined, b: JsonValue | undefined): boolean {
  if (a === b) return true
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((item, index) => jsonEqual(item, b[index]))
  if (isObject(a)) {
    if (!isObject(b)) return false
    const keys = Object.keys(a)
    return keys.length === Object.keys(b).length && keys.every((key) => key in b && jsonEqual(a[key], b[key]))
  }
  return false
}

// ---------------------------------------------------------------------------------------------------------------------
// nbformat text
// ---------------------------------------------------------------------------------------------------------------------

/** Python's `str.splitlines(True)`, which nbformat writes multi-line text with. */
const LINE = /[^\n\r\v\f\x1c-\x1e\x85\u2028\u2029]*(?:\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029])|[^\n\r\v\f\x1c-\x1e\x85\u2028\u2029]+/g

export function splitLines(text: string): string[] {
  return text.match(LINE) ?? []
}

/** nbformat writes these mimes as lines and keeps every other (base64, JSON) as it is. */
const isLineMime = (mime: string) => mime.startsWith("text/") || mime === "application/javascript" || mime === "image/svg+xml"

const EMPTY_NOTEBOOK = (): NotebookContent => ({ cells: [], metadata: {}, nbformat: 4, nbformat_minor: 5 })

/** Parse notebook text. Blank text (a new empty file) is an empty notebook, anything else that is not a notebook throws. */
export function parseNotebook(text: string): NotebookContent {
  if (text.trim() === "") return EMPTY_NOTEBOOK()
  const parsed: unknown = JSON.parse(text)
  if (!isObject(parsed) || !Array.isArray(parsed.cells) || typeof parsed.nbformat !== "number" || !parsed.cells.every(isObject)) {
    throw new Error("Not a notebook")
  }
  if (!isObject(parsed.metadata)) parsed.metadata = {}
  if (typeof parsed.nbformat_minor !== "number") parsed.nbformat_minor = 0
  return parsed as unknown as NotebookContent
}

const joinLines = (value: JsonValue | undefined): JsonValue | undefined => (isLines(value) ? value.join("") : value)

function rejoinBundle(bundle: JsonValue | undefined): JsonValue | undefined {
  if (!isObject(bundle)) return bundle
  const out: JsonObject = {}
  for (const [mime, value] of Object.entries(bundle)) out[mime] = isLineMime(mime) ? (joinLines(value) as JsonValue) : value
  return out
}

function rejoinOutput(output: JsonObject): JsonObject {
  const out: JsonObject = { ...output }
  if ((output.output_type === "execute_result" || output.output_type === "display_data") && isObject(output.data)) out.data = rejoinBundle(output.data) as JsonObject
  else if (output.output_type && isLines(output.text)) out.text = output.text.join("")
  return out
}

/** The notebook as nbformat readers hold it: `source` and text outputs as strings. A copy, the input stays untouched. */
export function rejoinNotebook(content: NotebookContent): NotebookContent {
  const cells = content.cells.map((cell) => {
    const next: NotebookCellContent = { ...cloneJson(cell as JsonObject), source: joinLines(cell.source) as string } as NotebookCellContent
    if (isObject(cell.attachments)) next.attachments = Object.fromEntries(Object.entries(cell.attachments).map(([name, bundle]) => [name, rejoinBundle(bundle) as JsonValue]))
    if (Array.isArray(cell.outputs)) next.outputs = cell.outputs.map((output) => rejoinOutput(cloneJson(output)))
    return next
  })
  return { ...content, cells, metadata: cloneJson(content.metadata) }
}

/** Cell metadata the notebook UI adds and nbformat never stores (`trusted`) or the library sets on every cell (`editable`). */
const CELL_UI_KEYS = ["trusted", "editable"]
/** Notebook metadata the library drops on load, which the file keeps. */
const NOTEBOOK_KEPT_KEYS = ["orig_nbformat", "orig_nbformat_minor"]

/** The cell of `original` that `cell` (at `index`) is, by id, else by position when the file had no id for it. */
function originalCellFinder(original: NotebookContent | undefined): (cell: NotebookCellContent, index: number) => NotebookCellContent | undefined {
  if (!original) return () => undefined
  const byId = new Map<string, NotebookCellContent>()
  for (const cell of original.cells) if (typeof cell.id === "string" && !byId.has(cell.id)) byId.set(cell.id, cell)
  return (cell, index) => {
    if (typeof cell.id === "string") {
      const found = byId.get(cell.id)
      if (found) return found
    }
    const positional = original.cells[index]
    if (positional && positional.cell_type === cell.cell_type && (positional.id === undefined || cell.id === undefined)) return positional
    return undefined
  }
}

/**
 * The content as the file should hold it, with strings for text (`rejoinNotebook`), against the notebook last stored (`original`):
 * library-injected cell `metadata.editable` and `trusted` dropped unless the stored cell had them, no ids where the file's format has none
 * (nbformat below 4.5, or the stored cell had none), output `transient` dropped and `orig_nbformat` kept from the stored notebook.
 */
function canonicalNotebook(content: NotebookContent, original: NotebookContent | undefined): NotebookContent {
  const rejoined = rejoinNotebook(content)
  const find = originalCellFinder(original)
  const noIds = rejoined.nbformat === 4 && rejoined.nbformat_minor < 5
  rejoined.cells = rejoined.cells.map((cell, index) => {
    const match = find(cell, index)
    if (noIds || (match && match.id === undefined)) delete cell.id
    const metadata: JsonObject = {}
    for (const [key, value] of Object.entries(cell.metadata ?? {})) {
      if (key === "trusted" || (key === "editable" && !(match && isObject(match.metadata) && "editable" in match.metadata))) continue
      metadata[key] = value
    }
    cell.metadata = metadata
    if (Array.isArray(cell.outputs)) cell.outputs = cell.outputs.map((output) => Object.fromEntries(Object.entries(output).filter(([key]) => key !== "transient")))
    return cell
  })
  if (original) {
    for (const key of NOTEBOOK_KEPT_KEYS) {
      if (!(key in rejoined.metadata) && key in original.metadata) rejoined.metadata[key] = cloneJson(original.metadata[key])
    }
  }
  return rejoined
}

const isSortedKeys = (keys: string[]) => keys.every((key, index) => index === 0 || keys[index - 1] < key)

/** `value` with its object keys sorted like nbformat does (`sort_keys`), or in the order of `original` when that was not sorted. `child` overrides one key's value. */
function arranged(value: JsonValue | undefined, original: JsonValue | undefined, child?: (key: string) => JsonValue | undefined): JsonValue | undefined {
  if (Array.isArray(value)) return value.map((item, index) => arranged(item, Array.isArray(original) ? original[index] : undefined) as JsonValue)
  if (!isObject(value)) return value
  const source = isObject(original) ? original : undefined
  const originalKeys = source ? Object.keys(source) : []
  const keys = Object.keys(value)
  const order = !source || isSortedKeys(originalKeys)
    ? [...keys].sort()
    : [...originalKeys.filter((key) => key in value), ...keys.filter((key) => !(key in source)).sort()]
  const out: JsonObject = {}
  for (const key of order) {
    const override = child?.(key)
    out[key] = override !== undefined ? override : (arranged(value[key], source?.[key]) as JsonValue)
  }
  return out
}

/** Text of a field nbformat writes as lines: the stored list of lines when it holds the same text, a stored string, else the lines. */
function styledText(value: JsonValue | undefined, stored: JsonValue | undefined): JsonValue | undefined {
  if (!isString(value)) return value
  if (isLines(stored) && stored.join("") === value) return [...stored]
  if (isString(stored)) return value
  return splitLines(value)
}

function styledBundle(bundle: JsonValue | undefined, stored: JsonValue | undefined): JsonValue | undefined {
  if (!isObject(bundle)) return bundle
  const storedBundle = isObject(stored) ? stored : undefined
  const out: JsonObject = {}
  for (const [mime, value] of Object.entries(bundle)) out[mime] = (isLineMime(mime) ? styledText(value, storedBundle?.[mime]) : value) as JsonValue
  return out
}

function styledOutput(output: JsonObject, stored: JsonValue | undefined): JsonObject {
  const storedOutput = isObject(stored) && stored.output_type === output.output_type ? stored : undefined
  const out: JsonObject = { ...output }
  if (isObject(output.data)) out.data = styledBundle(output.data, storedOutput?.data) as JsonObject
  else if (output.output_type === "stream" && isString(output.text)) out.text = styledText(output.text, storedOutput?.text) as JsonValue
  return out
}

function styledCell(cell: NotebookCellContent, stored: NotebookCellContent | undefined): JsonObject {
  const out: JsonObject = { ...(cell as JsonObject), source: styledText(cell.source, stored?.source) as JsonValue }
  if (isObject(cell.attachments)) {
    out.attachments = Object.fromEntries(Object.entries(cell.attachments).map(([name, bundle]) => [name, styledBundle(bundle, isObject(stored?.attachments) ? stored.attachments[name] : undefined) as JsonValue]))
  }
  if (Array.isArray(cell.outputs)) out.outputs = cell.outputs.map((output, index) => styledOutput(output, stored?.outputs?.[index]))
  return out
}

/**
 * The file text of `content` in the format Jupyter writes: 1-space indent, sorted keys, text as lines, a trailing newline.
 * `original` is the notebook as last stored (parsed, with its lists of lines): everything it does the same is written as it was, so an unedited notebook round trips byte for byte.
 */
export function serializeNotebook(content: NotebookContent, original?: NotebookContent): string {
  const canonical = canonicalNotebook(content, original)
  const find = originalCellFinder(original)
  const cells = canonical.cells.map((cell, index) => {
    const stored = find(cell, index)
    return arranged(styledCell(cell, stored), stored as JsonObject | undefined) as JsonValue
  })
  const document = arranged(canonical as unknown as JsonObject, original as unknown as JsonObject | undefined, (key) => (key === "cells" ? cells : undefined))
  return `${JSON.stringify(document, null, 1)}\n`
}

/** A single-line text of `content` that is equal for notebooks holding the same data, whatever their key order or lists of lines. For comparisons. */
export function normalizeNotebook(content: NotebookContent, original?: NotebookContent): string {
  return JSON.stringify(arranged(canonicalNotebook(content, original) as unknown as JsonObject, undefined))
}

// ---------------------------------------------------------------------------------------------------------------------
// Merge of a disk change
// ---------------------------------------------------------------------------------------------------------------------

/** Whether every cell has its own id, so cells can be matched by id. */
const hasUniqueIds = (cells: NotebookCellContent[]) => {
  const seen = new Set<string>()
  for (const cell of cells) {
    if (typeof cell.id !== "string" || seen.has(cell.id)) return false
    seen.add(cell.id)
  }
  return true
}

const isCode = (cell: NotebookCellContent) => cell.cell_type === "code"

/** Metadata keys of `cell` that take part in the merge. */
function mergeableMetadata(metadata: JsonObject | undefined, ignored: readonly string[]): JsonObject {
  const out: JsonObject = {}
  for (const [key, value] of Object.entries(metadata ?? {})) if (!ignored.includes(key)) out[key] = value
  return out
}

/** Take the incoming value unless it is what the editor already has, or what it started from (then only the editor changed it). */
const takesIncoming = (base: JsonValue | undefined, current: JsonValue | undefined, incoming: JsonValue | undefined) =>
  !jsonEqual(current, incoming) && !jsonEqual(base, incoming)

function metadataOps(base: JsonObject | undefined, current: JsonObject, incoming: JsonObject, ignored: readonly string[]): { set: JsonObject; remove: string[] } {
  const [b, c, i] = [mergeableMetadata(base, ignored), mergeableMetadata(current, ignored), mergeableMetadata(incoming, ignored)]
  const set: JsonObject = {}
  const remove: string[] = []
  for (const key of new Set([...Object.keys(c), ...Object.keys(i)])) {
    if (!takesIncoming(b[key], c[key], i[key])) continue
    if (i[key] === undefined) remove.push(key)
    else set[key] = cloneJson(i[key])
  }
  return { set, remove }
}

const sameCell = (a: NotebookCellContent, b: NotebookCellContent) =>
  a.cell_type === b.cell_type
  && jsonEqual(a.source, b.source)
  && jsonEqual(a.outputs, b.outputs)
  && jsonEqual(a.execution_count, b.execution_count)
  && jsonEqual(a.attachments, b.attachments)
  && jsonEqual(mergeableMetadata(a.metadata, CELL_UI_KEYS), mergeableMetadata(b.metadata, CELL_UI_KEYS))

/**
 * Merge a notebook written on disk (`incoming`) into what the editor shows (`current`), knowing what the editor started from (`base`, the previously stored notebook).
 * Returns the changes to make to the editor's model, nothing when it already shows `incoming`.
 *
 * Policy ("merge theirs") per value, a cell field or a metadata key:
 * - only the editor changed it: keep it
 * - only the file changed it, or both did: the file's value wins
 * Cells are matched by id (by position when any of the three notebooks lacks unique ids). A cell the file deleted is deleted, a cell only the editor added is kept,
 * a cell only the editor deleted stays deleted unless the file changed it, the file's order wins when the file moved a cell and the editor's order stays otherwise.
 * A cell whose type or attachments the file changed is replaced by the file's cell, a model cannot change them in place.
 */
export function diffNotebook(base: NotebookContent, current: NotebookContent, incoming: NotebookContent): NotebookOp[] {
  const [b, c, inc] = [rejoinNotebook(base), rejoinNotebook(current), rejoinNotebook(incoming)]
  const ids = hasUniqueIds(b.cells) && hasUniqueIds(c.cells) && hasUniqueIds(inc.cells)
  const keysOf = (cells: NotebookCellContent[]) => cells.map((cell, index) => (ids ? (cell.id as string) : `#${index}`))
  const [bKeys, cKeys, iKeys] = [keysOf(b.cells), keysOf(c.cells), keysOf(inc.cells)]
  const bCells = new Map(bKeys.map((key, index) => [key, b.cells[index]]))
  const cCells = new Map(cKeys.map((key, index) => [key, c.cells[index]]))
  const iCells = new Map(iKeys.map((key, index) => [key, inc.cells[index]]))

  const removed = new Set<string>()
  const replaced = new Set<string>()
  for (const key of cKeys) {
    const [was, now, next] = [bCells.get(key), cCells.get(key) as NotebookCellContent, iCells.get(key)]
    if (!next) {
      if (was) removed.add(key)
      continue
    }
    const typeChanged = now.cell_type !== next.cell_type && was?.cell_type !== next.cell_type
    const attachmentsChanged = takesIncoming(was?.attachments, now.attachments, next.attachments)
    if (typeChanged || attachmentsChanged) replaced.add(key)
  }
  const inserted = new Set<string>()
  for (const key of iKeys) {
    if (cCells.has(key)) continue
    const was = bCells.get(key)
    if (!was || !sameCell(was, iCells.get(key) as NotebookCellContent)) inserted.add(key)
  }

  // The order the cells end in
  const kept = cKeys.filter((key) => !removed.has(key))
  const baseOrder = bKeys.filter((key) => iCells.has(key))
  const incomingOrder = iKeys.filter((key) => bCells.has(key))
  const incomingMoved = baseOrder.some((key, index) => key !== incomingOrder[index])
  let target: string[]
  if (!incomingMoved) {
    target = [...kept]
    for (const key of iKeys) {
      if (!inserted.has(key)) continue
      let after = iKeys.indexOf(key) - 1
      while (after >= 0 && !target.includes(iKeys[after])) after--
      target.splice(after < 0 ? 0 : target.indexOf(iKeys[after]) + 1, 0, key)
    }
  } else {
    target = iKeys.filter((key) => inserted.has(key) || kept.includes(key))
    for (const key of kept) {
      if (iCells.has(key)) continue
      let before = cKeys.indexOf(key) - 1
      while (before >= 0 && !target.includes(cKeys[before])) before--
      target.splice(before < 0 ? 0 : target.indexOf(cKeys[before]) + 1, 0, key)
    }
  }

  const ops: NotebookOp[] = []
  const notebook = metadataOps(b.metadata, c.metadata, inc.metadata, NOTEBOOK_KEPT_KEYS)
  if (Object.keys(notebook.set).length || notebook.remove.length) ops.push({ type: "notebookMetadata", ...notebook })

  const order = [...cKeys]
  for (let index = order.length - 1; index >= 0; index--) {
    if (!removed.has(order[index]) && !replaced.has(order[index])) continue
    ops.push({ type: "deleteCell", index })
    order.splice(index, 1)
  }
  const cellToInsert = (key: string): NotebookCellContent => {
    const cell = cloneJson(iCells.get(key) as unknown as JsonObject) as unknown as NotebookCellContent
    if (!ids) delete cell.id
    return cell
  }
  target.forEach((key, index) => {
    if (inserted.has(key) || replaced.has(key)) {
      ops.push({ type: "insertCell", index, cell: cellToInsert(key) })
      order.splice(index, 0, key)
    } else if (order[index] !== key) {
      const from = order.indexOf(key, index)
      ops.push({ type: "moveCell", from, to: index })
      order.splice(index, 0, ...order.splice(from, 1))
    }
  })

  target.forEach((key, index) => {
    const [was, now, next] = [bCells.get(key), cCells.get(key), iCells.get(key)]
    if (inserted.has(key) || replaced.has(key) || !now || !next) return
    if (takesIncoming(was?.source, now.source, next.source)) ops.push({ type: "source", index, value: next.source as string })
    if (isCode(now) && isCode(next)) {
      if (takesIncoming(was?.outputs, now.outputs, next.outputs)) ops.push({ type: "outputs", index, outputs: cloneJson(next.outputs ?? []) })
      if (takesIncoming(was?.execution_count, now.execution_count, next.execution_count)) ops.push({ type: "executionCount", index, value: next.execution_count ?? null })
    }
    const metadata = metadataOps(was?.metadata, now.metadata, next.metadata, CELL_UI_KEYS)
    if (Object.keys(metadata.set).length || metadata.remove.length) ops.push({ type: "cellMetadata", index, ...metadata })
  })
  return ops
}

/** Replace the text of `cell` with `value`, touching only the span that differs so a cursor elsewhere in the cell stays put. Call inside `model.transact(..., false)` to keep it out of the undo stack. */
export function setCellSource(cell: Pick<SharedCellLike, "getSource" | "updateSource">, value: string): void {
  const old = cell.getSource()
  if (old === value) return
  let start = 0
  const shortest = Math.min(old.length, value.length)
  while (start < shortest && old.charCodeAt(start) === value.charCodeAt(start)) start++
  let oldEnd = old.length
  let newEnd = value.length
  while (oldEnd > start && newEnd > start && old.charCodeAt(oldEnd - 1) === value.charCodeAt(newEnd - 1)) {
    oldEnd--
    newEnd--
  }
  // Never cut a surrogate pair in two
  const isHigh = (code: number) => code >= 0xd800 && code <= 0xdbff
  const isLow = (code: number) => code >= 0xdc00 && code <= 0xdfff
  if (start > 0 && isHigh(old.charCodeAt(start - 1))) start--
  if (oldEnd < old.length && isLow(old.charCodeAt(oldEnd))) {
    oldEnd++
    newEnd++
  }
  cell.updateSource(start, oldEnd, value.slice(start, newEnd))
}

function applyMetadata(target: Pick<SharedNotebookLike, "setMetadata" | "deleteMetadata">, set: JsonObject, remove: string[]): void {
  for (const [key, value] of Object.entries(set)) target.setMetadata(key, value)
  for (const key of remove) target.deleteMetadata(key)
}

/**
 * Apply `diffNotebook`'s changes to `model`, outside the undo stack.
 *
 * A `YNotebook` updates its `cells` list when a transaction ends, so each change of the structure is its own transaction
 * (the next one finds the cells where the previous left them) and the field changes of the final cells share one.
 */
export function applyNotebookOps(model: SharedNotebookLike, ops: readonly NotebookOp[]): void {
  const fields: NotebookOp[] = []
  for (const op of ops) {
    if (op.type === "deleteCell") model.transact(() => model.deleteCellRange(op.index, op.index + 1), false)
    else if (op.type === "insertCell") model.transact(() => { model.insertCell(op.index, op.cell) }, false)
    else if (op.type === "moveCell") model.transact(() => model.moveCell(op.from, op.to), false)
    else fields.push(op)
  }
  if (fields.length === 0) return
  model.transact(() => {
    for (const op of fields) {
      switch (op.type) {
        case "source":
          setCellSource(model.cells[op.index], op.value)
          break
        case "outputs":
          model.cells[op.index].setOutputs?.(op.outputs)
          break
        case "executionCount":
          model.cells[op.index].execution_count = op.value
          break
        case "cellMetadata":
          applyMetadata(model.cells[op.index], op.set, op.remove)
          break
        case "notebookMetadata":
          applyMetadata(model, op.set, op.remove)
          break
      }
    }
  }, false)
}

/** Merge `incoming` into `model` (see `diffNotebook`), where `base` is the notebook the model last matched on disk. */
export function applyIncomingNotebook(model: SharedNotebookLike, base: NotebookContent, incoming: NotebookContent): void {
  applyNotebookOps(model, diffNotebook(base, notebookOf(model), incoming))
}

/** The notebook `model` holds right now. */
export function notebookOf(model: SharedNotebookLike): NotebookContent {
  return model.toJSON() as NotebookContent
}
