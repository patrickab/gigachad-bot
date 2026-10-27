import type { EdgeStyle } from "./edgeRouting"
import type { StrokeData } from "@/lib/drawing"

export const A4_WIDTH = 794
export const A4_ASPECT = 1123 / 794 // height / width — page frames are locked to this
export const FALLBACK_ASPECT = 1.3 // height/width used until an image's or PDF's real aspect is measured
export const CANVAS_ASPECT = 0.7 // nested canvas windows have no intrinsic aspect — fix one
export const TEXT_DEFAULT_WIDTH = 240 // box size for a new text note
export const TEXT_DEFAULT_HEIGHT = 80

// CanvasFrame  = content that IS the artwork (page or image). Rasterized into the PDF/image export.
// CanvasAttachment = a live, scrollable file reference (PDF, document, architecture), or a
//                nested canvas window carrying its own CanvasDocument inline. NOT baked into export.
// CanvasText = handwriting-style note in a resizable box, rasterized into export.

export interface CanvasFrame {
  id: string
  kind: "page" | "image"
  x: number
  y: number
  width: number
  path?: string // image source (library/project path); pages have none
}

export interface CanvasAttachment {
  id: string
  kind: "pdf" | "canvas" | "architecture-graph" | "document"
  path?: string // pdf source; nested canvases have none
  canvas?: CanvasDocument // nested canvas contents (kind === "canvas")
  x: number
  y: number
  width: number
  height?: number // Architecture Diagrams own their viewport height; legacy attachments retain their aspect
  title?: string // user label, independent of the backing document filename
  page?: number // selected PDF page, persisted with the canvas
  edgeStyle?: EdgeStyle // architecture windows: how connections are drawn; LikeC4 has no such setting
}

export interface CanvasText {
  id: string
  x: number
  y: number
  width: number
  height: number
  text: string
  color: string
  size: number
}

export interface CanvasDocument {
  version: 1
  viewport?: { scale: number; centerX: number; centerY: number }
  frames: CanvasFrame[]
  strokes: StrokeData[]
  attachments: CanvasAttachment[]
  texts: CanvasText[]
}

type CanvasEntity = CanvasFrame | StrokeData | CanvasAttachment | CanvasText

/** The document's id-keyed entity arrays. Order is the order mutations are derived in. */
export const CANVAS_COLLECTIONS = ["strokes", "frames", "attachments", "texts"] as const
export type CanvasCollection = (typeof CANVAS_COLLECTIONS)[number]

/** Singular entity name as the backend's mutation protocol spells it. */
type CanvasEntityKind = "stroke" | "frame" | "attachment" | "text"

export const KIND_COLLECTION: Record<CanvasEntityKind, CanvasCollection> = {
  stroke: "strokes",
  frame: "frames",
  attachment: "attachments",
  text: "texts",
}

export const COLLECTION_KIND: Record<CanvasCollection, CanvasEntityKind> = {
  strokes: "stroke",
  frames: "frame",
  attachments: "attachment",
  texts: "text",
}

/** One canvas change: upsert sets `value` on entity `id`, delete omits it. */
export interface CanvasMutation {
  mutationId: string
  kind: "upsert" | "delete"
  entity: CanvasEntityKind
  id: string
  value?: Record<string, unknown>
}

export interface CanvasAcceptedBatch {
  canvasKey: string
  revision: number
  mutations: CanvasMutation[]
}

/** A canonical materialized canvas: every entity carries a stable `id`. */
export interface CanvasSnapshot {
  revision: number
  document: CanvasDocument
}

export function emptyCanvasDoc(): CanvasDocument {
  return { version: 1, frames: [], strokes: [], attachments: [], texts: [] }
}

export function canvasEntityId(): string {
  return crypto.randomUUID()
}

// --- Persistence -----------------------------------------------------------

// Legacy on-disk shape (pages / pdfEmbeds / imageEmbeds) — migrated on load.
type LegacyEmbed = { id: string; path: string; x: number; y: number; width: number }
// earlier point-text model saved notes without width/height
type LegacyText = Omit<CanvasText, "width" | "height"> & { width?: number; height?: number }
type LegacyStroke = Omit<StrokeData, "id"> & { id?: string }
type LegacyDoc = {
  version: 1
  viewport?: CanvasDocument["viewport"]
  strokes?: LegacyStroke[]
  frames?: CanvasFrame[]
  attachments?: CanvasAttachment[]
  texts?: LegacyText[]
  pages?: { id: string; x: number; y: number }[]
  pdfEmbeds?: LegacyEmbed[]
  imageEmbeds?: LegacyEmbed[]
}

function migrateStrokes(strokes: LegacyStroke[] | undefined): StrokeData[] {
  return (strokes ?? []).map((stroke) => stroke.id ? stroke as StrokeData : { ...stroke, id: canvasEntityId() })
}

/** Copy of `value` without `key`; the type stays that of `value` since the key may be a legacy field. */
export function omitKey<T extends object>(value: T, key: string): T {
  return Object.fromEntries(Object.entries(value).filter(([entryKey]) => entryKey !== key)) as T
}

function migrateAttachment(attachment: CanvasAttachment): CanvasAttachment {
  // Nested frames once carried an `embeddingScale` mode; they now always scale with their host.
  const rest = omitKey(attachment, "embeddingScale")
  return rest.kind === "canvas" && rest.canvas
    ? { ...rest, canvas: migrate(rest.canvas) }
    : rest
}

function migrateTexts(texts: LegacyText[] | undefined): CanvasText[] {
  return (texts ?? []).map((t) => ({ ...t, width: t.width ?? TEXT_DEFAULT_WIDTH, height: t.height ?? TEXT_DEFAULT_HEIGHT }))
}

function migrate(d: LegacyDoc): CanvasDocument {
  if (Array.isArray(d.frames)) {
    return {
      version: 1,
      viewport: d.viewport,
      frames: d.frames,
      strokes: migrateStrokes(d.strokes),
      attachments: (d.attachments ?? []).map(migrateAttachment),
      texts: migrateTexts(d.texts),
    }
  }
  const frames: CanvasFrame[] = []
  for (const p of d.pages ?? []) frames.push({ id: p.id, kind: "page", x: p.x, y: p.y, width: A4_WIDTH })
  for (const im of d.imageEmbeds ?? []) frames.push({ id: im.id, kind: "image", x: im.x, y: im.y, width: im.width, path: im.path })
  const attachments: CanvasAttachment[] = (d.pdfEmbeds ?? []).map((e) => ({ id: e.id, kind: "pdf", x: e.x, y: e.y, width: e.width, path: e.path }))
  return { version: 1, viewport: d.viewport, frames, strokes: migrateStrokes(d.strokes), attachments, texts: migrateTexts(d.texts) }
}

/** Parse and migrate a saved canvas, returning an empty document for invalid JSON or unknown versions. */
export function parseCanvasDoc(text: string): CanvasDocument {
  try {
    const parsed = JSON.parse(text) as LegacyDoc
    if (parsed && parsed.version === 1) return migrate(parsed)
  } catch { /* */ }
  return emptyCanvasDoc()
}

export function serializeCanvasDoc(doc: CanvasDocument): string {
  return JSON.stringify(doc)
}

// --- Diffing ---------------------------------------------------------------

interface EntityChange<T extends { id: string }> {
  id: string
  before: T | null
  after: T | null
  beforeIndex: number
  afterIndex: number
}

const jsonEqual = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/**
 * Per-entity delta between two id-keyed lists. Unchanged references and entities `same` calls equal are ignored.
 * Upserts and new entities come first in `after` order, deletions follow in `before` order.
 */
export function diffEntities<T extends { id: string }>(
  before: readonly T[],
  after: readonly T[],
  same: (a: T, b: T) => boolean = jsonEqual,
): EntityChange<T>[] {
  const previous = new Map(before.map((entity, index) => [entity.id, { entity, index }]))
  const changes: EntityChange<T>[] = []
  after.forEach((entity, afterIndex) => {
    const prior = previous.get(entity.id)
    previous.delete(entity.id)
    if (prior && (prior.entity === entity || same(prior.entity, entity))) return
    changes.push({ id: entity.id, before: prior?.entity ?? null, after: entity, beforeIndex: prior?.index ?? -1, afterIndex })
  })
  for (const [id, { entity, index }] of previous) {
    changes.push({ id, before: entity, after: null, beforeIndex: index, afterIndex: -1 })
  }
  return changes
}

// --- History ---------------------------------------------------------------

interface CanvasHistoryChange extends EntityChange<CanvasEntity> {
  collection: CanvasCollection
}

/** Page-only attachment edits are navigation, never a conflicting artwork edit. */
function sameForHistory(collection: CanvasCollection, a: CanvasEntity | null, b: CanvasEntity | null): boolean {
  if (collection === "attachments" && a && b) {
    const withoutPageA = omitKey(a as CanvasAttachment, "page")
    const withoutPageB = omitKey(b as CanvasAttachment, "page")
    return jsonEqual(withoutPageA, withoutPageB)
  }
  return jsonEqual(a, b)
}

/** Per-entity delta between two docs, matched by id. */
export function historyChanges(before: CanvasDocument, after: CanvasDocument): CanvasHistoryChange[] {
  return CANVAS_COLLECTIONS.flatMap((collection) => {
    const beforeItems: CanvasEntity[] = before[collection]
    const afterItems: CanvasEntity[] = after[collection]
    if (beforeItems === afterItems) return []
    return diffEntities(beforeItems, afterItems, (a, b) => sameForHistory(collection, a, b)).map((change) => ({ ...change, collection }))
  })
}

/**
 * Replay a recorded delta onto the live doc. A change is skipped when its entity was edited since
 * (no longer equals the expected state), and removed entities return at their recorded index.
 */
export function restoreHistory(doc: CanvasDocument, changes: CanvasHistoryChange[], direction: "undo" | "redo"): CanvasDocument {
  const undoing = direction === "undo"
  let restored: CanvasDocument | null = null
  for (const collection of CANVAS_COLLECTIONS) {
    const collectionChanges = changes.filter((change) => change.collection === collection)
    if (collectionChanges.length === 0) continue
    const changesById = new Map(collectionChanges.map((change) => [change.id, change]))
    const current: CanvasEntity[] = doc[collection]
    const items = current.flatMap((entity) => {
      const change = changesById.get(entity.id)
      if (!change) return [entity]
      const expected = undoing ? change.after : change.before
      const replacement = undoing ? change.before : change.after
      if (!sameForHistory(collection, entity, expected)) return [entity]
      if (!replacement) return []
      if (collection === "attachments") {
        const page = (entity as CanvasAttachment).page
        return [{ ...(replacement as CanvasAttachment), ...(page === undefined ? {} : { page }) }]
      }
      return [replacement]
    })
    const indexOf = (change: CanvasHistoryChange) => undoing ? change.beforeIndex : change.afterIndex
    for (const change of collectionChanges
      .filter((change) => (undoing ? change.after : change.before) === null)
      .sort((a, b) => indexOf(a) - indexOf(b))) {
      const replacement = undoing ? change.before : change.after
      if (!replacement || items.some((entity) => entity.id === change.id)) continue
      items.splice(Math.max(0, Math.min(indexOf(change), items.length)), 0, replacement)
    }
    if (items.length === current.length && items.every((entity, index) => entity === current[index])) continue
    restored = { ...(restored ?? doc), [collection]: items }
  }
  return restored ?? doc
}
