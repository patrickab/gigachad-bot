import { CANVAS_ASPECT, FALLBACK_ASPECT, type AttachmentKind, type CanvasAttachment } from "./canvasDocument"

const GRAPH_SIZE = { width: 720, height: 480 }

interface AttachmentTraits {
  /** Window title when the attachment has none of its own (and, with `namedAfterFile`, no file name either). */
  name: string
  namedAfterFile: boolean
  /** The window owns its height: a resize grows it, instead of following the content's aspect. */
  fixedHeight: boolean
  /** Size of a new window: a stored height, or the height / width aspect it is centred with. */
  defaultSize: { width: number; height: number } | { width: number; aspect: number }
  /** Retitling the window renames its backing file, not just the window. */
  renamesFile: boolean
  /** Left out of the rendered canvas image. */
  excludedFromExport: boolean
}

/** Everything that differs between attachment kinds, in one place. */
export const attachmentTraits: Record<AttachmentKind, AttachmentTraits> = {
  pdf: { name: "PDF", namedAfterFile: true, fixedHeight: false, defaultSize: { width: 500, aspect: FALLBACK_ASPECT }, renamesFile: true, excludedFromExport: true },
  canvas: { name: "Canvas", namedAfterFile: true, fixedHeight: false, defaultSize: { width: 600, aspect: CANVAS_ASPECT }, renamesFile: true, excludedFromExport: false },
  // Architecture sources have no rename, so renaming an architecture window only retitles the window.
  "architecture-graph": { name: "Architecture", namedAfterFile: false, fixedHeight: true, defaultSize: GRAPH_SIZE, renamesFile: false, excludedFromExport: false },
  document: { name: "Document", namedAfterFile: true, fixedHeight: true, defaultSize: GRAPH_SIZE, renamesFile: true, excludedFromExport: false },
}

/** Title shown on an attachment window; `shownName` is the display name its project gives the file, if any. */
export function attachmentName(attachment: CanvasAttachment, shownName?: string): string {
  const traits = attachmentTraits[attachment.kind]
  return attachment.title ?? (traits.namedAfterFile ? shownName ?? attachment.path?.split("/").pop() : undefined) ?? traits.name
}
