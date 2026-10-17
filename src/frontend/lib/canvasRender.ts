import { FALLBACK_ASPECT, parseCanvasDoc, type CanvasAttachment, type CanvasDocument, type CanvasFrame } from "./canvasDocument"
import { renderCanvasToJpeg, type EmbedRect } from "./drawing"
import { fileViewerRawUrl } from "./urls"

// PDF attachments are not drawn into the rendered canvas image
const EXCLUDED_ATTACHMENT_KINDS: Partial<Record<CanvasAttachment["kind"], true>> = { pdf: true }

function imageEmbed(path: string, { x, y, width, height }: Pick<CanvasFrame | CanvasAttachment, "x" | "y" | "width"> & { height?: number }): EmbedRect {
  return { url: fileViewerRawUrl(path), x, y, width, aspect: height ? height / width : FALLBACK_ASPECT }
}

function renderableImages({ frames, attachments }: Pick<CanvasDocument, "frames" | "attachments">): EmbedRect[] {
  return [
    ...frames.filter((frame) => frame.kind === "image" && frame.path).map((frame) => imageEmbed(frame.path!, frame)),
    ...attachments
      .filter((attachment) => attachment.path && !EXCLUDED_ATTACHMENT_KINDS[attachment.kind])
      .map((attachment) => imageEmbed(attachment.path!, attachment)),
  ]
}

/** Rasterize saved canvas text into a JPEG, or null when it holds no ink, text or images. */
export async function renderCanvasTextToJpeg(text: string): Promise<Blob | null> {
  if (!text.trim()) return null
  const doc = parseCanvasDoc(text)
  const images = renderableImages(doc)
  if (!doc.strokes.length && !doc.texts.length && images.length === 0) return null
  return renderCanvasToJpeg(doc.strokes, 20, images, doc.texts)
}
