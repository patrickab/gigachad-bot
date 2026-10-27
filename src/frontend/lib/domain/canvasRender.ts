import { FALLBACK_ASPECT, parseCanvasDoc, type CanvasAttachment, type CanvasDocument, type CanvasFrame } from "./canvasDocument"
import { renderCanvasToJpeg, type EmbedRect } from "@/lib/drawing"
import { fileViewerRawUrl } from "@/lib/backend/urls"

// PDF attachments are not drawn into the rendered canvas image
const EXCLUDED_ATTACHMENT_KINDS: Partial<Record<CanvasAttachment["kind"], true>> = { pdf: true }

/** Embed rects of a canvas's image frames, `aspectOf` giving each one's height / width. */
export function imageFrameEmbeds(frames: readonly CanvasFrame[], aspectOf: (frame: CanvasFrame) => number): EmbedRect[] {
  return frames
    .filter((frame) => frame.kind === "image" && frame.path)
    .map((frame) => ({ url: fileViewerRawUrl(frame.path!), x: frame.x, y: frame.y, width: frame.width, aspect: aspectOf(frame) }))
}

function renderableImages({ frames, attachments }: Pick<CanvasDocument, "frames" | "attachments">): EmbedRect[] {
  return [
    ...imageFrameEmbeds(frames, () => FALLBACK_ASPECT),
    ...attachments
      .filter((attachment) => attachment.path && !EXCLUDED_ATTACHMENT_KINDS[attachment.kind])
      .map(({ path, x, y, width, height }) => ({ url: fileViewerRawUrl(path!), x, y, width, aspect: height ? height / width : FALLBACK_ASPECT })),
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
