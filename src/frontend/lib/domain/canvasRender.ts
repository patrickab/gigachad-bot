import { parseCanvasDoc, type CanvasDocument, type CanvasFrame } from "./canvasDocument"
import { attachmentTraits } from "./canvasAttachments"
import { renderCanvasToJpeg, type EmbedRect } from "@/lib/canvasExport"
import { fileViewerRawUrl } from "@/lib/backend/urls"

/** Embed rects of a canvas's image frames. */
export function imageFrameEmbeds(frames: readonly CanvasFrame[]): EmbedRect[] {
  return frames
    .filter((frame) => frame.kind === "image" && frame.path)
    .map((frame) => ({ url: fileViewerRawUrl(frame.path!), x: frame.x, y: frame.y, width: frame.width }))
}

function renderableImages({ frames, attachments }: Pick<CanvasDocument, "frames" | "attachments">): EmbedRect[] {
  return [
    ...imageFrameEmbeds(frames),
    ...attachments
      .filter((attachment) => attachment.path && !attachmentTraits[attachment.kind].excludedFromExport)
      .map(({ path, x, y, width }) => ({ url: fileViewerRawUrl(path!), x, y, width })),
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
