import type { Attachment } from "@/lib/domain/types"
import { piMediaUrl } from "@/lib/backend/urls"
import type { ToolImage, ToolRun } from "./types"

const EXTENSIONS: Record<string, string> = { "image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp" }

/** The image blocks of a tool run's result, in order. */
export function toolImagesOf(run: ToolRun | undefined): ToolImage[] {
  return (run?.result?.content ?? []).filter((block): block is ToolImage => block.type === "image")
}

/**
 * The images of `toolName`'s result as chat image attachments: a stored image links `/api/pi/media/<hash>` with its thumbnail,
 * an inline one (pi-host could not store it) its data URL. Names are unique per image (the attachment's identity), so the
 * same image twice shows once.
 */
export function toolImageAttachments(toolName: string, images: ToolImage[]): Attachment[] {
  const seen = new Set<string>()
  const attachments: Attachment[] = []
  for (const image of images) {
    const id = "hash" in image ? image.hash : image.data
    if (seen.has(id)) continue
    seen.add(id)
    const extension = EXTENSIONS[image.mimeType] ?? "png"
    const stored = "hash" in image
    attachments.push({
      name: `${toolName}-${stored ? image.hash.slice(0, 8) : String(attachments.length + 1)}.${extension}`,
      mime: image.mimeType,
      url: stored ? piMediaUrl(image.hash) : `data:${image.mimeType};base64,${image.data}`,
      ...(stored ? { thumbUrl: piMediaUrl(image.hash, true) } : {}),
      active: true,
    })
  }
  return attachments
}
