"use client"

import { FileText, Image as ImageIcon, type LucideIcon } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { rewriteImages } from "@/lib/backend/urls"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import { extOf, kindOf } from "@/lib/paths"
import { fenceCode } from "@gigachad/chat-ui"

export const PDF_MIME = "application/pdf"

/** Store an edited attachment text. A rejection means the source could not be saved and the edit did not happen. */
export type ContentChange = (newContent: string) => void | Promise<void>

/** Whether the attachment's own text is markdown, so rendering it may edit it (task-list checkboxes). */
export function isMarkdownAttachment(attachment: Attachment): boolean {
  return kindOf(attachment.name) === "markdown"
}

/** Markdown to render: a PDF's extraction, a markdown file as itself, other text as a code block highlighted by extension. */
export function renderableMarkdown(attachment: Attachment): string | null {
  if (attachment.mime === PDF_MIME) return attachment.parsedMd ?? null
  if (attachment.content == null) return null
  return isMarkdownAttachment(attachment) ? attachment.content : fenceCode(extOf(attachment.name), attachment.content)
}

/** Icon by mime: image, PDF, else `generic`. `size` is the Tailwind height/width pair; text files are muted unless `genericMuted` is false. */
export function AttachmentIcon({ mime, generic, size, genericMuted = true }: { mime: string; generic: LucideIcon; size: string; genericMuted?: boolean }) {
  const isImage = mime.startsWith("image/")
  const isPdf = mime === PDF_MIME
  const Icon = isImage ? ImageIcon : isPdf ? FileText : generic
  const tone = isImage || isPdf || !genericMuted ? "text-ink" : "text-ink-muted"
  return <Icon className={`${size} ${tone} shrink-0`} />
}

/** Rendered markdown typeset with LaTeX, from `renderableMarkdown`. Task-list checkboxes call `onContentChange`. */
export function RenderedAttachment({ attachment, markdown, onContentChange }: {
  attachment: Attachment
  markdown: string
  onContentChange?: ContentChange
}) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
      <div className="mx-auto max-w-3xl">
        <LaTeXMarkdown content={rewriteImages(markdown, attachment.name)} onContentChange={onContentChange && ((next) => { void Promise.resolve(onContentChange(next)).catch(() => {}) })} />
      </div>
    </div>
  )
}

/** Image or PDF as itself, nothing for other kinds. */
export function RawAttachment({ attachment }: { attachment: Attachment }) {
  if (attachment.mime === PDF_MIME) {
    return <div className="min-h-0 flex-1"><LazyPdfViewer url={attachment.url} /></div>
  }
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-6">
      {/* eslint-disable-next-line @next/next/no-img-element -- attachment URLs are blob/data/backend URLs, not optimizable by next/image under Tauri static export */}
      <img src={attachment.url} alt={attachment.name} className="max-h-full max-w-full rounded-lg object-contain" />
    </div>
  )
}
