"use client"

import { useState } from "react"
import { ChevronDown, ChevronRight, FileText, Image as ImageIcon, type LucideIcon } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { rewriteImages } from "@/lib/backend/urls"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"

const PDF_MIME = "application/pdf"

/** Icon by mime: image, PDF, else `generic`. `size` is the Tailwind height/width pair; text files are muted unless `genericMuted` is false. */
export function AttachmentIcon({ mime, generic, size, genericMuted = true }: { mime: string; generic: LucideIcon; size: string; genericMuted?: boolean }) {
  const isImage = mime.startsWith("image/")
  const isPdf = mime === PDF_MIME
  const Icon = isImage ? ImageIcon : isPdf ? FileText : generic
  const tone = isImage || isPdf || !genericMuted ? "text-ink" : "text-ink-muted"
  return <Icon className={`${size} ${tone} shrink-0`} />
}

function PdfWithMarkdown({ attachment, pdfWide, onTogglePdfWide }: { attachment: Attachment; pdfWide?: boolean; onTogglePdfWide?: () => void }) {
  const [mdOpen, setMdOpen] = useState(false)
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0">
        <LazyPdfViewer url={attachment.url} isWide={pdfWide} onToggleWide={onTogglePdfWide} />
      </div>
      {attachment.parsedMd && (
        <div className="border-t border-divider">
          <button
            onClick={() => setMdOpen((o) => !o)}
            className="w-full flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-subtle hover:text-ink hover:bg-surface/50 transition-colors"
          >
            {mdOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Markdown
          </button>
          {mdOpen && (
            <div className="p-2 max-h-[40vh] overflow-y-auto border-t border-divider">
              <LaTeXMarkdown content={rewriteImages(attachment.parsedMd, attachment.name)} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

interface AttachmentViewProps {
  attachment: Attachment
  /** `panel` sits inline in the context panel; `modal` fills the full-screen preview. */
  variant: "panel" | "modal"
  /** Panel only: lets task-list checkboxes in the rendered markdown edit the content. */
  onContentChange?: (newContent: string) => void
  pdfWide?: boolean
  onTogglePdfWide?: () => void
}

function AttachmentBody({ attachment, variant, onContentChange, pdfWide, onTogglePdfWide }: AttachmentViewProps) {
  const modal = variant === "modal"

  if (attachment.mime.startsWith("image/")) {
    return (
      <div className={modal ? "flex items-center justify-center p-6 overflow-auto max-h-[80vh]" : "p-2"}>
        {/* eslint-disable-next-line @next/next/no-img-element -- attachment URLs are blob/data/backend URLs, not optimizable by next/image under Tauri static export */}
        <img
          src={attachment.url}
          alt={attachment.name}
          className={modal ? "max-w-full max-h-[80vh] rounded-lg shadow-[var(--shadow-xl)] object-contain" : "max-w-full rounded border border-divider"}
        />
      </div>
    )
  }

  if (attachment.mime === PDF_MIME) {
    return modal ? (
      <div className="flex-1 min-h-0 overflow-hidden">
        <LazyPdfViewer url={attachment.url} />
      </div>
    ) : (
      <PdfWithMarkdown attachment={attachment} pdfWide={pdfWide} onTogglePdfWide={onTogglePdfWide} />
    )
  }

  const content = attachment.parsedMd ?? attachment.content
  if (content) {
    const markdown = <LaTeXMarkdown content={rewriteImages(content, attachment.name)} onContentChange={modal ? undefined : onContentChange} />
    return modal ? (
      <div className="flex-1 min-h-0 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto">{markdown}</div>
      </div>
    ) : (
      <div className="p-2 max-h-[500px] overflow-y-auto">{markdown}</div>
    )
  }

  return modal ? (
    <div className="flex-1 flex items-center justify-center">
      <p className="text-sm text-ink-subtle">No preview available</p>
    </div>
  ) : (
    <p className="p-2 text-xs text-ink-subtle">No preview available</p>
  )
}

/** Previews an attachment by mime: image, PDF, then parsed markdown or raw text, else a placeholder. */
export function AttachmentView(props: AttachmentViewProps) {
  if (props.variant === "modal") return <AttachmentBody {...props} />
  return (
    <div className={props.attachment.mime === PDF_MIME ? "h-[60vh]" : "max-h-[60vh] overflow-y-auto"}>
      <AttachmentBody {...props} />
    </div>
  )
}
