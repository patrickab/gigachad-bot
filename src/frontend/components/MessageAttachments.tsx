"use client"

import { memo } from "react"
import { FileText, Image as ImageIcon, File } from "lucide-react"
import { cn } from "@/lib/utils"
import type { Attachment } from "@/lib/types"

interface MessageAttachmentsProps {
  attachments: Attachment[]
  onClick: (attachment: Attachment) => void
}

function AttachmentIcon({ mime }: { mime: string }) {
  if (mime.startsWith("image/")) return <ImageIcon className="h-3 w-3 text-ink shrink-0" />
  if (mime === "application/pdf") return <FileText className="h-3 w-3 text-ink shrink-0" />
  return <File className="h-3 w-3 text-ink-muted shrink-0" />
}

function MessageAttachmentsInner({ attachments, onClick }: MessageAttachmentsProps) {
  if (attachments.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {attachments.map((att) => (
        <button
          key={att.name}
          onClick={() => onClick(att)}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs transition-colors",
            att.active
              ? "border-divider-strong/50 bg-surface-elevated/30 text-ink-muted hover:bg-hover hover:text-ink"
              : "border-divider/40 bg-surface-elevated/15 text-ink-faint hover:bg-hover hover:text-ink-subtle",
          )}
        >
          <AttachmentIcon mime={att.mime} />
          <span className="max-w-[140px] truncate">{att.name}</span>
        </button>
      ))}
    </div>
  )
}

export const MessageAttachments = memo(MessageAttachmentsInner)