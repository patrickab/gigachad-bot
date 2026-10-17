"use client"

import { memo, useState } from "react"
import { Image as ImageIcon, File } from "lucide-react"
import { cn } from "@/lib/utils"
import { isImageAttachment } from "@/lib/attachments"
import type { Attachment } from "@/lib/types"
import { AttachmentIcon } from "./AttachmentView"

interface MessageAttachmentsProps {
  attachments: Attachment[]
  onClick: (attachment: Attachment) => void
}

/** Image preview filling its tile, an icon when the upload is gone or unreadable. */
export function ImageThumbnail({ attachment }: { attachment: Attachment }) {
  const [failed, setFailed] = useState(false)
  if (failed || !attachment.url) return <ImageIcon className="h-4 w-4 text-ink-faint" />
  return (
    <img
      src={attachment.url}
      alt={attachment.name}
      loading="lazy"
      draggable={false}
      onError={() => setFailed(true)}
      className="h-full w-full object-cover"
    />
  )
}

function MessageAttachmentsInner({ attachments, onClick }: MessageAttachmentsProps) {
  if (attachments.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {attachments.map((att) => isImageAttachment(att) ? (
        <button
          key={att.name}
          aria-label={att.name}
          onClick={() => onClick(att)}
          className={cn(
            "flex h-20 w-20 items-center justify-center overflow-hidden rounded-lg border bg-surface-elevated/30 transition-opacity",
            att.active ? "border-divider-strong/50" : "border-divider/40 opacity-50 hover:opacity-80",
          )}
        >
          <ImageThumbnail attachment={att} />
        </button>
      ) : (
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
          <AttachmentIcon mime={att.mime} generic={File} size="h-3 w-3" />
          <span className="max-w-[140px] truncate">{att.name}</span>
        </button>
      ))}
    </div>
  )
}

export const MessageAttachments = memo(MessageAttachmentsInner)