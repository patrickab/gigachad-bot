"use client"

import { memo, useState } from "react"
import { Image as ImageIcon, File } from "lucide-react"
import { cn } from "@/lib/utils"
import { isImageAttachment } from "@/features/chat/attachments"
import type { Attachment } from "@/lib/domain/types"
import { AttachmentIcon } from "@/features/chat/preview/AttachmentView"
import { useAttachmentHoverCard } from "@/features/chat/preview/AttachmentHoverCard"

interface MessageAttachmentsProps {
  attachments: Attachment[]
  onClick: (attachment: Attachment) => void
}

/** Image preview filling its tile, an icon when the upload is gone or unreadable. */
function ImageThumbnail({ attachment }: { attachment: Attachment }) {
  const [failed, setFailed] = useState(false)
  if (failed || !attachment.url) return <ImageIcon className="h-4 w-4 text-ink-faint" />
  return (
    // eslint-disable-next-line @next/next/no-img-element -- attachment URLs are blob/data/backend URLs, not optimizable by next/image under Tauri static export
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

// The card renders beside its anchor, not inside: React events bubble through portals, so a press in the card would reach the button.
function MessageAttachment({ attachment, onClick }: { attachment: Attachment; onClick: (attachment: Attachment) => void }) {
  const hover = useAttachmentHoverCard(attachment)
  return (
    <>
      {isImageAttachment(attachment) ? (
        <button
          aria-label={attachment.name}
          onClick={() => onClick(attachment)}
          {...hover.anchorProps}
          className={cn(
            "flex h-20 w-20 items-center justify-center overflow-hidden rounded-lg border bg-surface-elevated/30 transition-opacity",
            attachment.active ? "border-divider-strong/50" : "border-divider/40 opacity-50 hover:opacity-80",
          )}
        >
          <ImageThumbnail attachment={attachment} />
        </button>
      ) : (
        <button
          onClick={() => onClick(attachment)}
          {...hover.anchorProps}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs transition-colors",
            attachment.active
              ? "border-divider-strong/50 bg-surface-elevated/30 text-ink-muted hover:bg-hover hover:text-ink"
              : "border-divider/40 bg-surface-elevated/15 text-ink-faint hover:bg-hover hover:text-ink-subtle",
          )}
        >
          <AttachmentIcon mime={attachment.mime} generic={File} size="h-3 w-3" />
          <span className="max-w-[140px] truncate">{attachment.name}</span>
        </button>
      )}
      {hover.card}
    </>
  )
}

function MessageAttachmentsInner({ attachments, onClick }: MessageAttachmentsProps) {
  if (attachments.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {attachments.map((attachment) => <MessageAttachment key={attachment.name} attachment={attachment} onClick={onClick} />)}
    </div>
  )
}

export const MessageAttachments = memo(MessageAttachmentsInner)