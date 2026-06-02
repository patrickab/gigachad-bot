"use client"

import { memo } from "react"
import { FileText, Image as ImageIcon, File } from "lucide-react"
import type { Attachment } from "@/lib/types"

interface MessageAttachmentsProps {
  attachments: Attachment[]
  onClick: (attachment: Attachment) => void
}

function AttachmentIcon({ mime }: { mime: string }) {
  if (mime.startsWith("image/")) return <ImageIcon className="h-3 w-3 text-emerald-400 shrink-0" />
  if (mime === "application/pdf") return <FileText className="h-3 w-3 text-blue-400 shrink-0" />
  return <File className="h-3 w-3 text-zinc-400 shrink-0" />
}

function MessageAttachmentsInner({ attachments, onClick }: MessageAttachmentsProps) {
  if (attachments.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {attachments.map((att) => (
        <button
          key={att.name}
          onClick={() => onClick(att)}
          className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-700/50 bg-zinc-800/30 px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-700/50 hover:text-zinc-200 transition-colors"
        >
          <AttachmentIcon mime={att.mime} />
          <span className="max-w-[140px] truncate">{att.name}</span>
        </button>
      ))}
    </div>
  )
}

export const MessageAttachments = memo(MessageAttachmentsInner)