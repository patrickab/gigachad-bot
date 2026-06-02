"use client"

import { memo, useCallback, useRef, useState } from "react"
import { PanelRightClose, FileText, Image as ImageIcon, File } from "lucide-react"
import type { Attachment } from "@/lib/types"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import dynamic from "next/dynamic"
const PdfViewer = dynamic(() => import("./PdfViewer").then(m => ({ default: m.PdfViewer })), { ssr: false })
import { API_BASE } from "@/lib/config"

interface AttachmentSidebarProps {
  attachment: Attachment | null
  chatId: string
  onClose: () => void
  isPdf?: boolean
}

function rewriteImages(content: string, chatId: string): string {
  const origin = new URL(API_BASE).origin
  return content.replace(/\(images\/([^)]+)\)/g, `(${origin}/chat-uploads/${chatId}/images/$1)`)
}

function AttachmentIcon({ mime }: { mime: string }) {
  if (mime.startsWith("image/")) return <ImageIcon className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
  if (mime === "application/pdf") return <FileText className="h-3.5 w-3.5 text-blue-400 shrink-0" />
  return <File className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
}

function AttachmentViewer({ attachment, chatId }: { attachment: Attachment; chatId: string }) {
  if (attachment.mime.startsWith("image/")) {
    return (
      <div className="p-2">
        <img src={attachment.url} alt={attachment.name} className="max-w-full rounded border border-zinc-800" />
      </div>
    )
  }

  if (attachment.mime === "application/pdf") {
    const parsedContent = attachment.parsedMd
    return (
      <div className="flex flex-col h-full">
        <PdfViewer url={attachment.url} />
        {parsedContent && (
          <div className="border-t border-zinc-800 p-2 max-h-[40%] overflow-y-auto">
            <LaTeXMarkdown content={rewriteImages(parsedContent, chatId)} />
          </div>
        )}
      </div>
    )
  }

  const content = attachment.parsedMd ?? attachment.content
  if (content) {
    const rewritten = rewriteImages(content, chatId)
    return (
      <div className="p-2 max-h-[500px] overflow-y-auto">
        <LaTeXMarkdown content={rewritten} />
      </div>
    )
  }

  return <p className="p-2 text-xs text-zinc-500">No preview available</p>
}

function AttachmentSidebarInner({ attachment, chatId, onClose, isPdf }: AttachmentSidebarProps) {
  if (!attachment) return null

  const [width, setWidth] = useState(isPdf ? 50 : 288)
  const dragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    const startX = e.clientX
    const startWidth = width

    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return
      const parent = containerRef.current?.parentElement
      if (!parent) return
      const totalWidth = parent.clientWidth
      const isPct = typeof startWidth === "number" && isPdf
      if (isPct) {
        const deltaPct = ((startX - ev.clientX) / totalWidth) * 100
        const newPct = Math.min(70, Math.max(20, (startWidth as number) + deltaPct))
        setWidth(Math.round(newPct))
      } else {
        const newPx = startWidth + (startX - ev.clientX)
        setWidth(Math.min(600, Math.max(200, newPx)))
      }
    }

    const onMouseUp = () => {
      dragging.current = false
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }

    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [width, isPdf])

  const isPct = isPdf

  return (
    <div
      ref={containerRef}
      className="shrink-0 border-l border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden relative"
      style={isPct ? { width: `${width}%` } : { width: `${width}px` }}
    >
      <div
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-zinc-700 active:bg-blue-500/50 transition-colors z-10"
        onMouseDown={handleMouseDown}
      />
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800/50">
        <AttachmentIcon mime={attachment.mime} />
        <span className="text-[11px] font-medium text-zinc-300 truncate flex-1">{attachment.name}</span>
        <button onClick={onClose} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors" title="Close">
          <PanelRightClose className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        <AttachmentViewer attachment={attachment} chatId={chatId} />
      </div>
    </div>
  )
}

export const AttachmentSidebar = memo(AttachmentSidebarInner)