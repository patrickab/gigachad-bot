"use client"

import { memo, useCallback, useEffect, useRef, useState } from "react"
import { ChevronDown, ChevronRight, PanelRightClose, FileText, Image as ImageIcon, File, X } from "lucide-react"
import type { Attachment } from "@/lib/types"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import dynamic from "next/dynamic"
const PdfViewer = dynamic(() => import("./PdfViewer").then(m => ({ default: m.PdfViewer })), { ssr: false })
import { API_BASE } from "@/lib/config"

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
        <div className="flex-1 min-h-0">
          <PdfViewer url={attachment.url} />
        </div>
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

function Collapse({ open, children }: { open: boolean; children: React.ReactNode }) {
  return (
    <div className={`grid transition-[grid-template-rows] duration-200 ease-in-out ${open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"}`}>
      <div className="overflow-hidden">
        {children}
      </div>
    </div>
  )
}

interface ExpandedEntry {
  messageIndex: number
  attachmentName: string
}

interface ContextSidebarProps {
  chatId: string
  messages: {
    messageIndex: number
    attachment: Attachment
  }[]
  expandedEntries: ExpandedEntry[]
  onToggle: (messageIndex: number, attachmentName: string) => void
  onRemoveAttachment: (messageIndex: number, attachmentName: string) => void
  onClose: () => void
}

function ContextSidebarInner({
  chatId,
  messages,
  expandedEntries,
  onToggle,
  onRemoveAttachment,
  onClose,
}: ContextSidebarProps) {
  const expandedPdf = expandedEntries.some(e => {
    const att = messages.find(m => m.messageIndex === e.messageIndex && m.attachment.name === e.attachmentName)
    return att?.attachment.mime === "application/pdf"
  })

  const [width, setWidth] = useState(expandedPdf ? 50 : 320)
  const [mounted, setMounted] = useState(false)
  const dragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const prevExpandedPdf = useRef(!expandedPdf)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (expandedPdf && !prevExpandedPdf.current) setWidth(50)
    if (!expandedPdf && prevExpandedPdf.current) setWidth(320)
    prevExpandedPdf.current = expandedPdf
  }, [expandedPdf])

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
      if (expandedPdf) {
        const deltaPct = ((startX - ev.clientX) / totalWidth) * 100
        setWidth(Math.min(70, Math.max(20, startWidth + deltaPct)))
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
  }, [width, expandedPdf])

  const isExpanded = (mi: number, name: string) =>
    expandedEntries.some(e => e.messageIndex === mi && e.attachmentName === name)

  const isPct = expandedPdf

  return (
    <div
      ref={containerRef}
      className={`shrink-0 border-l border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden relative transition-[width] duration-200 ease-in-out ${mounted ? "" : "!transition-none"}`}
      style={isPct ? { width: `${width}%` } : { width: `${width}px` }}
    >
      <div
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-zinc-700 active:bg-blue-500/50 transition-colors z-10"
        onMouseDown={handleMouseDown}
      />
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800/50">
        <FileText className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
        <span className="text-[11px] font-medium text-zinc-300 truncate flex-1">Context</span>
        <span className="text-[10px] text-zinc-600">{messages.length}</span>
        <button onClick={onClose} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors" title="Close">
          <PanelRightClose className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {messages.map(({ messageIndex: mi, attachment: att }) => {
          const expanded = isExpanded(mi, att.name)
          return (
            <div key={`${mi}-${att.name}`} className={`border-b border-zinc-800/50 ${expanded && att.mime === "application/pdf" ? "flex flex-col flex-1" : ""}`}>
              <div
                onClick={() => onToggle(mi, att.name)}
                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-zinc-900/50 transition-colors cursor-pointer"
              >
                {expanded ? (
                  <ChevronDown className="h-3 w-3 text-zinc-500 shrink-0" />
                ) : (
                  <ChevronRight className="h-3 w-3 text-zinc-500 shrink-0" />
                )}
                <AttachmentIcon mime={att.mime} />
                <span className="text-[11px] font-medium text-zinc-300 truncate flex-1 text-left">{att.name}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); onRemoveAttachment(mi, att.name) }}
                  className="rounded p-0.5 text-zinc-600 hover:text-red-400 hover:bg-zinc-800 transition-colors shrink-0"
                  title="Remove"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
              <Collapse open={expanded}>
                {expandedPdf ? (
                  <div className="flex-1 min-h-0 overflow-y-auto">
                    <AttachmentViewer attachment={att} chatId={chatId} />
                  </div>
                ) : (
                  <div className="max-h-[60vh] overflow-y-auto">
                    <AttachmentViewer attachment={att} chatId={chatId} />
                  </div>
                )}
              </Collapse>
            </div>
          )
        })}
        {messages.length === 0 && (
          <div className="flex items-center justify-center py-8 text-xs text-zinc-600">
            No attachments
          </div>
        )}
      </div>
    </div>
  )
}

export const ContextSidebar = memo(ContextSidebarInner)

export type { ExpandedEntry }