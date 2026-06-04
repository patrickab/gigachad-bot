"use client"

import { useState } from "react"
import { FileText, Image as ImageIcon, Globe, X, ChevronDown, ChevronRight } from "lucide-react"
import dynamic from "next/dynamic"
import type { Attachment, MorphicSearchResult } from "@/lib/types"
import { API_BASE } from "@/lib/config"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import type { ChatSidebarElementConfig } from "@/components/ChatSidebar"

const PdfViewer = dynamic(() => import("./PdfViewer").then((m) => ({ default: m.PdfViewer })), { ssr: false })

function rewriteImages(content: string, chatId: string): string {
  const origin = new URL(API_BASE).origin
  return content.replace(/\(images\/([^)]+)\)/g, `(${origin}/chat-uploads/${chatId}/images/$1)`)
}

function AttachmentIcon({ mime }: { mime: string }) {
  if (mime.startsWith("image/")) return <ImageIcon className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
  if (mime === "application/pdf") return <FileText className="h-3.5 w-3.5 text-blue-400 shrink-0" />
  return <FileText className="h-3.5 w-3.5 text-zinc-400 shrink-0" />
}

function AttachmentViewer({ attachment, chatId, onContentChange }: { attachment: Attachment; chatId: string; onContentChange?: (newContent: string) => void }) {
  if (attachment.mime.startsWith("image/")) {
    return (
      <div className="p-2">
        <img src={attachment.url} alt={attachment.name} className="max-w-full rounded border border-zinc-800" />
      </div>
    )
  }

  if (attachment.mime === "application/pdf") {
    const parsedContent = attachment.parsedMd
    return <PdfAttachmentViewer url={attachment.url} parsedContent={parsedContent} chatId={chatId} />
  }

  const content = attachment.parsedMd ?? attachment.content
  if (content) {
    const rewritten = rewriteImages(content, chatId)
    return (
      <div className="p-2 max-h-[500px] overflow-y-auto">
        <LaTeXMarkdown content={rewritten} onContentChange={onContentChange} />
      </div>
    )
  }

  return <p className="p-2 text-xs text-zinc-500">No preview available</p>
}

function PdfAttachmentViewer({ url, parsedContent, chatId }: { url: string; parsedContent?: string; chatId: string }) {
  const [mdOpen, setMdOpen] = useState(false)
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0">
        <PdfViewer url={url} />
      </div>
      {parsedContent && (
        <div className="border-t border-zinc-800">
          <button
            onClick={() => setMdOpen((o) => !o)}
            className="w-full flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/50 transition-colors"
          >
            {mdOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Markdown
          </button>
          {mdOpen && (
            <div className="p-2 max-h-[40vh] overflow-y-auto border-t border-zinc-800">
              <LaTeXMarkdown content={rewriteImages(parsedContent, chatId)} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ContextBody({
  chatId,
  allAttachments,
  expandedEntries,
  onToggleAttachment,
  onRemoveAttachment,
  onAttachmentContentChange,
}: {
  chatId: string
  allAttachments: { messageIndex: number; attachment: Attachment }[]
  expandedEntries: { messageIndex: number; attachmentName: string }[]
  onToggleAttachment: (messageIndex: number, attachmentName: string) => void
  onRemoveAttachment: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
}) {
  const isExpanded = (mi: number, name: string) =>
    expandedEntries.some((e) => e.messageIndex === mi && e.attachmentName === name)

  if (allAttachments.length === 0) {
    return (
      <div className="flex items-center justify-center py-6 text-xs text-zinc-600">
        No attachments
      </div>
    )
  }

  return (
    <div>
      {allAttachments.map(({ messageIndex: mi, attachment: att }) => {
        const expanded = isExpanded(mi, att.name)
        return (
          <div key={`${mi}-${att.name}`} className="border-b border-zinc-800/50">
            <div
              onClick={() => onToggleAttachment(mi, att.name)}
              className="w-full flex items-center gap-2 px-3 py-2 hover:bg-zinc-900/50 transition-colors cursor-pointer"
            >
              <AttachmentIcon mime={att.mime} />
              <span className="text-[11px] font-medium text-zinc-300 truncate flex-1 text-left">{att.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onRemoveAttachment(mi, att.name)
                }}
                className="rounded p-0.5 text-zinc-600 hover:text-red-400 hover:bg-zinc-800 transition-colors shrink-0"
                title="Remove"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
            {expanded && (
              <div className="max-h-[60vh] overflow-y-auto">
                <AttachmentViewer attachment={att} chatId={chatId} onContentChange={onAttachmentContentChange ? (nc: string) => onAttachmentContentChange(mi, att.name, nc) : undefined} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function SourcesBody({ result }: { result: MorphicSearchResult }) {
  const seen = new Set<string>()
  const sources = result.sources.filter((s) => !seen.has(s.url) && seen.add(s.url))
  const images = [...new Set(result.images)]

  if (sources.length === 0 && images.length === 0) {
    return <div className="flex items-center justify-center py-6 text-xs text-zinc-600">No sources</div>
  }

  return (
    <div className="p-2 space-y-2">
      {images.length > 0 && (
        <div className="mb-2">
          <div className="flex items-center gap-1.5 mb-1.5">
            <ImageIcon className="h-3 w-3 text-zinc-500" />
            <span className="text-[9px] font-medium text-zinc-500 uppercase tracking-wider">Images</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {images.slice(0, 4).map((img, i) => (
              <a key={i} href={img} target="_blank" rel="noopener noreferrer">
                <img
                  src={img}
                  alt=""
                  className="w-full h-16 object-cover rounded border border-zinc-800 hover:border-zinc-600 transition-colors"
                  onError={(e) => {
                    ;(e.target as HTMLImageElement).style.display = "none"
                  }}
                />
              </a>
            ))}
          </div>
        </div>
      )}
      {sources.map((s, i) => {
        const domain = s.url.replace(/^https?:\/\/(www\.)?/, "").split("/")[0]
        return (
          <a
            key={i}
            href={s.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block rounded-md border border-zinc-800 bg-zinc-900/40 p-2.5 hover:border-zinc-700 hover:bg-zinc-900/70 transition-colors"
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              <Globe className="h-2.5 w-2.5 text-sky-400 shrink-0" />
              <span className="text-[10px] font-medium text-sky-400 truncate">{domain}</span>
            </div>
            {s.title && <div className="text-[10px] font-medium text-zinc-300 mb-0.5 line-clamp-2">{s.title}</div>}
            {s.content && <div className="text-[9px] text-zinc-500 line-clamp-3">{s.content.slice(0, 150)}</div>}
          </a>
        )
      })}
    </div>
  )
}

export interface ChatSidebarConfig {
  chatId: string
  allAttachments: { messageIndex: number; attachment: Attachment }[]
  expandedEntries: { messageIndex: number; attachmentName: string }[]
  onToggleAttachment: (messageIndex: number, attachmentName: string) => void
  onRemoveAttachment: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
  lastMorphicResult?: MorphicSearchResult
  isElementOpen: (id: string) => boolean
  onElementOpenChange: (id: string, open: boolean) => void
}

export const getChatSidebarConfig = ({
  chatId,
  allAttachments,
  expandedEntries,
  onToggleAttachment,
  onRemoveAttachment,
  onAttachmentContentChange,
  lastMorphicResult,
  isElementOpen,
  onElementOpenChange,
}: ChatSidebarConfig): ChatSidebarElementConfig[] => {
  const elements: ChatSidebarElementConfig[] = []

  if (allAttachments.length > 0) {
    elements.push({
      id: "context",
      icon: FileText,
      title: "Context",
      badge: allAttachments.length,
      open: isElementOpen("context"),
      onOpenChange: (o) => onElementOpenChange("context", o),
        body: (
          <ContextBody
            chatId={chatId}
            allAttachments={allAttachments}
            expandedEntries={expandedEntries}
            onToggleAttachment={onToggleAttachment}
            onRemoveAttachment={onRemoveAttachment}
            onAttachmentContentChange={onAttachmentContentChange}
          />
        ),
    })
  }

  if (lastMorphicResult) {
    const seen = new Set<string>()
    const sources = lastMorphicResult.sources.filter((s) => !seen.has(s.url) && seen.add(s.url))
    elements.push({
      id: "sources",
      icon: Globe,
      title: lastMorphicResult.query,
      badge: sources.length,
      open: isElementOpen("sources"),
      onOpenChange: (o) => onElementOpenChange("sources", o),
      body: <SourcesBody result={lastMorphicResult} />,
    })
  }

  return elements
}
