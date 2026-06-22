"use client"

import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence } from "framer-motion"
import dynamic from "next/dynamic"
import type { Message, Attachment, WebSearchResult, ProjectDocument } from "@/lib/types"
import { ChatMessage, AssistantMessageContent } from "./ChatMessage"
import { ChatInput, type ChatInputHandle } from "./ChatInput"
import { ChatSidebar, type ChatSidebarElementConfig } from "./ChatSidebar"
import { rewriteImages, fileViewerRawUrl } from "@/lib/api"

import { ChevronDown, ChevronLeft, ChevronRight, ChevronsLeftRight, ChevronsRightLeft, FilePlus, FileText, FileType, Globe, GitFork, Image as ImageIcon, Library, Plus, User, X } from "lucide-react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { cn } from "@/lib/utils"
import { ElevationProvider, ElevatedContainer } from "./ElevatedContainer"

const PdfViewer = dynamic(() => import("./PdfViewer").then((m) => ({ default: m.PdfViewer })), { ssr: false })
import { DocumentEditor } from "./DocumentEditor"

function ObsidianGlyph({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinejoin="round" className={className} aria-hidden="true">
      <path d="M12 2 L20 8 L15 22 L6.5 20 L4 9 Z" />
      <path d="M12 2 L11 13 L15 22" />
      <path d="M11 13 L4 9" />
      <path d="M11 13 L6.5 20" />
    </svg>
  )
}

function AttachmentIcon({ mime }: { mime: string }) {
  if (mime.startsWith("image/")) return <ImageIcon className="h-3.5 w-3.5 text-ink shrink-0" />
  if (mime === "application/pdf") return <FileText className="h-3.5 w-3.5 text-ink shrink-0" />
  return <FileText className="h-3.5 w-3.5 text-ink-muted shrink-0" />
}

function AttachmentViewer({ attachment, chatId, slug, onContentChange, pdfWide, onTogglePdfWide }: { attachment: Attachment; chatId: string; slug: string | null; onContentChange?: (newContent: string) => void; pdfWide?: boolean; onTogglePdfWide?: () => void }) {
  if (attachment.mime.startsWith("image/")) {
    return (
      <div className="p-2">
        <img src={attachment.url} alt={attachment.name} className="max-w-full rounded border border-divider" />
      </div>
    )
  }

  if (attachment.mime === "application/pdf") {
    const parsedContent = attachment.parsedMd
    return <PdfAttachmentViewer url={attachment.url} parsedContent={parsedContent} chatId={chatId} slug={slug} pdfWide={pdfWide} onTogglePdfWide={onTogglePdfWide} />
  }

  const content = attachment.parsedMd ?? attachment.content
  if (content) {
    const rewritten = rewriteImages(content, chatId, slug)
    return (
      <div className="p-2 max-h-[500px] overflow-y-auto">
        <LaTeXMarkdown content={rewritten} onContentChange={onContentChange} />
      </div>
    )
  }

  return <p className="p-2 text-xs text-ink-subtle">No preview available</p>
}

function PdfAttachmentViewer({ url, parsedContent, chatId, slug, pdfWide, onTogglePdfWide }: { url: string; parsedContent?: string; chatId: string; slug: string | null; pdfWide?: boolean; onTogglePdfWide?: () => void }) {
  const [mdOpen, setMdOpen] = useState(false)
  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0">
        <PdfViewer url={url} isWide={pdfWide} onToggleWide={onTogglePdfWide} />
      </div>
      {parsedContent && (
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
              <LaTeXMarkdown content={rewriteImages(parsedContent, chatId, slug)} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ContextBody({
  chatId,
  slug,
  allAttachments,
  expandedEntries,
  onToggleExpand,
  onToggleActive,
  onRemoveAttachment,
  onAttachmentContentChange,
  pdfWide,
  onTogglePdfWide,
}: {
  chatId: string
  slug: string | null
  allAttachments: { messageIndex: number; attachment: Attachment }[]
  expandedEntries: { messageIndex: number; attachmentName: string }[]
  onToggleExpand: (messageIndex: number, attachmentName: string) => void
  onToggleActive?: (messageIndex: number, attachmentName: string) => void
  onRemoveAttachment: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
  pdfWide?: boolean
  onTogglePdfWide?: () => void
}) {
  const isExpanded = (mi: number, name: string) =>
    expandedEntries.some((e) => e.messageIndex === mi && e.attachmentName === name)

  if (allAttachments.length === 0) {
    return (
      <div className="flex items-center justify-center py-6 text-xs text-ink-faint">
        No attachments
      </div>
    )
  }

  return (
    <div>
      {allAttachments.map(({ messageIndex: mi, attachment: att }) => {
        const expanded = isExpanded(mi, att.name)
        return (
          <div key={`${mi}-${att.name}`} className="border-b border-divider/50">
            <div className="flex items-center gap-1 px-2 py-2 hover:bg-surface/50 transition-colors">
              <button
                type="button"
                onClick={() => onToggleExpand(mi, att.name)}
                className="rounded p-0.5 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors shrink-0"
                aria-label={expanded ? "Collapse preview" : "Expand preview"}
              >
                {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              </button>
              <button
                type="button"
                onClick={() => onToggleActive?.(mi, att.name)}
                className={cn(
                  "flex min-w-0 flex-1 items-center gap-2 text-left transition-colors",
                  att.active ? "text-ink" : "text-ink-faint",
                )}
              >
                <AttachmentIcon mime={att.mime} />
                <span className="text-[11px] font-medium truncate">{att.name}</span>
              </button>
              <button
                type="button"
                onClick={() => onRemoveAttachment(mi, att.name)}
                className="rounded p-0.5 text-ink-faint hover:text-danger hover:bg-surface-elevated transition-colors shrink-0"
                aria-label="Remove attachment"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
            {expanded && (
              <div className={att.mime === "application/pdf" ? "h-[60vh]" : "max-h-[60vh] overflow-y-auto"}>
                <AttachmentViewer attachment={att} chatId={chatId} slug={slug} onContentChange={onAttachmentContentChange ? (nc: string) => onAttachmentContentChange(mi, att.name, nc) : undefined} pdfWide={pdfWide} onTogglePdfWide={onTogglePdfWide} />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function DocumentsBody({ documents, slug, onSelect, editingPath, onEdit, onDelete, onSaved, onLiveContent, pdfWide, onTogglePdfWide }: { documents: ProjectDocument[]; slug: string | null; onSelect?: (path: string) => void; editingPath?: string | null; onEdit?: (path: string | null) => void; onDelete?: (path: string) => void; onSaved?: (filename?: string, content?: string) => void; onLiveContent?: (path: string, content: string | null) => void; pdfWide?: boolean; onTogglePdfWide?: () => void }) {
  if (documents.length === 0) {
    return (
      <div className="flex items-center justify-center py-6 text-xs text-ink-faint">
        No documents
      </div>
    )
  }

  const isEditable = (doc: ProjectDocument) => /\.(md|tex|canvas)$/.test(doc.name)
  const pdfDocs = documents.filter((d) => d.mime === "application/pdf").map((d) => ({ path: d.path, name: d.name }))
  const imageDocs = documents.filter((d) => d.mime.startsWith("image/")).map((d) => ({ path: d.path, name: d.name }))

  return (
    <div>
      {documents.map((doc) => {
        const Icon = doc.mime === "application/pdf" ? FileType : FileText
        const editable = isEditable(doc)
        const expanded = editingPath === doc.path
        return (
          <div key={doc.path} className="border-b border-divider/50">
            <div className="flex items-center gap-1 px-2 py-2 hover:bg-surface/50 transition-colors">
              <button
                type="button"
                onClick={() => onEdit?.(expanded ? null : doc.path)}
                className="rounded p-0.5 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors shrink-0"
              >
                {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              </button>
              <button
                type="button"
                onClick={() => onSelect?.(doc.path)}
                className="flex min-w-0 flex-1 items-center gap-2 text-left text-ink-muted hover:text-ink transition-colors"
              >
                <Icon className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
                <span className="min-w-0 flex-1 truncate text-[11px] font-medium">{doc.name}</span>
              </button>
              <button
                type="button"
                onClick={() => onDelete?.(doc.path)}
                className="rounded p-0.5 text-ink-faint hover:text-danger hover:bg-surface-elevated transition-colors shrink-0"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
            {expanded && editable && slug && (
              <DocumentEditor path={doc.path} slug={slug} onClose={() => onEdit?.(null)} onSaved={onSaved} onLiveContent={onLiveContent} availablePdfs={pdfDocs} availableImages={imageDocs} />
            )}
            {expanded && !editable && doc.mime === "application/pdf" && (
              <PdfViewer url={fileViewerRawUrl(doc.path)} isWide={pdfWide} onToggleWide={onTogglePdfWide} fitWidth />
            )}
            {expanded && !editable && doc.mime.startsWith("image/") && (
              <div className="p-2 max-h-[60vh] overflow-y-auto">
                <img src={fileViewerRawUrl(doc.path)} alt={doc.name} className="max-w-full rounded border border-divider" />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function SourcesBody({ result }: { result: WebSearchResult }) {
  const seen = new Set<string>()
  const sources = result.sources.filter((s) => !seen.has(s.url) && seen.add(s.url))
  const images = [...new Set(result.images)]
  const videos = result.videos ?? []

  if (sources.length === 0 && images.length === 0 && videos.length === 0) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">No sources</div>
  }

  return (
    <div className="p-2 space-y-2">
      {videos.length > 0 && (
        <div className="mb-2">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Globe className="h-3 w-3 text-ink-subtle" />
            <span className="text-[9px] font-medium text-ink-subtle uppercase tracking-wider">Videos</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {videos.slice(0, 4).map((v, i) => (
              <a key={i} href={v.url} target="_blank" rel="noopener noreferrer">
                <img
                  src={v.thumbnail}
                  alt=""
                  className="w-full h-16 object-cover rounded border border-divider hover:border-ink-muted transition-colors"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none" }}
                />
              </a>
            ))}
          </div>
        </div>
      )}
      {images.length > 0 && (
        <div className="mb-2">
          <div className="flex items-center gap-1.5 mb-1.5">
            <ImageIcon className="h-3 w-3 text-ink-subtle" />
            <span className="text-[9px] font-medium text-ink-subtle uppercase tracking-wider">Images</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {images.slice(0, 4).map((img, i) => (
              <a key={i} href={img} target="_blank" rel="noopener noreferrer">
                <img
                  src={img}
                  alt=""
                  className="w-full h-16 object-cover rounded border border-divider hover:border-ink-muted transition-colors"
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
            className="block rounded-md border border-divider bg-surface/40 p-2.5 hover:border-divider-strong hover:bg-surface/70 transition-colors"
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              <Globe className="h-2.5 w-2.5 text-ink shrink-0" />
              <span className="text-[10px] font-medium text-ink truncate">{domain}</span>
            </div>
            {s.title && <div className="text-[10px] font-medium text-ink mb-0.5 line-clamp-2">{s.title}</div>}
            {s.content && <div className="text-[9px] text-ink-subtle line-clamp-3">{s.content.slice(0, 150)}</div>}
          </a>
        )
      })}
    </div>
  )
}

function buildSidebarElements({
  chatId,
  slug,
  allAttachments,
  expandedEntries,
  onToggleExpand,
  onToggleAttachmentActive,
  onRemoveAttachment,
  onAttachmentContentChange,
  lastSearchResult,
  obsidianEnabled,
  onOpenObsidian,
  documents,
  onSelectDocument,
  onOpenDocuments,
  onCreateDocument,
  editingDocPath,
  onEditDocument,
  onDeleteDocument,
  onDocumentSaved,
  isElementOpen,
  onElementOpenChange,
  pdfWide,
  onTogglePdfWide,
  liveCanvasRef,
}: {
  chatId: string
  slug: string | null
  allAttachments: { messageIndex: number; attachment: Attachment }[]
  expandedEntries: { messageIndex: number; attachmentName: string }[]
  onToggleExpand: (messageIndex: number, attachmentName: string) => void
  onToggleAttachmentActive?: (messageIndex: number, attachmentName: string) => void
  onRemoveAttachment: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
  lastSearchResult?: WebSearchResult
  obsidianEnabled?: boolean
  onOpenObsidian?: () => void
  documents?: ProjectDocument[]
  onSelectDocument?: (path: string) => void
  onOpenDocuments?: () => void
  onCreateDocument?: () => void
  editingDocPath?: string | null
  onEditDocument?: (path: string | null) => void
  onDeleteDocument?: (path: string) => void
  onDocumentSaved?: (filename?: string, content?: string) => void
  isElementOpen: (id: string) => boolean
  onElementOpenChange: (id: string, open: boolean) => void
  pdfWide?: boolean
  onTogglePdfWide?: () => void
  liveCanvasRef?: React.MutableRefObject<{ path: string; content: string } | null>
}): ChatSidebarElementConfig[] {
  const elements: ChatSidebarElementConfig[] = []

  if (allAttachments.length > 0 || obsidianEnabled) {
    elements.push({
      id: "context",
      icon: FileText,
      title: "Context",
      badge: allAttachments.length > 0 ? allAttachments.length : undefined,
      action: obsidianEnabled ? (
        <button
          type="button"
          onClick={onOpenObsidian}
          aria-label="Load Obsidian note"
          className="rounded p-1 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors"
        >
          <ObsidianGlyph className="h-3.5 w-3.5" />
        </button>
      ) : undefined,
      open: isElementOpen("context"),
      onOpenChange: (o) => onElementOpenChange("context", o),
        body: (
          <ContextBody
            chatId={chatId}
            slug={slug}
            allAttachments={allAttachments}
            expandedEntries={expandedEntries}
            onToggleExpand={onToggleExpand}
            onToggleActive={onToggleAttachmentActive}
            onRemoveAttachment={onRemoveAttachment}
            onAttachmentContentChange={onAttachmentContentChange}
            pdfWide={pdfWide}
            onTogglePdfWide={onTogglePdfWide}
          />
        ),
    })
  }

  if (onOpenDocuments) {
    const docs = documents ?? []
    elements.push({
      id: "documents",
      icon: Library,
      title: "Documents",
      badge: docs.length > 0 ? docs.length : undefined,
      action: (
        <div className="flex items-center gap-0.5">
          {onCreateDocument && (
            <button
              type="button"
              onClick={onCreateDocument}
              aria-label="Create document"
              className="rounded p-1 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors"
            >
              <FilePlus className="h-3.5 w-3.5" />
            </button>
          )}
          <button
            type="button"
            onClick={onOpenDocuments}
            aria-label="Browse documents"
            className="rounded p-1 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>
      ),
      open: isElementOpen("documents"),
      onOpenChange: (o) => onElementOpenChange("documents", o),
      body: (
        <DocumentsBody documents={docs} slug={slug} onSelect={onSelectDocument} editingPath={editingDocPath} onEdit={onEditDocument} onDelete={onDeleteDocument} onSaved={onDocumentSaved} onLiveContent={liveCanvasRef ? (p, c) => { liveCanvasRef.current = c !== null ? { path: p, content: c } : null } : undefined} pdfWide={pdfWide} onTogglePdfWide={onTogglePdfWide} />
      ),
    })
  }

  if (lastSearchResult) {
    const seen = new Set<string>()
    const sources = lastSearchResult.sources.filter((s) => !seen.has(s.url) && seen.add(s.url))
    elements.push({
      id: "sources",
      icon: Globe,
      title: lastSearchResult.query,
      badge: sources.length,
      open: isElementOpen("sources"),
      onOpenChange: (o) => onElementOpenChange("sources", o),
      body: <SourcesBody result={lastSearchResult} />,
    })
  }

  return elements
}

const DEFAULT_EXPANDED_TAIL = 1
const BOTTOM_GAP_PX = 16

interface ChatContainerProps {
  chatId: string
  messages: Message[]
  isStreaming: boolean
  onSend: (text: string, attachments: Attachment[]) => void
  onCancel: () => void
  onDeletePair: (index: number) => void
  onRegenerate?: (index: number) => void
  onBranch?: (qaIndex: number) => void
  focusQaIndex?: number | null
  focusKey?: number
  branchMessageIdx?: number | null
  isActive?: boolean
  className?: string
  slug?: string | null
  chatMaxWidth?: number
  onOCRRequest?: (image: string) => void
  onRemoveAttachment?: (messageIndex: number, attachmentName: string) => void
  onToggleAttachmentActive?: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
  obsidianEnabled?: boolean
  onOpenObsidian?: () => void
  documents?: ProjectDocument[]
  onSelectDocument?: (path: string) => void
  onOpenDocuments?: () => void
  onCreateDocument?: () => void
  onDeleteDocument?: (path: string) => void
  onDocumentSaved?: (filename?: string, content?: string) => void
  chatInputRef?: React.RefObject<ChatInputHandle | null>
  liveCanvasRef?: React.MutableRefObject<{ path: string; content: string } | null>
}

export function ChatContainer({
  chatId,
  messages,
  isStreaming,
  onSend,
  onCancel,
  onDeletePair,
  onRegenerate,
  onBranch,
  focusQaIndex,
  focusKey,
  branchMessageIdx,
  isActive,
  className,
  slug = null,
  chatMaxWidth,
  onOCRRequest,
  onRemoveAttachment,
  onToggleAttachmentActive,
  onAttachmentContentChange,
  obsidianEnabled,
  onOpenObsidian,
  documents,
  onSelectDocument,
  onOpenDocuments,
  onCreateDocument,
  onDeleteDocument,
  onDocumentSaved,
  chatInputRef,
  liveCanvasRef,
}: ChatContainerProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const [editingDocPath, setEditingDocPath] = useState<string | null>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const isPinnedToBottomRef = useRef(true)

  const lastSearchResult = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].search_result) return messages[i].search_result
    }
    return undefined
  }, [messages])

  const allAttachments = useMemo<{ messageIndex: number; attachment: Attachment }[]>(() => {
    const seen = new Set<string>()
    const entries: { messageIndex: number; attachment: Attachment }[] = []
    for (let i = 0; i < messages.length; i++) {
      const atts = messages[i].attachments
      if (!atts) continue
      for (const att of atts) {
        const key = `${i}-${att.name}`
        if (seen.has(key)) continue
        seen.add(key)
        entries.push({ messageIndex: i, attachment: att })
      }
    }
    return entries
  }, [messages])

  const [contextOpen, setContextOpen] = useState(false)
  const [expandedEntries, setExpandedEntries] = useState<{ messageIndex: number; attachmentName: string }[]>([])
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const [openElements, setOpenElements] = useState<Set<string>>(new Set())
  const [isWideMode, setIsWideMode] = useState(false)
  const [pdfWide, setPdfWide] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const savedSidebarWidth = useRef(320)

  const togglePdfWide = useCallback(() => {
    setPdfWide((prev) => {
      if (!prev) {
        savedSidebarWidth.current = sidebarWidth
        const total = containerRef.current?.clientWidth ?? 0
        if (total > 0) setSidebarWidth(Math.round(total / 2))
      } else {
        setSidebarWidth(savedSidebarWidth.current)
      }
      return !prev
    })
  }, [sidebarWidth])

  const handleAttachmentClick = useCallback((messageIndex: number, attachment: Attachment) => {
    setContextOpen(true)
    setExpandedEntries([{ messageIndex, attachmentName: attachment.name }])
  }, [])

  const handleToggleExpand = useCallback((mi: number, name: string) => {
    setExpandedEntries(prev => {
      const idx = prev.findIndex(e => e.messageIndex === mi && e.attachmentName === name)
      if (idx >= 0) {
        return prev.filter((_, i) => i !== idx)
      }
      return [...prev, { messageIndex: mi, attachmentName: name }]
    })
  }, [])

  const handleRemoveAttachment = useCallback((mi: number, name: string) => {
    onRemoveAttachment?.(mi, name)
  }, [onRemoveAttachment])

  useEffect(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const handleScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
      isPinnedToBottomRef.current = nearBottom
    }
    el.addEventListener("scroll", handleScroll, { passive: true })
    return () => el.removeEventListener("scroll", handleScroll)
  }, [])

  useEffect(() => {
    if (!isPinnedToBottomRef.current) return
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const [manualOverrides, setManualOverrides] = useState<Map<number, boolean>>(new Map())

  useEffect(() => {
    if (focusQaIndex == null) {
      setManualOverrides(new Map())
      return
    }
    const gi = focusQaIndex * 2
    setManualOverrides(new Map([[gi, true]]))
    requestAnimationFrame(() => {
      const el = scrollContainerRef.current
      if (!el) return
      const anchorGi = gi - 2
      const target = anchorGi >= 0
        ? el.querySelector(`[data-qa-index="${anchorGi}"]`)
        : el.querySelector(`[data-qa-index="${gi}"]`)
      target?.scrollIntoView({ behavior: "smooth", block: "start" })
    })
  }, [focusQaIndex, focusKey])

  useEffect(() => {
    if (messages.length === 0) {
      setManualOverrides(new Map())
      setOpenElements(new Set())
    }
  }, [messages])

  const pairs = useMemo<{ user: Message; assistant: Message; globalIndex: number }[]>(() => {
    const out: { user: Message; assistant: Message; globalIndex: number }[] = []
    for (let i = 0; i < messages.length; i += 2) {
      if (messages[i]?.role === "user") {
        out.push({
          user: messages[i],
          assistant: messages[i + 1] ?? { role: "assistant", content: "" },
          globalIndex: i,
        })
      }
    }
    return out
  }, [messages])

  const tailStartGlobalIndex = pairs.length > DEFAULT_EXPANDED_TAIL
    ? pairs[pairs.length - DEFAULT_EXPANDED_TAIL].globalIndex
    : (pairs.length > 0 ? pairs[0].globalIndex : -1)

  function abbreviate(text: string, msg?: Message): string {
    if (text) {
      const firstLine = text.split("\n")[0]
      if (firstLine.length <= 80) return firstLine
      return firstLine.slice(0, 80) + "\u2026"
    }
    const atts = msg?.attachments
    if (atts && atts.length > 0) {
      const names = atts.map(a => a.name).join(", ")
      return names.length <= 80 ? names : names.slice(0, 80) + "\u2026"
    }
    return "(empty)"
  }

  function isExpandedNormal(idx: number): boolean {
    const override = manualOverrides.get(idx)
    if (override !== undefined) return override
    return idx >= tailStartGlobalIndex
  }

  function togglePair(idx: number) {
    setManualOverrides(prev => {
      const next = new Map(prev)
      const current = isExpandedNormal(idx)
      next.set(idx, !current)
      return next
    })
  }

  const [hoveredPair, setHoveredPair] = useState<number | null>(null)
  const [keyboardFocusIdx, setKeyboardFocusIdx] = useState<number | null>(null)
  const collapsedRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const leaveTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const allPairsList = useMemo<number[]>(() => {
    const gs: number[] = []
    for (let i = 0; i < messages.length; i += 2) {
      if (messages[i]?.role === "user") {
        gs.push(i)
      }
    }
    return gs
  }, [messages])

  const activeFocusedGlobalIndex =
    keyboardFocusIdx !== null && keyboardFocusIdx >= 0 && keyboardFocusIdx < allPairsList.length
      ? allPairsList[keyboardFocusIdx]
      : null

  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    isPinnedToBottomRef.current = true
    onSend(text, attachments)
  }, [onSend])

  const handleMouseEnter = (globalIndex: number) => {
    if (leaveTimeoutRef.current) {
      clearTimeout(leaveTimeoutRef.current)
      leaveTimeoutRef.current = null
    }
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    hoverTimeoutRef.current = setTimeout(() => {
      setHoveredPair(globalIndex)
    }, 500)
  }

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
    if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    leaveTimeoutRef.current = setTimeout(() => {
      setHoveredPair(null)
      leaveTimeoutRef.current = null
    }, 100)
  }

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current)
      if (leaveTimeoutRef.current) clearTimeout(leaveTimeoutRef.current)
    }
  }, [])

  const navigateFocus = useCallback((key: "ArrowUp" | "ArrowDown") => {
    if (allPairsList.length === 0) {
      setKeyboardFocusIdx(null)
      return
    }
    setKeyboardFocusIdx((cur) => {
      const start = cur ?? (key === "ArrowDown" ? -1 : allPairsList.length)
      return Math.max(0, Math.min(start + (key === "ArrowDown" ? 1 : -1), allPairsList.length - 1))
    })
  }, [allPairsList])

  const confirmFocus = useCallback(() => {
    if (keyboardFocusIdx === null || keyboardFocusIdx < 0 || keyboardFocusIdx >= allPairsList.length) return
    togglePair(allPairsList[keyboardFocusIdx])
    setKeyboardFocusIdx(null)
  }, [keyboardFocusIdx, allPairsList, togglePair])

  useEffect(() => {
    const eat = (e: KeyboardEvent) => { e.preventDefault(); e.stopPropagation() }
    const handleKeyDown = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey
      if (ctrl && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        eat(e)
        navigateFocus(e.key)
        scrollContainerRef.current?.focus()
      } else if (e.key === "Enter" && keyboardFocusIdx !== null) {
        eat(e)
        confirmFocus()
      } else if (e.key === "Escape" && keyboardFocusIdx !== null) {
        eat(e)
        setKeyboardFocusIdx(null)
      }
    }
    window.addEventListener("keydown", handleKeyDown, true)
    return () => window.removeEventListener("keydown", handleKeyDown, true)
  }, [navigateFocus, confirmFocus, keyboardFocusIdx])

  useEffect(() => {
    if (keyboardFocusIdx === null) return
    const container = scrollContainerRef.current
    if (!container) return

    const atTop = keyboardFocusIdx < 3
    const targetIdx = atTop ? 0 : keyboardFocusIdx - 3
    const targetGlobalIdx = allPairsList[targetIdx]
    if (targetGlobalIdx === undefined) return
    const el = collapsedRefs.current.get(targetGlobalIdx)
    if (!el) return

    const elRect = el.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    const offset = (atTop ? elRect.top : elRect.bottom) - containerRect.top

    container.scrollTo({
      top: container.scrollTop + offset,
      behavior: "smooth"
    })
  }, [keyboardFocusIdx, allPairsList])

  const sidebarElements = useMemo(
    () =>
      buildSidebarElements({
        chatId,
        slug,
        allAttachments,
        expandedEntries,
        onToggleExpand: handleToggleExpand,
        onToggleAttachmentActive,
        onRemoveAttachment: handleRemoveAttachment,
        onAttachmentContentChange,
        lastSearchResult,
        obsidianEnabled,
        onOpenObsidian,
        documents,
        onSelectDocument,
        onOpenDocuments,
        onCreateDocument,
        editingDocPath,
        onEditDocument: setEditingDocPath,
        onDeleteDocument,
        onDocumentSaved,
        isElementOpen: (id) => openElements.has(id),
        onElementOpenChange: (id, open) => {
          setOpenElements((prev) => {
            const next = new Set(prev)
            if (open) next.add(id)
            else next.delete(id)
            return next
          })
        },
        pdfWide,
        onTogglePdfWide: togglePdfWide,
        liveCanvasRef,
      }),
    [chatId, slug, allAttachments, expandedEntries, handleToggleExpand, onToggleAttachmentActive, handleRemoveAttachment, onAttachmentContentChange, lastSearchResult, obsidianEnabled, onOpenObsidian, documents, onSelectDocument, onOpenDocuments, onCreateDocument, editingDocPath, onDeleteDocument, onDocumentSaved, openElements, pdfWide, togglePdfWide, liveCanvasRef]
  )

  const hasSidebarContent = sidebarElements.length > 0
  const inputAreaRef = useRef<HTMLDivElement>(null)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)

  useEffect(() => {
    const el = inputAreaRef.current
    if (!el) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setInputAreaHeight(entry.contentRect.height)
      }
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!hasSidebarContent) {
      setContextOpen(false)
    }
  }, [hasSidebarContent])

  useEffect(() => {
    if (!contextOpen) setPdfWide(false)
  }, [contextOpen])

  const effectiveMaxWidth = isWideMode ? undefined : (chatMaxWidth || undefined)
  const toggleWideMode = useCallback(() => setIsWideMode((w) => !w), [])

  return (
    <div ref={containerRef} className={cn("flex h-full relative", className)}>
      <div className="flex-1 min-w-0 flex flex-col relative">
        <div
          ref={scrollContainerRef}
          tabIndex={-1}
          className="flex-1 overflow-y-auto text-ink outline-none"
          style={{ paddingBottom: inputAreaHeight + BOTTOM_GAP_PX }}
        >
          <div className="mx-auto" style={{ maxWidth: effectiveMaxWidth }}>
          <button
            onClick={toggleWideMode}
            aria-label={isWideMode ? "Exit wide mode" : "Enter wide mode"}
            className="absolute left-2 top-2 z-30 p-1 text-ink-faint opacity-0 hover:opacity-100 focus-visible:opacity-100 hover:text-ink-muted transition-opacity duration-200 cursor-pointer"
          >
            {isWideMode ? <ChevronsRightLeft className="h-4 w-4" /> : <ChevronsLeftRight className="h-4 w-4" />}
          </button>
          <ElevationProvider darkColor="var(--paper)" brightColor="var(--surface-elevated)" numLevels={3} startLevel={1}>
            <AnimatePresence>
            {pairs.map(({ user, assistant, globalIndex }, qaIndex) => {
              const expanded = isExpandedNormal(globalIndex)
              const qLabel = abbreviate(user.content, user)
              const showPreview = hoveredPair === globalIndex || activeFocusedGlobalIndex === globalIndex
              const showBranchDivider = branchMessageIdx != null && qaIndex === branchMessageIdx
              return (
                <Fragment key={globalIndex}>
                <div data-qa-index={globalIndex} className="group relative">
                  {!expanded && (
                    <ElevatedContainer
                      ref={(el) => {
                        if (el) collapsedRefs.current.set(globalIndex, el)
                        else collapsedRefs.current.delete(globalIndex)
                      }}
                      onMouseEnter={() => handleMouseEnter(globalIndex)}
                      onMouseLeave={handleMouseLeave}
                      onClick={() => {
                        togglePair(globalIndex)
                        setKeyboardFocusIdx(null)
                      }}
                      className={cn(
                        "mx-3 my-4 rounded-xl border border-divider/40 shadow-[var(--shadow-sm)] overflow-hidden cursor-pointer transition-all duration-300",
                        activeFocusedGlobalIndex === globalIndex && "ring-1 ring-ink-muted/50"
                      )}
                    >
                      <div className="flex gap-3 px-5 py-4 items-start">
                        <div className="mt-0.5 shrink-0">
                          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-surface-elevated">
                            <User className="h-3.5 w-3.5 text-ink-muted" />
                          </div>
                        </div>
                        <div className="min-w-0 flex-1 flex flex-col">
                          <div className="mb-0.5 text-xs font-medium text-ink-subtle">You</div>
                          {showPreview ? (
                            <div className="max-h-[12.5vh] overflow-y-auto text-ink-muted">
                              <LaTeXMarkdown content={user.content} compact />
                            </div>
                          ) : (
                            <span className="text-sm text-ink-muted truncate">{qLabel}</span>
                          )}
                        </div>
                        <ChevronRight className="h-4 w-4 shrink-0 text-ink-faint mt-1" />
                      </div>
                      <div
                        className={cn(
                          "grid transition-[grid-template-rows] duration-300 ease-out",
                          showPreview ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
                        )}
                      >
                        <div className="overflow-hidden">
                          <div className="px-5 pb-4">
                            {assistant.content && (
                              <ElevatedContainer className="rounded-lg border border-divider/30 overflow-hidden">
                                <div className="px-4 py-3">
                                  <div className="text-xs font-medium text-ink-subtle mb-1">Assistant</div>
                        <div className="max-h-[25vh] overflow-y-auto text-ink-muted">
                          <AssistantMessageContent content={assistant.content} compact />
                        </div>
                                  </div>
                              </ElevatedContainer>
                            )}
                          </div>
                        </div>
                      </div>
                    </ElevatedContainer>
                  )}
                  {expanded && (
                    <>

                      <ElevatedContainer className={cn("mx-3 my-4 rounded-xl border border-divider/40 overflow-hidden transition-all duration-300", activeFocusedGlobalIndex === globalIndex && "ring-1 ring-ink-muted/50")}>
                        <ChatMessage
                          role="user"
                          content={user.content}
                          attachments={user.attachments}
                          index={globalIndex}
                          onAttachmentClick={(att) => handleAttachmentClick(globalIndex, att)}
                          onDelete={onDeletePair}
                          collapsibleUser
                          onCollapse={() => togglePair(globalIndex)}
                        />
                        <ElevatedContainer className="mx-5 mb-5 rounded-lg border border-divider/30 overflow-hidden">
                          <ChatMessage
                            role="assistant"
                            content={assistant.content}
                            search_result={assistant.search_result}
                            research_steps={assistant.research_steps}
                            research_progress={assistant.research_progress}
                            research_trace_id={assistant.research_trace_id}
                            isStreaming={isStreaming}
                            index={globalIndex}
                            onDelete={onDeletePair}
                            onRegenerate={onRegenerate}
                             onBranch={onBranch ? () => onBranch(qaIndex) : undefined}
                          />
                        </ElevatedContainer>
                      </ElevatedContainer>
                    </>
                  )}
                </div>
                {showBranchDivider && (
                  <div role="separator" className="flex items-center gap-2 mx-3 my-3">
                    <GitFork className="h-3.5 w-3.5 shrink-0 text-ink-subtle" aria-hidden="true" />
                    <span className="text-[10px] font-medium text-ink-muted tabular-nums">branched@{branchMessageIdx}</span>
                    <div className="flex-1 h-px bg-divider-strong" />
                  </div>
                )}
                </Fragment>
              )
            })}
          </AnimatePresence>
          </ElevationProvider>
          <div ref={bottomRef} />
          </div>
        </div>
        <div ref={inputAreaRef} className="absolute bottom-0 left-0 right-0 z-10 pointer-events-none">
          <div className="absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t from-paper via-paper/40 via-30% to-transparent pointer-events-none" />
          <div className="relative pb-6 pointer-events-auto">
            <div className="mx-auto" style={chatMaxWidth && !isWideMode ? { maxWidth: chatMaxWidth } : undefined}>
            <ChatInput
              ref={chatInputRef}
              chatId={chatId}
              onSend={handleSend}
              onOCRRequest={onOCRRequest}
              disabled={isStreaming}
              isStreaming={isStreaming}
              onCancel={onCancel}
              slug={slug}
            />
            </div>
          </div>
        </div>
        {hasSidebarContent && (
          <button
            onClick={() => setContextOpen((c) => !c)}
            aria-label={contextOpen ? "Collapse sidebar" : "Open sidebar"}
            aria-expanded={contextOpen}
            className="absolute right-0 top-3 z-30 flex items-center justify-center h-12 w-4 rounded-l-md border border-r-0 border-divider-strong bg-surface-elevated/80 text-ink hover:bg-surface-elevated hover:text-ink hover:w-5 transition-all shadow-[var(--shadow-sm)]"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}
      </div>
      {hasSidebarContent && contextOpen && chatId && (
        <ChatSidebar elements={sidebarElements} width={sidebarWidth} onWidthChange={setSidebarWidth} maxWidth={pdfWide ? sidebarWidth : undefined} />
      )}
    </div>
  )
}
