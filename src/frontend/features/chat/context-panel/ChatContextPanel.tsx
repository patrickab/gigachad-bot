"use client"

import { createContext, useContext, useMemo } from "react"
import { ChevronDown, ChevronRight, FilePlus, FileText, FolderOpen, Globe, Library, Plus, X } from "lucide-react"
import type { Attachment, ProjectDocument, WebSearchResult } from "@/lib/domain/types"
import { cn } from "@/lib/utils"
import { AttachmentIcon, AttachmentView } from "./AttachmentView"
import { DocumentsBody } from "./DocumentsBody"
import type { SidePanelSection } from "@/components/ui/SidePanel"
import { SourcesBody, uniqueSources } from "./SourcesBody"

// Document/vault/attachment callbacks the sidebar (Context + Documents
// elements) needs but ChatContainer itself never touches (provided by
// TabContent so they don't have to thread through ChatContainerProps).
export interface ChatSidebarContextValue {
  onRemoveAttachment?: (messageIndex: number, attachmentName: string) => void
  onToggleAttachmentActive?: (messageIndex: number, attachmentName: string) => void
  onAttachmentContentChange?: (messageIndex: number, attachmentName: string, newContent: string) => void
  filesEnabled?: boolean
  onOpenFilePicker?: () => void
  documents?: ProjectDocument[]
  onSelectDocument?: (path: string) => void
  /** Attaches every architecture source as one text attachment; absent when the project has none. */
  onAttachArchitecture?: () => Promise<void>
  /** Why the last architecture attach failed; cleared when the next one starts. */
  architectureError?: string
  onOpenDocuments?: () => void
  onCreateDocument?: () => void
  onDeleteDocument?: (path: string) => void
  onRenameDocument?: (path: string, name: string) => void | Promise<void>
  onDocumentSaved?: (filename?: string, content?: string) => void
  liveCanvasRef?: React.MutableRefObject<{ path: string; content: string } | null>
  /** Model behind the editor's LLM features (assistant sidebar, inline edit). */
  editorModel?: string
}

const ChatSidebarContext = createContext<ChatSidebarContextValue>({})
export const ChatSidebarProvider = ChatSidebarContext.Provider

/** An attachment with the index of the message that carries it. */
export interface AttachmentEntry {
  messageIndex: number
  attachment: Attachment
}

/** Identifies one attachment row whose preview is open. */
export interface ExpandedAttachment {
  messageIndex: number
  attachmentName: string
}

function ContextBody({
  allAttachments,
  expandedEntries,
  onToggleExpand,
  onToggleActive,
  onRemoveAttachment,
  onAttachmentContentChange,
  pdfWide,
  onTogglePdfWide,
}: {
  allAttachments: AttachmentEntry[]
  expandedEntries: ExpandedAttachment[]
  onToggleExpand: (messageIndex: number, attachmentName: string) => void
  onToggleActive?: (messageIndex: number, attachmentName: string) => void
  onRemoveAttachment?: (messageIndex: number, attachmentName: string) => void
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
                <AttachmentIcon mime={att.mime} generic={FileText} size="h-3.5 w-3.5" />
                <span className="text-[11px] font-medium truncate">{att.name}</span>
              </button>
              <button
                type="button"
                onClick={() => onRemoveAttachment?.(mi, att.name)}
                className="rounded p-0.5 text-ink-faint hover:text-danger hover:bg-surface-elevated transition-colors shrink-0"
                aria-label="Remove attachment"
              >
                <X className="h-3 w-3" />
              </button>
            </div>
            {expanded && (
              <AttachmentView
                attachment={att}
                variant="panel"
                onContentChange={onAttachmentContentChange ? (nc: string) => onAttachmentContentChange(mi, att.name, nc) : undefined}
                pdfWide={pdfWide}
                onTogglePdfWide={onTogglePdfWide}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

/** Build the chat side panel's Context, Documents and Sources sections from the surrounding `ChatSidebarContext`. */
export function useSidebarElements({
  slug,
  allAttachments,
  expandedEntries,
  onToggleExpand,
  lastSearchResult,
  editingDocPath,
  onEditDocument,
  isElementOpen,
  onElementOpenChange,
  pdfWide,
  onTogglePdfWide,
}: {
  slug: string | null
  allAttachments: AttachmentEntry[]
  expandedEntries: ExpandedAttachment[]
  onToggleExpand: (messageIndex: number, attachmentName: string) => void
  lastSearchResult?: WebSearchResult
  editingDocPath?: string | null
  onEditDocument?: (path: string | null) => void
  isElementOpen: (id: string) => boolean
  onElementOpenChange: (id: string, open: boolean) => void
  pdfWide?: boolean
  onTogglePdfWide?: () => void
}): SidePanelSection[] {
  const {
    onToggleAttachmentActive,
    onRemoveAttachment,
    onAttachmentContentChange,
    filesEnabled,
    onOpenFilePicker,
    documents,
    onSelectDocument,
    onOpenDocuments,
    onAttachArchitecture,
    architectureError,
    onCreateDocument,
    onDeleteDocument,
    onRenameDocument,
    onDocumentSaved,
    liveCanvasRef,
    editorModel,
  } = useContext(ChatSidebarContext)

  return useMemo(() => {
    const elements: SidePanelSection[] = []

    if (allAttachments.length > 0 || filesEnabled) {
      elements.push({
        id: "context",
        icon: FileText,
        title: "Context",
        badge: allAttachments.length > 0 ? allAttachments.length : undefined,
        action: filesEnabled ? (
          <button
            type="button"
            onClick={onOpenFilePicker}
            aria-label="Load file"
            className="rounded p-1 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors"
          >
            <FolderOpen className="h-3.5 w-3.5" />
          </button>
        ) : undefined,
        open: isElementOpen("context"),
        onOpenChange: (o) => onElementOpenChange("context", o),
        body: (
          <ContextBody
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
          <DocumentsBody
            documents={docs}
            slug={slug}
            onSelect={onSelectDocument}
            onAttachArchitecture={onAttachArchitecture}
            architectureError={architectureError}
            editingPath={editingDocPath}
            onEdit={onEditDocument}
            onDelete={onDeleteDocument}
            onRename={onRenameDocument}
            onSaved={onDocumentSaved}
            onLiveContent={liveCanvasRef ? (p, c) => { liveCanvasRef.current = c !== null ? { path: p, content: c } : null } : undefined}
            editorModel={editorModel}
            pdfWide={pdfWide}
            onTogglePdfWide={onTogglePdfWide}
          />
        ),
      })
    }

    if (lastSearchResult) {
      elements.push({
        id: "sources",
        icon: Globe,
        title: lastSearchResult.query,
        badge: uniqueSources(lastSearchResult).length,
        open: isElementOpen("sources"),
        onOpenChange: (o) => onElementOpenChange("sources", o),
        body: <SourcesBody result={lastSearchResult} />,
      })
    }

    return elements
  }, [
    slug, allAttachments, expandedEntries, onToggleExpand,
    onToggleAttachmentActive, onRemoveAttachment, onAttachmentContentChange,
    lastSearchResult, filesEnabled, onOpenFilePicker,
    documents, onSelectDocument, onAttachArchitecture, architectureError, onOpenDocuments, onCreateDocument,
    editingDocPath, onEditDocument, onDeleteDocument, onRenameDocument, onDocumentSaved,
    isElementOpen, onElementOpenChange, pdfWide, onTogglePdfWide,
    liveCanvasRef, editorModel,
  ])
}
