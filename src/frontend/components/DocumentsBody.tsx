"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { ChevronDown, ChevronRight, FileText, FileType, X } from "lucide-react"
import type { ProjectDocument } from "@/lib/types"
import { fileViewerRawUrl } from "@/lib/urls"
import { AttachmentView } from "./AttachmentView"
import { DocumentEditor } from "./DocumentEditor"

/** Callbacks and state shared by the documents list and each of its rows. */
interface DocumentActions {
  slug: string | null
  onSelect?: (path: string) => void
  editingPath?: string | null
  onEdit?: (path: string | null) => void
  onDelete?: (path: string) => void
  onRename?: (path: string, name: string) => void | Promise<void>
  onSaved?: (filename?: string, content?: string) => void
  onLiveContent?: (path: string, content: string | null) => void
  /** Model behind the editor's LLM features. */
  editorModel?: string
  pdfWide?: boolean
  onTogglePdfWide?: () => void
}

function DocumentRow({ doc, slug, onSelect, editingPath, onEdit, onDelete, onRename, onSaved, onLiveContent, editorModel, pdfWide, onTogglePdfWide }: DocumentActions & { doc: ProjectDocument }) {
  const [editingTitle, setEditingTitle] = useState(false)
  // Delay single-click attach so a double-click can start a rename instead
  const selectTimer = useRef<number | undefined>(undefined)
  const cancelRename = useRef(false)
  const isPdf = doc.mime === "application/pdf"
  const isImage = doc.mime.startsWith("image/")
  // Extensions DocumentEditor can open
  const editable = /\.(md|tex|canvas|c4)$/.test(doc.name)
  const expanded = editingPath === doc.path
  const Icon = isPdf ? FileType : FileText

  useEffect(() => () => window.clearTimeout(selectTimer.current), [])

  const select = useCallback(() => {
    window.clearTimeout(selectTimer.current)
    selectTimer.current = window.setTimeout(() => onSelect?.(doc.path), 180)
  }, [doc.path, onSelect])

  const startRename = useCallback((event: React.MouseEvent) => {
    event.stopPropagation()
    window.clearTimeout(selectTimer.current)
    setEditingTitle(true)
  }, [])

  const finishRename = useCallback((name: string) => {
    setEditingTitle(false)
    if (cancelRename.current) {
      cancelRename.current = false
      return
    }
    if (name.trim() && name !== doc.name) void onRename?.(doc.path, name)
  }, [doc.name, doc.path, onRename])

  return (
    <div className="border-b border-divider/50">
      <div className="flex items-center gap-1 px-2 py-2 hover:bg-surface/50 transition-colors">
        <button
          type="button"
          aria-label={`${expanded ? "Close" : "Open"} ${doc.name} editor`}
          onClick={() => onEdit?.(expanded ? null : doc.path)}
          className="rounded p-0.5 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors shrink-0"
        >
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        </button>
        <Icon className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
        {editingTitle ? (
          <input
            autoFocus
            defaultValue={doc.name}
            onClick={(event) => event.stopPropagation()}
            onBlur={(event) => finishRename(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") event.currentTarget.blur()
              if (event.key === "Escape") {
                cancelRename.current = true
                event.currentTarget.blur()
              }
            }}
            className="min-w-0 flex-1 bg-transparent text-[11px] font-medium text-ink outline-none"
            aria-label={`Rename ${doc.name}`}
          />
        ) : (
          <button
            type="button"
            aria-label={`Attach ${doc.name} to chat`}
            onClick={select}
            className="min-w-0 flex-1 text-left text-ink-muted hover:text-ink transition-colors"
          >
            <span className="block truncate text-[11px] font-medium" onDoubleClick={startRename}>{doc.name}</span>
          </button>
        )}
        <button aria-label={`Delete ${doc.name}`}
          type="button"
          onClick={() => onDelete?.(doc.path)}
          className="rounded p-0.5 text-ink-faint hover:text-danger hover:bg-surface-elevated transition-colors shrink-0"
        >
          <X className="h-3 w-3" />
        </button>
      </div>
      {expanded && editable && slug && (
        <DocumentEditor path={doc.path} slug={slug} onClose={() => onEdit?.(null)} onSaved={onSaved} onLiveContent={onLiveContent} model={editorModel} />
      )}
      {expanded && !editable && (isImage || isPdf) && (
        <AttachmentView
          attachment={{ name: doc.name, mime: doc.mime, url: fileViewerRawUrl(doc.path), active: true }}
          variant="panel"
          pdfWide={pdfWide}
          onTogglePdfWide={onTogglePdfWide}
        />
      )}
    </div>
  )
}

/** Context-panel body listing the project's documents, each expandable to an editor or preview. */
export function DocumentsBody({ documents, onAttachArchitecture, architectureError, ...rest }: DocumentActions & {
  documents: ProjectDocument[]
  onAttachArchitecture?: () => Promise<void>
  architectureError?: string
}) {
  if (documents.length === 0) {
    return (
      <div className="flex items-center justify-center py-6 text-xs text-ink-faint">
        No documents
      </div>
    )
  }

  return (
    <div>
      {onAttachArchitecture && <button type="button" onClick={() => void onAttachArchitecture()} className="w-full border-b border-divider/50 px-3 py-2 text-left text-[11px] text-ink-muted hover:bg-surface/50 hover:text-ink">Attach architecture to chat</button>}
      {architectureError && <p role="alert" className="px-3 py-1 text-[11px] text-danger">{architectureError}</p>}
      {documents.map((doc) => <DocumentRow key={doc.path} doc={doc} {...rest} />)}
    </div>
  )
}
