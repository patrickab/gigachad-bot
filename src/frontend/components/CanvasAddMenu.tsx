"use client"

import { useEffect, useRef, useState, type ReactNode } from "react"
import { FileType, ImageIcon, Network, Pencil, PenLine, Plus, SquarePen, X } from "lucide-react"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { listNotes, listProjectDocuments, removeDocument, renameDocument } from "@/lib/api"
import type { ProjectDocument } from "@/lib/types"
import { baseName } from "@/lib/paths"

const ITEM = "flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors truncate"

interface CanvasAddMenuProps {
  slug?: string
  depth: number
  /** Path of the `.canvas` file being edited — kept out of the canvas picker so a canvas can't embed itself. */
  docPath?: string
  onAddPage: () => void
  onAddImage: (path: string) => void
  onAddPdf: (path: string) => void
  onAddDocument: (path: string) => void
  onAddArchitecture: () => void
  onAddNewCanvas: () => void
  onAddCanvasFile: (path: string) => void
  /** Attachments on the canvas follow a renamed file. */
  onDocumentRenamed: (oldPath: string, newPath: string) => void
  /** A deleted file's attachments go once the delete commits. */
  onDocumentDeleted: (path: string) => void
}

function Section({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <>
      <div className="mx-2 my-1 border-t border-divider/50" />
      {title && <div className="px-3 py-0.5 text-[9px] text-ink-faint uppercase tracking-wider">{title}</div>}
      {children}
    </>
  )
}

function FileButton({ icon, name, onClick }: { icon: ReactNode; name: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className={ITEM}>
      {icon}
      <span className="truncate">{name}</span>
    </button>
  )
}

/**
 * The toolbar's "+" popover: pages, project images/PDFs/documents, an architecture window and canvases.
 * It fetches project assets itself, so chat and canvas mode share one data path. Only the outermost
 * editor offers other canvases, so a canvas window can hold a file without handing out more.
 * Documents (generated tool artifacts included) are renamed and deleted in place.
 */
export function CanvasAddMenu({ slug, depth, docPath, onAddPage, onAddImage, onAddPdf, onAddDocument, onAddArchitecture, onAddNewCanvas, onAddCanvasFile, onDocumentRenamed, onDocumentDeleted }: CanvasAddMenuProps) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const [docs, setDocs] = useState<ProjectDocument[]>([])
  const [renamingPath, setRenamingPath] = useState<string | null>(null)
  const cancelRename = useRef(false)
  const { schedule: scheduleDelete, pending: deletePending } = useUndoDelete()
  const deleteKey = (path: string) => `canvas-document:${slug ?? ""}:${path}`

  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [open])

  useEffect(() => {
    if (!open) return
    const load = slug ? listProjectDocuments(slug) : listNotes()
    load.then(setDocs).catch(() => { /* picker just stays empty */ })
  }, [open, slug])

  const images = slug ? docs.filter((d) => d.mime.startsWith("image/")) : []
  const pdfs = slug ? docs.filter((d) => d.mime === "application/pdf") : []
  const documents = docs.filter((d) => d.path.endsWith(".md") || d.path.endsWith(".plot.json")).filter((d) => !deletePending(deleteKey(d.path)))
  const canvases = slug && depth === 0 ? docs.filter((d) => d.path.endsWith(".canvas") && d.path !== docPath) : []

  /** Every pick closes the menu; renaming and deleting keep it open. */
  const pick = (action: () => void) => () => { action(); setOpen(false) }

  const renameDocumentRow = async (path: string, currentName: string, nextName: string) => {
    setRenamingPath(null)
    if (cancelRename.current) { cancelRename.current = false; return }
    const name = nextName.trim()
    if (!name || name === currentName) return
    try {
      const renamed = await renameDocument(slug ?? "", path, name)
      setDocs((all) => all.map((d) => d.path === path ? { ...d, path: renamed.path, name: renamed.name } : d))
      onDocumentRenamed(path, renamed.path)
    } catch { /* the stored name stays authoritative when rename fails */ }
  }

  // The row hides during its undo window; the file and its attachments go once the delete commits.
  const deleteDocumentRow = (path: string) => {
    scheduleDelete(deleteKey(path), baseName(path), async () => {
      await removeDocument(slug ?? "", path)
      setDocs((all) => all.filter((d) => d.path !== path))
      onDocumentDeleted(path)
    })
  }

  return (
    <div className="relative" ref={menuRef}>
      <button aria-label="Add to canvas"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 rounded px-2 py-1 text-[10px] font-medium text-ink-muted hover:text-ink hover:bg-hover transition-colors"
      >
        <Plus className="h-3 w-3" />
      </button>
      {open && (
        <div className="absolute top-full left-0 mt-1 z-10 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] py-1 min-w-[160px] max-h-[60vh] overflow-y-auto">
          <button onClick={pick(onAddPage)} className={ITEM}>Page</button>
          {images.length > 0 && (
            <Section title="Images">
              {images.map((img) => <FileButton key={img.path} icon={<ImageIcon className="h-3 w-3 shrink-0 text-ink-faint" />} name={img.name} onClick={pick(() => onAddImage(img.path))} />)}
            </Section>
          )}
          {pdfs.length > 0 && (
            <Section title="PDFs">
              {pdfs.map((pdf) => <FileButton key={pdf.path} icon={<FileType className="h-3 w-3 shrink-0 text-ink-faint" />} name={pdf.name} onClick={pick(() => onAddPdf(pdf.path))} />)}
            </Section>
          )}
          {documents.length > 0 && (
            <Section title="Documents">
              {documents.map((docItem) => (
                <div key={docItem.path} className="flex items-center gap-1 pr-2 hover:bg-hover transition-colors">
                  {renamingPath === docItem.path ? (
                    <div className="flex min-w-0 flex-1 items-center gap-2 px-3 py-1.5">
                      <FileType className="h-3 w-3 shrink-0 text-ink-faint" />
                      <input
                        autoFocus
                        defaultValue={docItem.name}
                        aria-label={`New name for ${docItem.name}`}
                        className="min-w-0 flex-1 bg-transparent text-[11px] text-ink outline-none"
                        onBlur={(e) => { void renameDocumentRow(docItem.path, docItem.name, e.currentTarget.value) }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") e.currentTarget.blur()
                          if (e.key === "Escape") { cancelRename.current = true; e.currentTarget.blur() }
                        }}
                      />
                    </div>
                  ) : (
                    <button
                      onClick={pick(() => onAddDocument(docItem.path))}
                      className="flex min-w-0 flex-1 items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink transition-colors"
                    >
                      <FileType className="h-3 w-3 shrink-0 text-ink-faint" />
                      <span className="truncate">{docItem.name}</span>
                    </button>
                  )}
                  <button
                    onClick={() => setRenamingPath(docItem.path)}
                    aria-label={`Rename ${docItem.name}`}
                    className="rounded p-0.5 text-ink-faint hover:text-ink transition-colors shrink-0"
                  >
                    <Pencil className="h-3 w-3" />
                  </button>
                  <button
                    onClick={() => deleteDocumentRow(docItem.path)}
                    aria-label={`Delete ${docItem.name}`}
                    className="rounded p-0.5 text-ink-faint hover:text-danger transition-colors shrink-0"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </Section>
          )}
          <Section>
            {slug ? (
              <FileButton icon={<Network className="h-3 w-3 shrink-0 text-ink-faint" />} name="Architecture" onClick={pick(onAddArchitecture)} />
            ) : (
              // An architecture belongs to a project; say how to get one instead of hiding the entry.
              <div className="flex items-center gap-2 px-3 py-1.5 text-[11px] text-ink-faint">
                <Network className="h-3 w-3 shrink-0" />
                <span>Architecture: open a project canvas</span>
              </div>
            )}
          </Section>
          <Section title="Canvases">
            <FileButton icon={<SquarePen className="h-3 w-3 shrink-0 text-ink-faint" />} name="New canvas" onClick={pick(onAddNewCanvas)} />
            {canvases.map((cv) => <FileButton key={cv.path} icon={<PenLine className="h-3 w-3 shrink-0 text-ink-faint" />} name={cv.name} onClick={pick(() => onAddCanvasFile(cv.path))} />)}
          </Section>
        </div>
      )}
    </div>
  )
}
