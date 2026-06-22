"use client"

import { useCallback, useMemo, useRef, useState } from "react"
import { Upload, Loader2, Plus, Layers, FolderOpen } from "lucide-react"
import type { ProjectDocument } from "@/lib/types"
import { FileViewer } from "@/components/FileViewer"

type Scope = "project" | "global"

interface DocumentPickerProps {
  projectSlug: string | null
  projectDocuments: ProjectDocument[]
  allDocuments: ProjectDocument[]
  onAttach: (path: string) => void
  onAddToProject?: (path: string) => void
  onUpload: (files: File[]) => void | Promise<void>
  onClose: () => void
}

export function DocumentPicker({
  projectSlug,
  projectDocuments,
  allDocuments,
  onAttach,
  onAddToProject,
  onUpload,
  onClose,
}: DocumentPickerProps) {
  const [scope, setScope] = useState<Scope>(projectSlug ? "project" : "global")
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const docs = scope === "project" ? projectDocuments : allDocuments
  const files = useMemo(() => docs.map((d) => d.path), [docs])
  const projectPaths = useMemo(() => new Set(projectDocuments.map((d) => d.path)), [projectDocuments])

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files ? Array.from(e.target.files) : []
    e.target.value = ""
    if (picked.length === 0) return
    setUploading(true)
    try {
      await onUpload(picked)
    } finally {
      setUploading(false)
    }
  }, [onUpload])

  const headerActions = projectSlug ? (
    <>
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        disabled={uploading}
        title="Add documents (PDF → MinerU)"
        aria-label="Add documents"
        className="rounded p-1 text-ink-faint hover:text-ink hover:bg-surface-elevated transition-colors disabled:cursor-not-allowed disabled:opacity-60"
      >
        {uploading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
      </button>
      <input ref={fileInputRef} type="file" accept="application/pdf" multiple className="hidden" onChange={handleUpload} />
    </>
  ) : undefined

  const statusLabel = scope === "project" ? (
    <><FolderOpen className="h-3 w-3 shrink-0" /><span className="truncate">project — {projectSlug}</span></>
  ) : (
    <><Layers className="h-3 w-3 shrink-0" /><span className="truncate">all documents</span></>
  )

  const renderRowSuffix = useCallback((path: string) => {
    if (scope !== "global" || !onAddToProject || !projectSlug || projectPaths.has(path)) return null
    return (
      <span
        role="button"
        tabIndex={-1}
        onClick={(e) => { e.stopPropagation(); onAddToProject(path) }}
        title="Add to current project"
        className="rounded p-0.5 text-ink-faint opacity-0 group-hover:opacity-100 hover:text-ink hover:bg-surface transition"
      >
        <Plus className="h-3 w-3" />
      </span>
    )
  }, [scope, onAddToProject, projectSlug, projectPaths])

  return (
    <FileViewer
      files={files}
      onSelect={onAttach}
      onClose={onClose}
      placeholder="Search documents…"
      emptyLabel="No documents"
      statusLabel={statusLabel}
      headerActions={headerActions}
      renderRowSuffix={renderRowSuffix}
      onArrowLeft={projectSlug ? () => setScope((s) => (s === "project" ? "global" : "project")) : undefined}
    />
  )
}
